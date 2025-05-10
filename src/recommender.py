from __future__ import annotations

"""src/recommender.py  –  CLAP + Collaborative hybrid recommender

• uses pre‑computed CLAP audio vectors (audio_vectors.npy)
• looks up readable titles from FMA metadata (tracks.csv)
• optionally blends a co‑occurrence matrix saved by 03_build_co_matrix.ipynb
• supports genre‑aware querying and a tiny MMR diversity step
"""

from typing import List, Optional
import os
import re

import numpy as np
import pandas as pd
import faiss
import torch
from transformers import ClapProcessor, ClapModel


class SimpleRecommender:
    """Lightweight, in‑memory recommender.

    Parameters
    ----------
    vector_path   : path to ``audio_vectors.npy`` (shape ≈ N×512)
    title_path    : path to ``audio_filenames.npy`` (N filenames)
    metadata_path : FMA ``tracks.csv`` for artist / title / genre lookup
    co_path       : optional ``co_matrix.npy`` – collaborative similarity
    """

    def __init__(
        self,
        vector_path  = "/home/ivan/PycharmProjects/MPr/notebooks/embeddings/audio_vectors.npy",
        title_path   = "/home/ivan/PycharmProjects/MPr/notebooks/embeddings/audio_filenames.npy",
        metadata_path= "/home/ivan/PycharmProjects/MPr/audio_samples/fma_metadata/tracks.csv",
        co_path      = "/home/ivan/PycharmProjects/MPr/embeddings/co_matrix.npy",
        alpha: float = 0.7,                    # blend weight content vs collaborative
    ) -> None:
        # ── load content embeddings ────────────────────────────────
        self.embeddings: np.ndarray = np.load(vector_path).astype("float32")
        self.filepaths: np.ndarray = np.load(title_path)
        self.dim = self.embeddings.shape[1]
        self.n = self.embeddings.shape[0]

        # ── load metadata (artist / title / genre) ────────────────
        self.metadata = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
        self.titles  = [self._filename_to_title(p) for p in self.filepaths]
        self.genres  = [self._filename_to_genre(p) for p in self.filepaths]

        # ── load optional co‑occurrence matrix ────────────────────
        self.co_matrix: Optional[np.ndarray]
        if co_path and os.path.exists(co_path):
            self.co_matrix = np.load(co_path).astype("float32")
            if self.co_matrix.shape[0] != self.n:
                print("[WARN] co_matrix size mismatch – ignoring collaborative part")
                self.co_matrix = None
        else:
            self.co_matrix = None
        self.alpha = alpha  # blend weight

        # ── init CLAP text encoder ────────────────────────────────
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model     = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(
            "cuda" if torch.cuda.is_available() else "cpu")

        # ── build FAISS index (cosine) ─────────────────────────────
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    # ──────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────
    def _filename_to_title(self, path: str) -> str:
        tid = int(os.path.basename(path).split(".")[0])
        try:
            artist = self.metadata.loc[tid, ("artist", "name")]
            title  = self.metadata.loc[tid, ("track", "title")]
            return f"{artist} – {title}"
        except Exception:
            return os.path.basename(path)

    def _filename_to_genre(self, path: str) -> Optional[str]:
        tid = int(os.path.basename(path).split(".")[0])
        try:
            return self.metadata.loc[tid, ("track", "genre_top")]
        except Exception:
            return None

    def _embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            vec = self.model.get_text_features(**inputs)
        vec = vec.cpu().numpy().astype("float32")
        faiss.normalize_L2(vec)
        return vec

    # ──────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────
    def recommend(self, query: str, genre: Optional[str] = None, k: int = 5) -> List[str]:
        """Return *k* track titles similar to *query* (optionally constrained to *genre*)."""

        # 1️⃣ embed query (prompt‑inject genre soft hint)
        full_query = f"{query} in the genre {genre}" if genre else query
        q_vec = self._embed_text(full_query)

        # 2️⃣ initial FAISS search (top‑50)
        _, idx = self.index.search(q_vec, 50)
        idx = idx[0].tolist()

        # 3️⃣ hard genre filter (if requested)
        if genre:
            idx = [i for i in idx if self.genres[i] and genre.lower() in self.genres[i].lower()]
            if not idx:
                idx = self.index.search(q_vec, k)[1][0].tolist()  # fallback to content only

        # 4️⃣ blend with collaborative score if available
        if self.co_matrix is not None:
            content_score = np.dot(self.embeddings[idx], q_vec.T).flatten()  # already normalized
            cf_score      = self.co_matrix[idx][:, idx].max(axis=1)          # best co‑occurrence per candidate
            final_score   = self.alpha*content_score + (1-self.alpha)*cf_score
            idx           = [idx[i] for i in np.argsort(final_score)[::-1]]

        # 5️⃣ simple diversity via cosine‑based MMR
        picked = []
        for cand in idx:
            if len(picked) == 0:
                picked.append(cand)
                continue
            sim_to_set = max(np.dot(self.embeddings[cand], self.embeddings[p]) for p in picked)
            if sim_to_set < 0.85:  # threshold
                picked.append(cand)
            if len(picked) == k:
                break
        if len(picked) < k:
            picked.extend(idx[: k - len(picked)])

        return [self.titles[i] for i in picked[:k]]


# single global instance ------------------------------------------------------
_model = SimpleRecommender()


def chat_wrapper(message: str, history=None) -> str:
    """Gradio‑style callable: (message, history) → str list."""
    # extract optional  "in the genre X" phrase
    genre_match = re.search(r"genre ['\"]?([^'\"]+)['\"]?", message, flags=re.I)
    genre = genre_match.group(1) if genre_match else None
    cleaned = re.sub(r"genre ['\"]?([^'\"]+)['\"]?", "", message, flags=re.I).strip()

    recs = _model.recommend(cleaned, genre=genre, k=5)
    return "\n".join(f"• {t}" for t in recs)
