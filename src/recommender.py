from typing import List, Optional, Tuple
import os
import re

import numpy as np
import pandas as pd
import faiss
import torch
from transformers import ClapProcessor, ClapModel

_RE_ARTIST = re.compile(r"artist\s+\"([^\"]+)\"", re.I)
_RE_SONG = re.compile(r"song\s+\"([^\"]+)\"", re.I)
_RE_GENRE = re.compile(r"genre\s+\"([^\"]+)\"", re.I)


class SimpleRecommender:
    def __init__(self,
                 vector_path="/home/ivan/PycharmProjects/MPr/notebooks/embeddings/embedings_large/audio_vectors.npy",
                 title_path="/home/ivan/PycharmProjects/MPr/notebooks/embeddings/embedings_large/audio_filenames.npy",
                 metadata_path="/home/ivan/PycharmProjects/MPr/audio_samples/fma_metadata/tracks.csv",
                 co_path="/home/ivan/PycharmProjects/MPr/notebooks/embeddings/embedings_large/co_matrix.npy",
                 alpha=0.7):

        self.embeddings = np.load(vector_path)
        assert self.embeddings.ndim == 2 and self.embeddings.dtype != object
        self.embeddings = self.embeddings.astype("float32")
        self.filepaths = np.load(title_path)
        self.n, self.dim = self.embeddings.shape

        self.metadata = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
        self.titles = [self._id_to_title(p) for p in self.filepaths]

        # Genre sanitation: None instead of NaN, force strings
        self.genres = []
        for p in self.filepaths:
            genre = self._id_to_genre(p)
            if pd.isna(genre):
                self.genres.append(None)
            else:
                self.genres.append(str(genre))

        self.co_matrix: Optional[np.ndarray] = None
        if co_path and os.path.exists(co_path):
            cm = np.load(co_path)
            if cm.shape == (self.n, self.n):
                self.co_matrix = cm.astype("float32")
        self.alpha = alpha

        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(
            "cuda" if torch.cuda.is_available() else "cpu")

        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def _id_from_path(self, path: str) -> int:
        return int(os.path.basename(path).split(".")[0])

    def _id_to_title(self, path: str) -> str:
        tid = self._id_from_path(path)
        try:
            artist = self.metadata.loc[tid, ("artist", "name")]
            title = self.metadata.loc[tid, ("track", "title")]
            return f"{artist} – {title}"
        except Exception:
            return os.path.basename(path)

    def _id_to_genre(self, path: str) -> Optional[str]:
        tid = self._id_from_path(path)
        try:
            return self.metadata.loc[tid, ("track", "genre_top")]
        except Exception:
            return None

    def _embed_text(self, prompt: str) -> np.ndarray:
        inp = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            vec = self.model.get_text_features(**inp)
        vec = vec.cpu().numpy().astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def recommend(self, prompt_parts: Tuple[str, str | None, str | None], k: int = 10) -> List[str]:
        seed, artist, genre = prompt_parts

        anchor = f"{seed or ''} by {artist or ''} in the genre {genre or ''}".strip()
        q_vec = self._embed_text(anchor)

        _, idx_all = self.index.search(q_vec, 100)
        idx_all = idx_all[0].tolist()

        # Safe genre filtering
        if genre:
            idx_all = [
                i for i in idx_all
                if isinstance(self.genres[i], str) and genre.lower() in self.genres[i].lower()
            ]
            if not idx_all:
                idx_all = self.index.search(q_vec, 100)[1][0].tolist()

        seed_norm = f"{artist or ''} – {seed}".lower() if artist else seed.lower()
        idx_all = [i for i in idx_all if seed_norm not in self.titles[i].lower()]

        seed_idx = None
        if self.co_matrix is not None and artist:
            try:
                seed_idx = next(i for i, t in enumerate(self.titles) if artist.lower() in t.lower())
            except StopIteration:
                seed_idx = None

        seen = set()
        content_only, blended = [], []

        for i in idx_all:
            if len(content_only) >= k // 2:
                break
            if i not in seen:
                content_only.append(i)
                seen.add(i)

        if seed_idx is not None and self.co_matrix is not None:
            scores = self.alpha * np.dot(self.embeddings[idx_all], q_vec.T).flatten()
            scores += (1 - self.alpha) * self.co_matrix[seed_idx][idx_all]
            sorted_idx = np.argsort(scores)[::-1]
            for si in sorted_idx:
                idx = idx_all[si]
                if idx not in seen:
                    blended.append(idx)
                    seen.add(idx)
                if len(blended) >= k // 2:
                    break

        extra = [i for i in idx_all if i not in seen]
        while len(content_only) < k // 2 and extra:
            content_only.append(extra.pop(0))
        while len(blended) < k // 2 and extra:
            blended.append(extra.pop(0))

        final = (
            [f"{self.titles[i]}  [from: content-only]" for i in content_only] +
            [f"{self.titles[i]}  [from: blended (02+03)]" for i in blended]
        )
        return final[:k]

# Global model instance
_model = SimpleRecommender()

def _parse_message(msg: str) -> Tuple[str, str | None, str | None]:
    artist = _RE_ARTIST.search(msg)
    song = _RE_SONG.search(msg)
    genre = _RE_GENRE.search(msg)

    cleaned = _RE_ARTIST.sub("", msg)
    cleaned = _RE_SONG.sub("", cleaned)
    cleaned = _RE_GENRE.sub("", cleaned)
    cleaned = cleaned.replace("Recommend me a track", "").strip(" ,.")

    return (
        song.group(1) if song else cleaned,
        artist.group(1) if artist else None,
        genre.group(1) if genre else None
    )

def chat_wrapper(message: str, history=None) -> str:
    seed, artist, genre = _parse_message(message)
    recs = _model.recommend((seed, artist, genre), k=10)
    return "\n".join(f"• {t}" for t in recs)
