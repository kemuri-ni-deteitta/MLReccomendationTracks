# Этот файл реализует класс SimpleRecommender — прототип рекомендательной системы музыкальных треков.
# Файл отвечает за:
# 1. Загрузку предвычисленных эмбеддингов аудио и метаданных треков (название, исполнитель, жанр).
# 2. Построение FAISS-индекса для быстрого поиска похожих треков по косинусному сходству эмбеддингов.
# 3. Генерацию рекомендаций: контентная схема (CLAP-эмбеддинги), поведенческая схема (ко-матрица совместных прослушиваний)
#    и гибридный подход (смешение контента и поведения).
# 4. Формирование HTML-фрагментов с встроенными <audio> плеерами, позволяющими слушать рекомендации прямо в чат-боте.

from typing import List, Optional, Tuple
import os
import re
import base64  # для кодирования аудио в base64

import numpy as np
import pandas as pd
import faiss                  # библиотека для быстрого поиска по векторным эмбеддингам
import torch
from transformers import ClapProcessor, ClapModel

# Регулярные выражения для разбора запроса пользователя:
_RE_ARTIST = re.compile(r'artist\s+"([^"]+)"', re.I)
_RE_SONG   = re.compile(r'song\s+"([^"]+)"',   re.I)
_RE_GENRE  = re.compile(r'genre\s+"([^"]+)"',  re.I)

class SimpleRecommender:
    """
    Основной класс рекомендателя.
    - Загружает эмбеддинги аудио и метаданные
    - Строит FAISS-индекс по эмбеддингам
    - Реализует методы recommend_indices, recommend, recommend_html
    """
    def __init__(self,
                 vector_path: str = "notebooks/embeddings/embedings_large/audio_vectors.npy",
                 title_path: str  = "notebooks/embeddings/embedings_large/audio_filenames.npy",
                 metadata_path: str = "audio_samples/fma_metadata/tracks.csv",
                 co_path: str     = "notebooks/embeddings/embedings_large/co_matrix.npy",
                 alpha: float     = 0.7):
        # 1) Загружаем эмбеддинги аудио (матрица shape=(N, dim))
        self.embeddings = np.load(vector_path).astype("float32")
        # 2) Загружаем список путей к аудио-файлам (N элементов)
        self.filepaths  = np.load(title_path)
        self.n, self.dim = self.embeddings.shape  # число треков и размерность эмбеддингов

        # 3) Загружаем CSV с метаданными: artist.name, track.title, track.genre_top
        self.metadata = pd.read_csv(metadata_path, index_col=0, header=[0,1])
        # Преобразуем каждый путь к названию "Исполнитель – Название трека"
        self.titles   = [self._id_to_title(p) for p in self.filepaths]
        # Приводим жанры к строкам, заменяем NaN на None
        self.genres   = [self._sanitize_genre(self._id_to_genre(p))
                         for p in self.filepaths]

        # 4) Загружаем ко-матрицу совместных прослушиваний (поведенческий модуль)
        self.co_matrix = None
        if os.path.exists(co_path):
            cm = np.load(co_path)
            # Проверяем размерность (N, N)
            if cm.shape == (self.n, self.n):
                self.co_matrix = cm.astype("float32")

        self.alpha     = alpha  # вес контентной части в гибридном алгоритме
        # 5) Инициализируем CLAP-модель и процессор для текстовой части
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model     = ClapModel.from_pretrained("laion/clap-htsat-unfused") \
                           .to("cuda" if torch.cuda.is_available() else "cpu")

        # 6) Нормализуем L2 и строим FAISS-индекс для IP (cosine similarity)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def _id_from_path(self, path: str) -> int:
        """Извлекает numerical ID трека из имени файла, например '066689.mp3' -> 66689"""
        return int(os.path.basename(path).split(".")[0])

    def _id_to_title(self, path: str) -> str:
        """Считает artist и track title по ID из metadata и объединяет в строку"""
        tid = self._id_from_path(path)
        try:
            artist = self.metadata.loc[tid, ("artist","name")]
            title  = self.metadata.loc[tid, ("track","title")]
            return f"{artist} – {title}"
        except KeyError:
            # fallback: выводим просто имя файла
            return os.path.basename(path)

    def _id_to_genre(self, path: str) -> Optional[str]:
        """Извлекает поля genre_top из metadata по ID трека"""
        tid = self._id_from_path(path)
        try:
            return self.metadata.loc[tid, ("track","genre_top")]
        except KeyError:
            return None

    def _sanitize_genre(self, g):
        """Преобразует NaN в None и все в строки"""
        if pd.isna(g): return None
        return str(g)

    def _embed_text(self, prompt: str) -> np.ndarray:
        """Получает текстовый эмбеддинг запроса через CLAP"""
        inp = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            vec = self.model.get_text_features(**inp)
        vec = vec.cpu().numpy().astype("float32")
        faiss.normalize_L2(vec)  # L2-нормализация для cosine
        return vec

    def recommend_indices(self,
                          prompt_parts: Tuple[str,str|None,str|None],
                          k: int=10
                         ) -> List[int]:
        """
        Основная логика рекомендаций:
        1) Формируем строку-запрос anchor = "{song} by {artist} in the genre {genre}".
        2) Получаем текстовый эмбеддинг, ищем ближайшие 100 по cosine.
        3) Фильтруем по жанру (если задан), удаляем seed-трек из списка.
        4) Берем k/2 топ «content-only» и k/2 «blended» (контент + co_matrix).
        5) Возвращаем индексы рекомендованных треков.
        """
        seed, artist, genre = prompt_parts
        # Формируем анкор
        anchor = f"{seed or ''} by {artist or ''} in the genre {genre or ''}".strip()
        q_vec, = self._embed_text(anchor)  # вектор запроса
        _, idx_all = self.index.search(q_vec.reshape(1,-1), 100)
        idx_all = idx_all[0].tolist()

        # Фильтрация по жанру
        if genre:
            filtered = [i for i in idx_all
                        if isinstance(self.genres[i], str)
                        and genre.lower() in self.genres[i].lower()]
            idx_all = filtered or idx_all

        # Убираем исходный трек из рекомендаций
        seed_norm = f"{artist or ''} – {seed}".lower() if artist else seed.lower()
        idx_all = [i for i in idx_all if seed_norm not in self.titles[i].lower()]

        # Выбор top k/2 pure content
        content, blended = [], []
        seen = set()
        for i in idx_all:
            if len(content) >= k//2: break
            content.append(i); seen.add(i)

        # Если есть co_matrix и задан artist, добавляем blended
        if self.co_matrix is not None and artist:
            try:
                seed_idx = next(i for i,t in enumerate(self.titles)
                                if artist.lower() in t.lower())
            except StopIteration:
                seed_idx = None
            if seed_idx is not None:
                scores = ( self.alpha * np.dot(self.embeddings[idx_all], q_vec) +
                          (1-self.alpha) * self.co_matrix[seed_idx][idx_all] )
                sorti = np.argsort(scores)[::-1]
                for si in sorti:
                    j = idx_all[si]
                    if j not in seen:
                        blended.append(j); seen.add(j)
                    if len(blended) >= k//2: break

        # Дополняем недостающие позиции
        extra = [i for i in idx_all if i not in seen]
        while len(content) < k//2 and extra:   content.append(extra.pop(0))
        while len(blended) < k//2 and extra:   blended.append(extra.pop(0))

        return (content + blended)[:k]

    def recommend(self,
                  prompt_parts: Tuple[str,str|None,str|None],
                  k: int=10
                 ) -> List[str]:
        """Устаревший метод: возвращает просто названия треков без плееров"""
        idxs = self.recommend_indices(prompt_parts, k)
        return [ self.titles[i] for i in idxs ]

    def recommend_html(self,
                       prompt_parts: Tuple[str,str|None,str|None],
                       k: int=10
                      ) -> str:
        """
        Возвращает HTML-фрагмент со встроенными <audio> плеерами.
        Плееры используют data URI с base64-кодированием mp3 из локальных файлов.
        """
        idxs = self.recommend_indices(prompt_parts, k)
        html_fragments = []
        for i in idxs:
            title = self.titles[i]
            path  = self.filepaths[i]
            # Читаем файл и кодируем в base64
            try:
                data = open(path, "rb").read()
                b64  = base64.b64encode(data).decode()
                src  = f"data:audio/mpeg;base64,{b64}"
            except FileNotFoundError:
                src  = ""  # если файл не найден, плеер будет пустым
            frag = f"""
            <div style="margin-bottom:1.5em">
              <b>{title}</b><br/>
              <audio controls preload="none" style="width:400px"
                     src="{src}"></audio>
            </div>
            """
            html_fragments.append(frag)
        return "\n".join(html_fragments)

# Создаем один глобальный экземпляр модели при загрузке модуля
_model = SimpleRecommender()

def _parse_message(msg: str) -> Tuple[str,str|None,str|None]:
    """Извлекает seed, artist и genre из сообщения пользователя"""
    a = _RE_ARTIST.search(msg)
    s = _RE_SONG.search(msg)
    g = _RE_GENRE.search(msg)
    clean = _RE_ARTIST.sub("", msg)
    clean = _RE_SONG.sub("", clean)
    clean = _RE_GENRE.sub("", clean)
    clean = clean.replace("Recommend me a track", "").strip(" ,.")
    return (
      s.group(1) if s else clean,
      a.group(1) if a else None,
      g.group(1) if g else None
    )

def chat_wrapper_html(message: str, history=None) -> str:
    """
    Функция-обертка для Gradio: принимает текст запроса, парсит его и возвращает HTML-фрагмент с рекомендациями.
    """
    seed, artist, genre = _parse_message(message)
    return _model.recommend_html((seed, artist, genre), k=10)
