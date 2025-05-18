# -*- coding: utf-8 -*-
"""
Файл: app.py
Описание:
Приложение запускает веб-интерфейс (через Gradio), в котором пользователь может указать исполнителя, название песни и жанр.
Далее формируется запрос к рекомендательной системе и выводуются рекомендации с встроенными аудиоплеерами.
"""
import gradio as gr
from src.recommender import chat_wrapper_html

# Список доступных жанров для выпадающего списка
GENRES = [
    "Rock", "Electronic", "Hip-Hop", "Instrumental",
    "International", "Pop", "Folk", "Experimental", "Classical"
]


def build_prompt(artist: str, song: str, genre: str) -> str:
    """
    Формирует текстовый запрос (prompt) для модели на основе введённых пользователем параметров.
    Пример результата: "Recommend me a track in the style of artist "Radiohead", song "Karma Police", genre "Rock""

    :param artist: Имя исполнителя
    :param song: Название трека
    :param genre: Жанр
    :return: Сформированная строка-запрос
    """
    parts = []
    # Добавляем фразы-части запроса только если пользователь ввёл соответствующее поле
    if artist:
        parts.append(f'artist "{artist}"')
    if song:
        parts.append(f'song "{song}"')
    if genre:
        parts.append(f'genre "{genre}"')
    # Собираем окончательную строку
    return "Recommend me a track in the style of " + ", ".join(parts)


def respond(artist: str, song: str, genre: str) -> str:
    """
    Функция-обработчик нажатия кнопки "Recommend".
    1. Строит prompt
    2. Вызывает рекомендательную систему chat_wrapper_html
    3. Возвращает HTML-фрагмент с рекомендациями и аудиоплеерами

    :param artist: Текст из поля Artist
    :param song: Текст из поля Song
    :param genre: Выбранный жанр
    :return: HTML-контент с рекомендациями
    """
    # Строим запрос для модели
    prompt = build_prompt(artist, song, genre)
    # Вызываем рекомендатель и возвращаем HTML
    return chat_wrapper_html(prompt, None)


# === Gradio UI ===
# Определяем интерфейс в режиме Blocks для гибкой компоновки элементов
with gr.Blocks(css="""
  /* Настраиваем внешний вид кнопки через elem_id */
  #recc-btn {
    margin-top: 1em;
    width: 200px;
  }
""") as demo:
    # Заголовок страницы
    gr.Markdown("## 🎵 Music-track Recommender")

    # Вертикальный ряд полей ввода и кнопки
    with gr.Row():
        artist_in = gr.Textbox(label="Artist", placeholder="e.g. Radiohead")
        song_in   = gr.Textbox(label="Song",   placeholder="e.g. Karma Police")
        genre_in  = gr.Dropdown(choices=GENRES, label="Genre", value="Rock")
        # Кнопка запуска рекомендаций
        btn = gr.Button("🎶 Recommend", elem_id="recc-btn")

    # HTML-вывод для рекомендаций (с <audio> плеерами)
    html_out = gr.HTML()

    # Привязываем событие click к кнопке
    btn.click(
        fn=respond,
        inputs=[artist_in, song_in, genre_in],
        outputs=html_out
    )

# Запуск приложения при прямом исполнении файла
if __name__ == "__main__":
    demo.launch()
