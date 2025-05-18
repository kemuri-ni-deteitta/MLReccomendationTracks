# server.py
# Этот файл запускает простой Flask-сервер, который отдаёт статические mp3-файлы
# из локальной директории по HTTP-запросу. Используется вместе с Gradio UI,
# чтобы плееры в браузере могли подгружать нужные треки.

from flask import Flask, send_from_directory
import os

# Создаём экземпляр Flask-приложения
app = Flask(__name__)

# Директория, где хранятся ваши аудиофайлы (.mp3).
# Можно указать либо fma_small, либо fma_large в зависимости от используемой коллекции.
AUDIO_DIR = "/home/ivan/PycharmProjects/MPr/audio_samples/fma_small"  # or fma_large

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """
    Маршрут отдачи аудио-файла.
    Клиент отправляет GET-запрос на /audio/<имя_файла>.mp3,
    и этот хендлер ищет файл в AUDIO_DIR и возвращает его.
    Опция conditional=True позволяет использовать заголовки If-Modified-Since
    для кэширования в браузере.
    """
    return send_from_directory(AUDIO_DIR, filename, conditional=True)

if __name__ == "__main__":
    # Запускаем Flask-сервер на порту 8000 без режима отладки.
    # Gradio UI (обычно на порту 7860) будет обращаться к этому серверу
    # чтобы подгружать треки в аудио-плееры.
    app.run(port=8000, debug=False)
