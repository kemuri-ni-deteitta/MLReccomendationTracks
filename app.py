from src.recommender import chat_wrapper
import gradio as gr

# Static genre list
GENRES = [
    "Rock", "Electronic", "Hip-Hop", "Instrumental", "International",
    "Pop", "Folk", "Experimental"
]

def build_prompt(artist, song, genre):
    parts = []
    if artist:
        parts.append(f'artist "{artist}"')
    if song:
        parts.append(f'song "{song}"')
    if genre:
        parts.append(f'genre "{genre}"')
    return "Recommend me a track in the style of " + ", ".join(parts)

def respond(artist, song, genre, history):
    # 1️⃣ build the text prompt
    prompt = build_prompt(artist, song, genre)
    # 2️⃣ get the reply from your recommender
    reply = chat_wrapper(prompt, history)
    # 3️⃣ append to the chat history [(user, bot), ...]
    history = history + [(prompt, reply)]
    return history

with gr.Blocks(theme="default") as demo:
    gr.Markdown("## 🎵 Music-track Recommender")
    chatbot = gr.Chatbot(label="Recommendations")
    with gr.Row():
        artist_in = gr.Textbox(label="Artist", placeholder="e.g. Radiohead")
        song_in   = gr.Textbox(label="Song",   placeholder="e.g. Karma Police")
        genre_in  = gr.Dropdown(choices=GENRES, label="Genre", value="Rock")
        btn       = gr.Button("🎶 Recommend")

    # Wire the button to our respond function, updating the chatbot
    btn.click(respond, [artist_in, song_in, genre_in, chatbot], chatbot)

if __name__ == "__main__":
    demo.launch()
