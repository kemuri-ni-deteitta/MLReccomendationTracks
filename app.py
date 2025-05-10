from src.recommender import chat_wrapper
import gradio as gr

GENRES = ["Rock","Electronic","Hip-Hop","Instrumental","International","Pop","Folk","Experimental"]

def build_prompt(artist, song, genre):
    parts = []
    if artist: parts.append(f'artist "{artist}"')
    if song:   parts.append(f'song "{song}"')
    if genre:  parts.append(f'genre "{genre}"')
    return "Recommend me a track in the style of " + ", ".join(parts)

def respond(artist, song, genre, history):
    prompt = build_prompt(artist, song, genre)
    reply = chat_wrapper(prompt, history)
    history = history + [(prompt, reply)]
    return history

with gr.Blocks() as demo:
    gr.Markdown("## 🎵 Music-track Recommender")
    # Only the `height` argument is accepted:
    chatbot = gr.Chatbot(label="Recommendations", height=800)

    with gr.Row():
        artist_in = gr.Textbox(label="Artist", placeholder="e.g. Radiohead")
        song_in   = gr.Textbox(label="Song",   placeholder="e.g. Karma Police")
        genre_in  = gr.Dropdown(choices=GENRES, label="Genre", value="Rock")
        btn       = gr.Button("🎶 Recommend")

    btn.click(respond, [artist_in, song_in, genre_in, chatbot], chatbot)

if __name__ == "__main__":
    demo.launch()
