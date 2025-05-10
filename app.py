"""
Launches a local chat UI at http://localhost:7860
"""
from src.recommender import chat_wrapper
import gradio as gr

demo = gr.ChatInterface(
    fn=chat_wrapper,
    title="Music‑track recommender (demo)",
    examples=[["Recommend me a track in the style of artist ""Lightning Bolt"", song ""Dracula Mountain"", in the genre ""Rock"]],
)

if __name__ == "__main__":
    demo.launch()
