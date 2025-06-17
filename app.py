from fastapi import FastAPI
import gradio as gr
from main import get_answer
from main import get_answer_with_history
from gradio.themes import Soft

app = FastAPI()
gradio_app = gr.Blocks(theme=Soft())

def chat_fn(user_message, chat_history):
    if chat_history is None:
        chat_history = []
    response = get_answer(user_message)
    chat_history.append((user_message, response))
    return chat_history, chat_history

with gr.Blocks() as gradio_app:
    chatbot = gr.Chatbot()
    user_input = gr.Textbox()
    user_input.submit(chat_fn, [user_input, chatbot], [chatbot, chatbot])
    user_input.submit(lambda: "", None, user_input)

app = gr.mount_gradio_app(app, gradio_app, path="/chat")