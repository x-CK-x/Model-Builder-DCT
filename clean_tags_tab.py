import gradio as gr
import tag_cleaner_app

__all__ = ["add_clean_tags_tabs"]

def add_clean_tags_tabs():
    with gr.Tab("Tag Cleaner Utility"):
        tag_cleaner_app.demo.render()
