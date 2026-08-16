"""
app.py - Streamlit entrypoint for Streamlit Cloud deployment.
Launches the Gradio UI as a subprocess and embeds it via iframe.
"""
import subprocess
import sys
import time
import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Chatbot RAG",
    page_icon="🤖",
    layout="wide",
)

GRADIO_PORT = 7860

# Launch the Gradio app in background (only once per session)
if "gradio_started" not in st.session_state:
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "app/ui.py"],
        env=env,
    )
    st.session_state["gradio_started"] = True
    st.session_state["gradio_pid"] = proc.pid
    # Give Gradio a moment to start
    time.sleep(4)

# Embed Gradio inside Streamlit via iframe
components.iframe(
    src=f"http://localhost:{GRADIO_PORT}",
    height=800,
    scrolling=True,
)
