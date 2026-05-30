import gradio as gr
import requests
import uuid

# URL of your local FastAPI backend
BACKEND_URL = "http://localhost:8000/chat"

# Generate a unique thread ID for tracking context memory
session_thread_id = str(uuid.uuid4())

def predict(message, history):
    payload = {
        "message": message,
        "thread_id": session_thread_id
    }
    try:
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("reply", "No reply payload found.")
        return "Backend Server Error."
    except Exception as e:
        return f"Could not connect to backend: {e}"

# Event-driven chat interface (Does not rerun your code on click!)
demo = gr.ChatInterface(
    fn=predict, 
    title="LangGraph Chatbot (Persistent Memory)",
    textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7)
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)