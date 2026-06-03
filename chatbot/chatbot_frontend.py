import os
import uuid
import gradio as gr
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: State):
    system_prompt = (
        "You are an expert, highly intelligent, and objective AI assistant named 'Nexus', "
        "designed to match the professional tone and deep utility of state-of-the-art models like GPT.\n\n"
        "CORE DIRECTIVES:\n"
        "1. GROUNDED IN CONTEXT: Always base your responses strictly on the historical data and history provided.\n"
        "2. NO HALLUCINATIONS: Do not assume or fabricate facts.\n"
        "3. EXCELLENCE IN STRUCTURE: Provide insightful, comprehensive answers using Markdown styling, lists, or tables."
    )
    messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


def get_history_from_graph(thread_id: str):
    """Fetches and formats historical messages out of LangGraph's memory store."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    formatted_history = []
    if state and "messages" in state.values:
        messages = state.values["messages"]
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "assistant", "content": msg.content})
    return formatted_history

async def user_message(message, history):
    """Instantly adds user message to UI chat log and clears the text box."""
    return "", history + [{"role": "user", "content": message}]

async def stream_response(history, active_thread_id):
    """Streams the model response for the current active thread ID."""
    if not active_thread_id:
        active_thread_id = str(uuid.uuid4())
        
    config = {"configurable": {"thread_id": active_thread_id}}
    last_user_message = history[-1]["content"]
    input_state = {"messages": [("user", last_user_message)]}
    
    history.append({"role": "assistant", "content": ""})
    token_accumulator = ""
    
    async for event in graph.astream_events(input_state, config, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                token_accumulator += chunk.content
                history[-1]["content"] = token_accumulator
                yield history


def start_new_chat(sessions_list):
    """Triggers when 'Start New Chat' button is clicked."""
    new_id = f"Chat-{str(uuid.uuid4())[:8]}"  
    updated_sessions = [new_id] + sessions_list

    return [], updated_sessions, gr.update(choices=updated_sessions, value=new_id), new_id


def load_selected_chat(selected_id):
    """Triggers when a user picks an existing thread ID from the selector."""
    if not selected_id:
        return [], ""
    history = get_history_from_graph(selected_id)
    return history, selected_id


custom_theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

custom_css = """
body, .gradio-container { background: #0b0f19 !important; min-height: 100vh; }
.sidebar-panel { background: #0f172a !important; border-right: 1px solid #1e293b !important; padding: 20px !important; }
.chat-panel { padding: 20px !important; }
.bot { background: #1e293b !important; border: 1px solid rgba(99, 102, 241, 0.2) !important; border-radius: 8px !important; }
.user { background: #0f172a !important; border-radius: 8px !important; }
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="Nexus GPT") as demo:
    
    all_threads = gr.State(value=["Chat-Default"])
    active_thread_id = gr.State(value="Chat-Default")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar-panel"):
            gr.Markdown("### 🧠 Nexus Workspace")
            new_chat_btn = gr.Button("➕ Start New Chat", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("#### 📋 My Conversations")
            session_selector = gr.Dropdown(
                choices=["Chat-Default"], 
                value="Chat-Default", 
                label="Select Conversation Thread",
                interactive=True,
                container=False
            )
            
            gr.Markdown("<br><br><small>Each thread maintains completely isolated contextual memory states.</small>")
            
        with gr.Column(scale=3, elem_classes="chat-panel"):
            gr.Markdown("# 🔮 Nexus Expert Assistant")
            
            chatbot = gr.Chatbot(
                label="Conversation Window", 
                type="messages",
                elem_id="chat-box",
                height=550
            )
            
            with gr.Row():
                txt_input = gr.Textbox(
                    placeholder="Ask a technical query or continue your thread...",
                    container=False,
                    scale=8
                )
                submit_btn = gr.Button("Send", variant="secondary", scale=1)

    submit_event = submit_btn.click(
        fn=user_message, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot], queue=False
    ).then(
        fn=stream_response, inputs=[chatbot, active_thread_id], outputs=chatbot
    )
    
    txt_input.submit(
        fn=user_message, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot], queue=False
    ).then(
        fn=stream_response, inputs=[chatbot, active_thread_id], outputs=chatbot
    )

    new_chat_btn.click(
        fn=start_new_chat, 
        inputs=[all_threads], 
        outputs=[chatbot, all_threads, session_selector, active_thread_id]
    )

    session_selector.change(
        fn=load_selected_chat,
        inputs=[session_selector],
        outputs=[chatbot, active_thread_id]
    )

if __name__ == "__main__":
    demo.launch()
