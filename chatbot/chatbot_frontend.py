import os
import uuid
import gradio as gr
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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


uncompiled_graph = workflow.compile()



async def get_history_from_graph(thread_id: str, saver: AsyncSqliteSaver):
    """Safely fetches and formats historical messages from the Async SQLite checkpointer."""
    if not thread_id:
        return []
        
    config = {"configurable": {"thread_id": str(thread_id)}}
    
    try:
      
        state = await saver.aget_state(config)
    except Exception:
        return []
        
    formatted_history = []
    
    if state and hasattr(state, "values") and state.values and "messages" in state.values:
        messages = state.values["messages"]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": getattr(msg, "content", "")})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "assistant", "content": getattr(msg, "content", "")})
                
    return formatted_history


async def user_message(message, history):
    if history is None:
        history = []
    if not message or str(message).strip() == "":
        return "", history
        
    return "", history + [{"role": "user", "content": str(message)}]

async def stream_response(history, active_thread_id):
    if not history or len(history) == 0:
        yield history
        return

    if not active_thread_id:
        active_thread_id = "Chat-Default"
        
    config = {"configurable": {"thread_id": str(active_thread_id)}}
    
    try:
        last_user_message = history[-1].get("content", "") if isinstance(history[-1], dict) else history[-1][1]
    except Exception:
        yield history
        return

    input_state = {"messages": [("user", str(last_user_message))]}
    history.append({"role": "assistant", "content": ""})
    token_accumulator = ""
    
    async with AsyncSqliteSaver.from_conn_string("memory.db") as saver:
        graph = workflow.compile(checkpointer=saver)
        try:
            async for event in graph.astream_events(input_state, config, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token_accumulator += str(chunk.content)
                        history[-1]["content"] = token_accumulator
                        yield history
        except Exception as e:
            history[-1]["content"] = f"⚠️ [Streaming Core Exception]: {str(e)}"
            yield history


def start_new_chat(sessions_list):
    if not sessions_list:
        sessions_list = ["Chat-Default"]
    new_id = f"Chat-{str(uuid.uuid4())[:8]}"
    updated_sessions = [new_id] + sessions_list
    return [], updated_sessions, gr.update(choices=updated_sessions, value=new_id), new_id


async def load_selected_chat(selected_id):
    if not selected_id:
        return [], "Chat-Default"
    
    async with AsyncSqliteSaver.from_conn_string("memory.db") as saver:
        history = await get_history_from_graph(selected_id, saver)
    return history, selected_id


custom_theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

custom_css = """
body, .gradio-container { background: #0b0f19 !important; min-height: 100vh; }
.sidebar-panel { background: #0f172a !important; border-right: 1px solid #1e293b !important; padding: 20px !important; }
.chat-panel { padding: 20px !important; }
.bot { background: #1e293b !important; border: 1px solid rgba(99, 102, 241, 0.2) !important; border-radius: 8px !important; }
.user { background: #0f172a !important; border-radius: 8px !important; }
"""

with gr.Blocks(title="Nexus GPT") as demo:
    
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
            
            gr.Markdown("<br><br><small>Each thread maintains completely isolated database memory tables.</small>")
            
 
        with gr.Column(scale=3, elem_classes="chat-panel"):
            gr.Markdown("# 🔮 Nexus Persistent Assistant")
            
            chatbot = gr.Chatbot(
                label="Conversation Window", 
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
    demo.launch(theme=custom_theme, css=custom_css)
