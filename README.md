# LangGraph Chatbot with Persistent Memory

A production-ready conversational AI application built using **LangGraph** for state management, **FastAPI** for a robust asynchronous backend, and **Gradio** for a seamless, event-driven user interface. The chatbot utilizes the `gemini-2.5-flash` model and features persistent cross-request memory tracking via a thread-safe checkpointer.

---

## 🌐 Live Demo

You can try out the fully functional, stateful chatbot live in your browser here:  
**[Launch Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/rraunak41/langgraph-memory-chatbot)**

## Features

* **Stateful Conversations:** Built on top of LangGraph's `StateGraph` and `MessagesState` architectures to properly manage message arrays.
* **Persistent Memory:** Uses LangGraph's `MemorySaver` checkpointer to retain session context and conversation threads across independent API calls.
* **Decoupled Architecture:** Features a dedicated FastAPI backend server acting as an API gateway and an event-driven Gradio frontend UI.
* **Hot-Reload Ready:** Fully configured to support real-time UI hot-reloads during active development.

---

## Tech Stack

* **Core Framework:** LangGraph (`StateGraph`, `MemorySaver`)
* **LLM Engine:** LangChain Google GenAI (`gemini-2.5-flash`)
* **Backend Framework:** FastAPI / Uvicorn (ASGI)
* **Frontend UI:** Gradio (ChatInterface)
* **Environment Management:** Python Dotenv (`python-dotenv`)

---

## Repository Structure

```text
└── chatbot/
    ├── .env                     # Local environment variables (API Keys)
    ├── chatbot_backend.py       # FastAPI application & LangGraph compilation
    ├── chatbot_frontend.py      # Gradio UI client interface
    └── requirements.txt         # Project dependencies
