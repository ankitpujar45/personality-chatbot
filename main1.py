from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import (
    create_conversation,
    ensure_conversation,
    generate_response,
    get_conversation_detail,
    list_persona_conversations,
    load_memory,
    load_persona_history,
    normalize_persona_key,
)
from personas import TEAM_PERSONAS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}


class ChatRequest(BaseModel):
    message: str
    persona: str
    user_id: str
    conversation_id: Optional[str] = None


class CreateChatSessionRequest(BaseModel):
    persona: str
    user_id: str
    title: Optional[str] = "New chat"


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/personas")
def get_personas():
    return {
        key: {
            "name": value["name"],
            "tagline": value["tagline"],
        }
        for key, value in TEAM_PERSONAS.items()
    }


@app.get("/chat-sessions/{persona}")
def get_chat_sessions(persona: str, user_id: str = "ankit"):
    persona_key = normalize_persona_key(persona)
    sessions_list = list_persona_conversations(user_id, persona_key)
    return {"persona": persona_key, "user_id": user_id, "sessions": sessions_list}


@app.post("/chat-sessions")
def create_chat_session(req: CreateChatSessionRequest):
    persona_key = normalize_persona_key(req.persona)
    session = create_conversation(req.user_id, persona_key, req.title or "New chat")
    return {"persona": persona_key, "user_id": req.user_id, "session": session}


@app.get("/chat-history/{persona}")
def get_latest_chat_history(persona: str, user_id: str = "ankit"):
    persona_key = normalize_persona_key(persona)
    history = load_persona_history(user_id, persona_key)
    sessions_list = list_persona_conversations(user_id, persona_key)
    active_session = sessions_list[0] if sessions_list else None
    return {
        "persona": persona_key,
        "user_id": user_id,
        "session": active_session,
        "messages": history,
    }


@app.get("/chat-history/{persona}/{conversation_id}")
def get_chat_history(persona: str, conversation_id: str, user_id: str = "ankit"):
    persona_key = normalize_persona_key(persona)
    detail = get_conversation_detail(user_id, persona_key, conversation_id)
    return {
        "persona": persona_key,
        "user_id": user_id,
        "session": detail["session"],
        "messages": detail["messages"],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    persona_key = normalize_persona_key(req.persona)
    session_meta = ensure_conversation(req.user_id, persona_key, req.conversation_id)
    conversation_id = session_meta["id"]
    session_key = f"{req.user_id}_{persona_key}_{conversation_id}"

    if session_key not in sessions:
        sessions[session_key] = load_persona_history(req.user_id, persona_key, conversation_id)

    mem = load_memory()

    emotion, reply, history, mem, session_meta = generate_response(
        req.message,
        sessions[session_key],
        mem,
        persona_key=persona_key,
        user_id=req.user_id,
        conversation_id=conversation_id,
    )

    sessions[session_key] = history
    last_message = history[-1] if history else {}

    return {
        "response": reply,
        "emotion": emotion,
        "confidence": last_message.get("confidence"),
        "safety": last_message.get("safety"),
        "conversation_id": session_meta["id"],
        "session": session_meta,
    }
