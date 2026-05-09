# Personality Chatbot Process Document

## 1. Purpose

This document explains how the `Personality Chatbot` project works end to end, from startup to response generation to data persistence. It is meant to help with:

- onboarding
- debugging
- explaining the project in reviews or demos
- making future changes without breaking the flow

## 2. Project Summary

The project is a local multi-persona chatbot with:

- a React frontend for chat UI
- a FastAPI backend for API routes and orchestration
- Ollama as the local LLM runtime
- local JSON files for memory, session history, and audit logs

The chatbot supports three personas:

- `ankit`: casual friend-style replies
- `mentor`: direct, strict coaching replies
- `therapist`: calm, reflective replies

## 3. Main Components

### Frontend

File:

- `frontend/persona-ui/src/App.js`

Responsibilities:

- load persona list
- load existing chat sessions
- load message history for a selected session
- create a new chat session
- send user messages to the backend
- render persona list, history list, and chat bubbles

### Backend API

File:

- `main1.py`

Responsibilities:

- expose HTTP endpoints
- normalize persona selection
- create and fetch sessions
- load history from storage
- call the response engine
- return structured chat responses to the frontend

### Response Engine

File:

- `engine.py`

Responsibilities:

- clean and normalize input
- predict emotion
- run safety checks
- build persona prompt and context
- call Ollama
- validate persona consistency
- persist messages and metadata

### Configuration

Files:

- `config.py`
- `personas.py`

Responsibilities:

- define file paths and limits
- define persona names, taglines, and prompt instructions

### Memory and Logs

Primary runtime files:

- `memory/user_profile.json`
- `memory/persona_chat_logs.json`
- `chatbot_conversations.json`

Support/legacy notebook helper:

- `notebook/memory_manager.py`

Note:

`notebook/memory_manager.py` is a simple experimental helper and is not the main runtime memory path for the current app flow. The production runtime logic is handled in `engine.py`.

## 4. Startup Process

The app requires three pieces to be available:

1. Ollama server
2. FastAPI backend
3. React frontend

### Normal startup order

1. Start Ollama:

```powershell
ollama serve
```

2. Start backend from project root:

```powershell
python -m uvicorn main1:app --host 127.0.0.1 --port 8000
```

3. Start frontend:

```powershell
cd frontend\persona-ui
npm start
```

### Runtime endpoints

- Frontend: `http://localhost:3000`
- Backend: `http://127.0.0.1:8000`
- Ollama API: `http://127.0.0.1:11434`

## 5. API Process

The backend exposes the following routes:

- `GET /`
- `GET /personas`
- `GET /chat-sessions/{persona}`
- `POST /chat-sessions`
- `GET /chat-history/{persona}`
- `GET /chat-history/{persona}/{conversation_id}`
- `POST /chat`

### Route purposes

`GET /`

- health check
- returns whether the backend is running

`GET /personas`

- returns persona names and taglines for the UI

`GET /chat-sessions/{persona}`

- returns saved conversation session metadata for one persona and one user

`POST /chat-sessions`

- creates a new empty conversation session

`GET /chat-history/...`

- loads saved message history for an existing session

`POST /chat`

- main route for generating a persona reply

## 6. End-to-End Chat Flow

### A. Initial page load

When the frontend opens:

1. `App.js` requests `GET /personas`
2. the first persona is selected automatically
3. frontend requests `GET /chat-sessions/{persona}?user_id=ankit`
4. if sessions exist, frontend loads the most recent conversation
5. if no session exists, frontend creates one with `POST /chat-sessions`

### B. User sends a message

When the user clicks `Send` or presses Enter:

1. frontend ensures there is an active `conversation_id`
2. frontend optimistically adds the user message to UI state
3. frontend sends `POST /chat` with:
   - `message`
   - `persona`
   - `user_id`
   - `conversation_id`
4. backend loads or creates the correct conversation
5. backend calls `generate_response(...)` in `engine.py`
6. backend returns:
   - `response`
   - `emotion`
   - `confidence`
   - `safety`
   - `conversation_id`
   - `session`
7. frontend appends the bot reply and refreshes session metadata

## 7. Internal Response Generation Process

The main logic lives in `generate_response()` inside `engine.py`.

### Step 1. Normalize and prepare

- clean the user text
- normalize the persona key
- ensure a conversation exists

### Step 2. Emotion prediction

The engine calls `predict_emotion()`:

- if the transformer model exists, it predicts emotion and confidence
- if the model is unavailable, it falls back to `neutral`

Model path is defined in `config.py`:

- `notebook/models/emotion_distilbert`
- `notebook/models/emotion_label_encoder.pkl`

### Step 3. Input safety check

The engine calls `analyze_safety()` on the user message.

It looks for:

- self-harm signals
- violence
- hate or harassment
- sexual minor content
- profanity

If the input is blocked:

- the model is not called
- a safe guardrail reply is returned immediately
- the blocked event is still persisted to logs

### Step 4. Persona prompt construction

The engine pulls the persona prompt from `personas.py`.

It adds:

- persona-specific system instructions
- detected emotion
- a style anchor
- safety instructions
- recent chat context

### Step 5. Context window construction

The engine calls `build_context(chat_history)`.

Behavior:

- only the most recent messages are used
- maximum messages come from `MAX_CONTEXT_MESSAGES` in `config.py`
- current value is `8`

### Step 6. Ollama generation

The engine sends the prompt to:

- model: `qwen2.5:1.5b`

via:

- `ollama.chat(...)`

### Step 7. Response cleanup and persona validation

After generation, the engine:

- removes stray `ASSISTANT` or `USER` prefixes
- trims whitespace
- reduces the reply to a compact form
- checks for assistant-like phrasing
- checks whether the reply still matches the selected persona

If the first reply feels off-persona:

- the engine makes a second Ollama call with stricter persona instructions

### Step 8. Output safety check

The reply is checked again with `analyze_safety()`.

If the output is unsafe:

- the generated reply is replaced
- a safer persona-appropriate fallback is returned
- the guardrail action is marked as `replaced_output`

### Step 9. Persistence

The final user turn and assistant turn are saved through `persist_turn()`.

This updates:

- in-memory chat history for the active runtime session
- `memory/user_profile.json`
- `memory/persona_chat_logs.json`
- `chatbot_conversations.json`

## 8. Data Persistence Process

### A. `memory/user_profile.json`

Purpose:

- global user memory summary

Current data includes:

- `history`
- `emotion_counts`
- `last_seen`

Important note:

This file currently behaves more like a lightweight global profile tracker than a rich long-term memory system.

### B. `memory/persona_chat_logs.json`

Purpose:

- structured per-persona conversation storage

Structure conceptually:

- persona
- user id
- conversations
- messages inside each conversation

Each conversation stores:

- `id`
- `title`
- `created_at`
- `updated_at`
- `messages`

Each message can store:

- `role`
- `content`
- `timestamp`
- `detected_emotion`
- `confidence`
- `safety`

### C. `chatbot_conversations.json`

Purpose:

- audit-style conversation log for the whole application

Each log item stores:

- timestamp
- user id
- persona
- conversation id
- detected emotion
- confidence
- user message
- reply
- safety metadata

## 9. Session Management Process

Session management is handled in `engine.py` and exposed through `main1.py`.

### Key functions

- `create_conversation()`
- `ensure_conversation()`
- `list_persona_conversations()`
- `get_conversation_detail()`
- `load_persona_history()`
- `save_persona_history()`

### How sessions work

- every conversation gets a short generated id
- a conversation title is derived from the first user message if possible
- sessions are sorted by `updated_at`
- the backend also keeps an in-memory `sessions` cache during runtime

Important distinction:

- the JSON files are the durable source of truth
- the in-memory `sessions` dictionary in `main1.py` is only a runtime cache

## 10. Persona Process

Persona behavior is controlled by `TEAM_PERSONAS` in `personas.py`.

Each persona defines:

- `name`
- `tagline`
- `prompt`

### Current persona styles

`ankit`

- friend-like
- casual
- playful
- emotionally present

`mentor`

- blunt
- focused
- accountability-driven

`therapist`

- calm
- validating
- reflective

When changing personas, update:

- the prompt text in `personas.py`
- any frontend copy in `frontend/persona-ui/src/App.js`
- optional persona-specific safety fallback behavior in `engine.py`

## 11. Safety Process

Safety is handled in two layers.

### Input safety

Runs before the LLM call.

Goal:

- block clearly unsafe prompts
- de-escalate dangerous content early

### Output safety

Runs after the LLM call.

Goal:

- prevent unsafe model text from reaching the user

### Safety outcomes

Common actions:

- `allow`
- `block_input`
- `replaced_output`

## 12. Frontend State Process

The frontend maintains:

- selected persona
- sessions by persona
- active conversation by persona
- cached messages
- current message list
- loading and sending state

Important frontend behaviors:

- messages are cached by `persona:conversationId`
- the chat UI updates optimistically before backend reply returns
- persona switching loads that persona's own session history

## 13. Change Process

When making updates, use this sequence.

### If changing persona behavior

1. update `personas.py`
2. review persona validation in `engine.py`
3. test at least one conversation per persona

### If changing safety behavior

1. update pattern lists in `engine.py`
2. test blocked input cases
3. test replaced output cases
4. verify logs still capture safety metadata

### If changing storage format

1. update normalization helpers in `engine.py`
2. keep backward compatibility where possible
3. verify existing chat history still loads

### If changing frontend message flow

1. update `frontend/persona-ui/src/App.js`
2. test new chat creation
3. test history switching
4. test send-message and reload behavior

## 14. Known Operational Notes

- The backend depends on Ollama being available locally.
- If Ollama is down, chat generation will fail even if the frontend loads.
- Emotion detection gracefully falls back to `neutral` if the model files are missing.
- The notebook helper files are not the main serving path for the live app.
- The current architecture uses JSON files, so it is best suited for local development and demos rather than multi-user production deployment.

## 15. Suggested Future Improvements

- move persistent storage from JSON files to a database
- separate runtime memory from analytics logs
- add formal API schemas and endpoint documentation
- add tests for session creation, safety behavior, and persistence
- add a dedicated service layer instead of concentrating most orchestration in `engine.py`
- make the frontend API base configurable through environment variables

## 16. Quick Reference

### Main files

- `main1.py`: API routes
- `engine.py`: chat orchestration and persistence
- `personas.py`: persona definitions
- `config.py`: paths and limits
- `frontend/persona-ui/src/App.js`: frontend chat flow

### Main data files

- `memory/user_profile.json`
- `memory/persona_chat_logs.json`
- `chatbot_conversations.json`

### Core dependency chain

Frontend -> FastAPI -> Engine -> Ollama -> Safety/Persistence -> Frontend response
