# Personality Chatbot

Local chatbot project with:

- a FastAPI backend
- a React frontend
- Ollama for local LLM responses

## One-Command Run

From the project root:

```powershell
.\run-project.ps1
```

This script:

- starts Ollama if `http://127.0.0.1:11434/api/tags` is not already reachable
- starts the FastAPI backend if `http://127.0.0.1:8000` is not already reachable
- starts the React frontend if `http://localhost:3000` is not already reachable
- reuses anything you already have running instead of launching duplicates

To stop the services the script started:

```powershell
.\stop-project.ps1
```

## Quick Start

Use these steps when everything is already installed.

### 1. Start Ollama

Open a terminal and run:

```powershell
ollama serve
```

If you have not downloaded the model yet, run this once in another terminal:

```powershell
ollama pull qwen2.5:1.5b
```

### 2. Start the backend

From the project root:

```powershell
cd "C:\Projects\Personality Chatbot"
.\.venv\Scripts\Activate.ps1
python -m uvicorn main1:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

### 3. Start the frontend

Open a second terminal:

```powershell
cd "C:\Projects\Personality Chatbot\frontend\persona-ui"
npm start
```

Frontend URL:

```text
http://localhost:3000
```

## First-Time Setup

### 1. Create a virtual environment

From the project root:

```powershell
cd "C:\Projects\Personality Chatbot"
python -m venv .venv
```

### 2. Install Python dependencies

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```powershell
cd "C:\Projects\Personality Chatbot\frontend\persona-ui"
npm install
```

### 4. Install Ollama and model

Install Ollama if needed:

```text
https://ollama.com/download
```

Then pull the model used by the backend:

```powershell
ollama pull qwen2.5:1.5b
```

## What Runs What

- `main1.py` starts the FastAPI API.
- `engine.py` sends chat requests to Ollama using model `qwen2.5:1.5b`.
- `frontend/persona-ui` is the React app.

## Chat Logs

- `memory/persona_chat_logs.json` stores saved chat history for each persona.
- `chatbot_conversations.json` stores the project-wide conversation log with timestamp, detected emotion, message, and reply.
- Both log files now also store emotion confidence and safety metadata for each turn.

## Safety Layer

- Input guardrails screen for high-risk self-harm, violence, sexual-minor content, and heavy harassment.
- Output guardrails replace unsafe model replies with a safer fallback.
- Safety decisions are logged in `chatbot_conversations.json` and `memory/persona_chat_logs.json` under `safety`.

## Smoke Test

After starting backend and Ollama, this should work:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/personas
```

For a quick chat test:

```powershell
$body = @{
  message = "hello"
  persona = "ankit"
  user_id = "ankit"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Troubleshooting

### `.venv` exists but is broken

If activation works but packages are missing, recreate the virtual environment:

```powershell
cd "C:\Projects\Personality Chatbot"
rmdir .venv -Recurse -Force
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Backend starts but `/chat` returns 500

Usually Ollama is not running. Start it with:

```powershell
ollama serve
```

### Frontend opens but shows no personas

Make sure the backend is running on:

```text
http://127.0.0.1:8000
```

### PowerShell blocks script activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate the venv again.
