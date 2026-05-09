import json
import re
from datetime import datetime, timezone
from uuid import uuid4

import ollama

from config import (
    CHAT_LOG_FILE,
    EMOTION_LABEL_ENCODER_FILE,
    EMOTION_MODEL_DIR,
    MAX_CONTEXT_MESSAGES,
    MEMORY_FILE,
    PERSONA_CHAT_FILE,
)
from personas import TEAM_PERSONAS

try:
    import joblib
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
except ImportError:
    joblib = None
    torch = None
    DistilBertForSequenceClassification = None
    DistilBertTokenizerFast = None


OLLAMA_MODEL = "qwen2.5:1.5b"

_emotion_model = None
_emotion_tokenizer = None
_emotion_label_encoder = None
_emotion_device = "cpu"
_emotion_load_attempted = False

SEVERE_SELF_HARM_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bsuicide\b",
    r"\bwant to die\b",
    r"\bhurt myself\b",
    r"\bself harm\b",
]

SEVERE_VIOLENCE_PATTERNS = [
    r"\bkill you\b",
    r"\bhurt you\b",
    r"\bstab\b",
    r"\bshoot\b",
    r"\bmurder\b",
    r"\bbeat (him|her|them|you)\b",
]

HATE_HARASSMENT_PATTERNS = [
    r"\bidiot\b",
    r"\bstupid\b",
    r"\bdumb\b",
    r"\bbitch\b",
    r"\bbastard\b",
    r"\bfuck you\b",
    r"\bshut up\b",
]

SEXUAL_MINOR_PATTERNS = [
    r"\bminor\b",
    r"\bunderage\b",
    r"\bchild porn\b",
    r"\bkid sex\b",
]

PROFANITY_PATTERNS = [
    r"\bfuck\b",
    r"\bshit\b",
    r"\basshole\b",
    r"\bdamn\b",
]


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize_persona_key(persona_key):
    key = (persona_key or "").strip().lower()
    return key if key in TEAM_PERSONAS else "ankit"


def default_memory():
    return {"history": [], "emotion_counts": {}, "last_seen": None}


def default_persona_chat_store():
    return {persona: {} for persona in TEAM_PERSONAS}


def default_conversation_log():
    return {persona: [] for persona in TEAM_PERSONAS}


def default_user_chat_state():
    return {"conversations": []}


def default_safety_result():
    return {
        "toxicity_score": 0.0,
        "categories": [],
        "blocked": False,
        "guardrail_action": "allow",
    }


def load_memory():
    return load_json(MEMORY_FILE, default_memory())


def save_memory(mem):
    save_json(MEMORY_FILE, mem)


def clean_text(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def title_from_text(text, max_len=38):
    base = clean_text(text)
    if not base:
        return "New chat"
    if len(base) <= max_len:
        return base
    return f"{base[: max_len - 3].rstrip()}..."


def normalize_message(item):
    if not isinstance(item, dict):
        return None

    role = item.get("role")
    content = item.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        return None

    confidence = item.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence = round(float(confidence), 4)
    else:
        confidence = None

    return {
        "role": role,
        "content": content,
        "timestamp": item.get("timestamp") or utc_now_iso(),
        "detected_emotion": item.get("detected_emotion"),
        "confidence": confidence,
        "safety": item.get("safety"),
    }


def conversation_timestamps(messages):
    timestamps = [msg.get("timestamp") for msg in messages if isinstance(msg, dict) and msg.get("timestamp")]
    created_at = timestamps[0] if timestamps else utc_now_iso()
    updated_at = timestamps[-1] if timestamps else created_at
    return created_at, updated_at


def normalize_conversation(conversation, fallback_id):
    if not isinstance(conversation, dict):
        conversation = {}

    raw_messages = conversation.get("messages", [])
    messages = []
    for item in raw_messages:
        normalized = normalize_message(item)
        if normalized:
            messages.append(normalized)

    created_at, updated_at = conversation_timestamps(messages)
    first_user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")

    return {
        "id": str(conversation.get("id") or fallback_id),
        "title": clean_text(conversation.get("title") or title_from_text(first_user_message)),
        "created_at": conversation.get("created_at") or created_at,
        "updated_at": conversation.get("updated_at") or updated_at,
        "messages": messages,
    }


def normalize_user_chat_state(raw_user_state):
    changed = False

    if raw_user_state is None:
        return default_user_chat_state(), True

    if isinstance(raw_user_state, list):
        changed = True
        legacy_conversation = normalize_conversation(
            {"id": f"legacy-{uuid4().hex[:8]}", "messages": raw_user_state},
            f"legacy-{uuid4().hex[:8]}",
        )
        return {"conversations": [legacy_conversation] if legacy_conversation["messages"] else []}, changed

    if isinstance(raw_user_state, dict) and "conversations" in raw_user_state:
        raw_conversations = raw_user_state.get("conversations", [])
    elif isinstance(raw_user_state, dict) and "messages" in raw_user_state:
        changed = True
        raw_conversations = [raw_user_state]
    elif isinstance(raw_user_state, dict):
        changed = True
        raw_conversations = []
        for key, value in raw_user_state.items():
            if isinstance(value, list):
                raw_conversations.append({"id": key, "messages": value})
    else:
        return default_user_chat_state(), True

    conversations = []
    for index, conversation in enumerate(raw_conversations):
        normalized = normalize_conversation(conversation, f"conversation-{index + 1}")
        conversations.append(normalized)

    return {"conversations": conversations}, changed


def load_persona_chat_store():
    store = load_json(PERSONA_CHAT_FILE, default_persona_chat_store())
    changed = False

    for persona in TEAM_PERSONAS:
        persona_bucket = store.get(persona)
        if not isinstance(persona_bucket, dict):
            store[persona] = {}
            changed = True
            continue

        for user_id, raw_user_state in list(persona_bucket.items()):
            normalized_state, state_changed = normalize_user_chat_state(raw_user_state)
            if state_changed or normalized_state != raw_user_state:
                persona_bucket[user_id] = normalized_state
                changed = True

    if changed:
        save_json(PERSONA_CHAT_FILE, store)

    return store


def _get_user_state(store, persona_key, user_id):
    persona = normalize_persona_key(persona_key)
    store.setdefault(persona, {})
    raw_user_state = store[persona].get(user_id)
    user_state, changed = normalize_user_chat_state(raw_user_state)
    if changed or raw_user_state != user_state:
        store[persona][user_id] = user_state
    return user_state


def _conversation_meta(conversation):
    messages = conversation.get("messages", [])
    last_message_preview = ""
    if messages:
        last_message_preview = title_from_text(messages[-1].get("content", ""), max_len=54)

    return {
        "id": conversation["id"],
        "title": conversation.get("title") or "New chat",
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "message_count": len(messages),
        "last_message_preview": last_message_preview,
    }


def list_persona_conversations(user_id, persona_key):
    persona = normalize_persona_key(persona_key)
    store = load_persona_chat_store()
    user_state = _get_user_state(store, persona, user_id)
    conversations = user_state.get("conversations", [])

    metas = [_conversation_meta(conversation) for conversation in conversations]
    metas.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return metas


def create_conversation(user_id, persona_key, title="New chat"):
    persona = normalize_persona_key(persona_key)
    store = load_persona_chat_store()
    user_state = _get_user_state(store, persona, user_id)

    conversation = {
        "id": uuid4().hex[:12],
        "title": clean_text(title) or "New chat",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "messages": [],
    }

    user_state["conversations"].append(conversation)
    store[persona][user_id] = user_state
    save_json(PERSONA_CHAT_FILE, store)
    return _conversation_meta(conversation)


def ensure_conversation(user_id, persona_key, conversation_id=None, title="New chat"):
    persona = normalize_persona_key(persona_key)
    store = load_persona_chat_store()
    user_state = _get_user_state(store, persona, user_id)

    if conversation_id:
        for conversation in user_state["conversations"]:
            if conversation["id"] == conversation_id:
                return _conversation_meta(conversation)

    conversation = {
        "id": conversation_id or uuid4().hex[:12],
        "title": clean_text(title) or "New chat",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "messages": [],
    }
    user_state["conversations"].append(conversation)
    store[persona][user_id] = user_state
    save_json(PERSONA_CHAT_FILE, store)
    return _conversation_meta(conversation)


def get_conversation_detail(user_id, persona_key, conversation_id):
    persona = normalize_persona_key(persona_key)
    store = load_persona_chat_store()
    user_state = _get_user_state(store, persona, user_id)

    for conversation in user_state["conversations"]:
        if conversation["id"] == conversation_id:
            return {
                "session": _conversation_meta(conversation),
                "messages": [normalize_message(item) for item in conversation.get("messages", []) if normalize_message(item)],
            }

    return {"session": None, "messages": []}


def load_persona_history(user_id, persona_key, conversation_id=None):
    detail = get_conversation_detail(user_id, persona_key, conversation_id) if conversation_id else None
    if detail and detail["session"]:
        return detail["messages"]

    conversations = list_persona_conversations(user_id, persona_key)
    if not conversations:
        return []

    latest_id = conversations[0]["id"]
    return get_conversation_detail(user_id, persona_key, latest_id)["messages"]


def save_persona_history(user_id, persona_key, conversation_id, chat_history):
    persona = normalize_persona_key(persona_key)
    store = load_persona_chat_store()
    user_state = _get_user_state(store, persona, user_id)

    serializable_history = []
    for item in chat_history:
        normalized = normalize_message(item)
        if normalized:
            serializable_history.append(normalized)

    first_user_message = next((msg["content"] for msg in serializable_history if msg["role"] == "user"), "")
    created_at, updated_at = conversation_timestamps(serializable_history)

    target = None
    for conversation in user_state["conversations"]:
        if conversation["id"] == conversation_id:
            target = conversation
            break

    if target is None:
        target = {
            "id": conversation_id or uuid4().hex[:12],
            "title": title_from_text(first_user_message),
            "created_at": created_at,
            "updated_at": updated_at,
            "messages": [],
        }
        user_state["conversations"].append(target)

    if (not target.get("title")) or target.get("title") == "New chat":
        target["title"] = title_from_text(first_user_message)

    target["messages"] = serializable_history
    target["created_at"] = target.get("created_at") or created_at
    target["updated_at"] = updated_at

    store[persona][user_id] = user_state
    save_json(PERSONA_CHAT_FILE, store)
    return _conversation_meta(target)


def append_conversation_log(user_id, persona_key, conversation_id, user_message, emotion, reply, confidence=None, safety=None):
    persona = normalize_persona_key(persona_key)
    logs = load_json(CHAT_LOG_FILE, default_conversation_log())

    for key in TEAM_PERSONAS:
        if key not in logs or not isinstance(logs[key], list):
            logs[key] = []

    logs[persona].append(
        {
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "persona": persona,
            "conversation_id": conversation_id,
            "detected_emotion": emotion,
            "confidence": round(confidence, 4) if isinstance(confidence, (int, float)) else None,
            "message": user_message,
            "reply": reply,
            "safety": safety or default_safety_result(),
        }
    )
    save_json(CHAT_LOG_FILE, logs)


def _count_matches(text, patterns):
    return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))


def analyze_safety(text):
    lowered = (text or "").strip().lower()
    if not lowered:
        return default_safety_result()

    self_harm_hits = _count_matches(lowered, SEVERE_SELF_HARM_PATTERNS)
    violence_hits = _count_matches(lowered, SEVERE_VIOLENCE_PATTERNS)
    hate_hits = _count_matches(lowered, HATE_HARASSMENT_PATTERNS)
    sexual_minor_hits = _count_matches(lowered, SEXUAL_MINOR_PATTERNS)
    profanity_hits = _count_matches(lowered, PROFANITY_PATTERNS)

    categories = []
    if self_harm_hits:
        categories.append("self_harm")
    if violence_hits:
        categories.append("violence")
    if hate_hits:
        categories.append("hate_or_harassment")
    if sexual_minor_hits:
        categories.append("sexual_minor")
    if profanity_hits:
        categories.append("profanity")

    weighted_score = (
        self_harm_hits * 1.0
        + violence_hits * 1.0
        + hate_hits * 0.7
        + sexual_minor_hits * 1.0
        + profanity_hits * 0.25
    )
    toxicity_score = min(1.0, round(weighted_score / 2.0, 4))

    blocked = any(cat in categories for cat in {"self_harm", "violence", "sexual_minor"}) or toxicity_score >= 0.85
    action = "block_input" if blocked else "allow"

    return {
        "toxicity_score": toxicity_score,
        "categories": categories,
        "blocked": blocked,
        "guardrail_action": action,
    }


def is_unsafe_output(safety_result):
    blocked_categories = {"self_harm", "violence", "sexual_minor", "hate_or_harassment"}
    return safety_result["blocked"] or any(cat in blocked_categories for cat in safety_result["categories"])


def build_guardrail_reply(safety_result, emotion):
    categories = set(safety_result.get("categories", []))
    if "self_harm" in categories:
        return "I'm really glad you said it out loud. Please reach out to someone with you right now or call emergency help if you're in immediate danger."
    if "violence" in categories:
        return "Pause right here and step back from acting on that. Put some distance between you and the person or object, then talk to someone safe now."
    if "sexual_minor" in categories:
        return "I can't help with that. Keep the conversation safe and legal."
    if "hate_or_harassment" in categories:
        return "Let's keep this from turning cruel. Tell me what happened without attacking anyone."
    if emotion in {"sadness", "fear"}:
        return "Let's slow it down for a second and keep this safe."
    return "Let's keep this safe and talk about what's going on."


def build_safe_output_reply(persona_key, emotion):
    persona = normalize_persona_key(persona_key)
    if persona == "mentor":
        return "Stop there and cool it down first."
    if persona == "therapist":
        return "Let's pause and keep this conversation safe."
    if emotion == "sadness":
        return "Let's keep this steady for a second."
    return "Let's keep it safe and real."


def load_emotion_model():
    global _emotion_model
    global _emotion_tokenizer
    global _emotion_label_encoder
    global _emotion_device
    global _emotion_load_attempted

    if _emotion_model is not None and _emotion_tokenizer is not None and _emotion_label_encoder is not None:
        return _emotion_model, _emotion_tokenizer, _emotion_label_encoder, _emotion_device

    if _emotion_load_attempted:
        return None

    _emotion_load_attempted = True

    if not all([joblib, torch, DistilBertForSequenceClassification, DistilBertTokenizerFast]):
        return None

    if not EMOTION_MODEL_DIR.exists() or not EMOTION_LABEL_ENCODER_FILE.exists():
        return None

    _emotion_device = "cuda" if torch.cuda.is_available() else "cpu"
    _emotion_model = DistilBertForSequenceClassification.from_pretrained(EMOTION_MODEL_DIR).to(_emotion_device)
    _emotion_tokenizer = DistilBertTokenizerFast.from_pretrained(EMOTION_MODEL_DIR)
    _emotion_label_encoder = joblib.load(EMOTION_LABEL_ENCODER_FILE)
    _emotion_model.eval()

    return _emotion_model, _emotion_tokenizer, _emotion_label_encoder, _emotion_device


def predict_emotion(text):
    if not text or not text.strip():
        return "neutral", None

    components = load_emotion_model()
    if components is None:
        return "neutral", None

    model, tokenizer, label_encoder, device = components

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    try:
        label = str(label_encoder.inverse_transform([pred_idx])[0]).lower()
    except Exception:
        label = "neutral"

    return label, confidence


def build_persona_prompt(persona_key):
    persona = normalize_persona_key(persona_key)
    return TEAM_PERSONAS.get(persona, TEAM_PERSONAS["ankit"])["prompt"]


def build_context(chat_history):
    formatted = []
    for msg in chat_history[-MAX_CONTEXT_MESSAGES:]:
        normalized = normalize_message(msg)
        if normalized:
            formatted.append({"role": normalized["role"], "content": normalized["content"]})
    return formatted


def style_anchor(persona_key):
    persona = normalize_persona_key(persona_key)
    if persona == "ankit":
        return "Casual, playful, and emotionally present."
    if persona == "mentor":
        return "Blunt, demanding, and focused."
    if persona == "therapist":
        return "Calm, reflective, and emotionally grounded."
    return "Natural and human."


def is_bad_response(reply):
    lowered = reply.lower()
    bad_patterns = [
        "as an ai",
        "language model",
        "how can i help",
        "how can i assist",
        "i'm here to help",
    ]
    return any(pattern in lowered for pattern in bad_patterns)


def looks_persona_consistent(reply, persona_key):
    reply_lower = reply.lower()
    persona = normalize_persona_key(persona_key)

    if persona == "ankit":
        banned = [
            "i understand your concern",
            "let us",
            "as your friend",
            "here are",
        ]
        return not any(phrase in reply_lower for phrase in banned)

    if persona == "mentor":
        too_soft = [
            "it's okay",
            "thats okay",
            "that's okay",
            "that's understandable",
            "i totally get that",
            "no worries",
            "take your time",
        ]
        return not any(phrase in reply_lower for phrase in too_soft)

    if persona == "therapist":
        harsh_or_slang = [
            "bro",
            "fr",
            "lowkey",
            "damn",
            "fix it",
            "then fix it",
        ]
        return not any(phrase in reply_lower for phrase in harsh_or_slang)

    return True


def trim_reply(reply):
    sentences = re.split(r"(?<=[.!?])\s+", reply.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return reply.strip()
    return " ".join(sentences[:2]).strip()


def persist_turn(user_id, persona, conversation_id, user_input, reply, emotion, confidence, chat_history, mem, safety):
    user_timestamp = utc_now_iso()
    reply_timestamp = utc_now_iso()

    chat_history.append(
        {
            "role": "user",
            "content": user_input,
            "timestamp": user_timestamp,
            "detected_emotion": emotion,
            "confidence": confidence,
            "safety": safety,
        }
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": reply,
            "timestamp": reply_timestamp,
            "detected_emotion": emotion,
            "confidence": confidence,
            "safety": safety,
        }
    )

    mem.setdefault("emotion_counts", {})
    mem["emotion_counts"][emotion] = mem["emotion_counts"].get(emotion, 0) + 1
    mem["last_seen"] = reply_timestamp

    save_memory(mem)
    session_meta = save_persona_history(user_id, persona, conversation_id, chat_history)
    append_conversation_log(user_id, persona, session_meta["id"], user_input, emotion, reply, confidence, safety=safety)
    return session_meta


def generate_response(user_input, chat_history, mem, persona_key="ankit", user_id="ankit", conversation_id=None):
    user_input = clean_text(user_input)
    persona = normalize_persona_key(persona_key)
    conversation_meta = ensure_conversation(user_id, persona, conversation_id)
    conversation_id = conversation_meta["id"]

    if not user_input:
        return "neutral", "say something", chat_history, mem, conversation_meta

    emotion, confidence = predict_emotion(user_input)
    input_safety = analyze_safety(user_input)
    system_prompt = build_persona_prompt(persona)
    style = style_anchor(persona)

    if input_safety["blocked"]:
        reply = build_guardrail_reply(input_safety, emotion)
        session_meta = persist_turn(
            user_id,
            persona,
            conversation_id,
            user_input,
            reply,
            emotion,
            confidence,
            chat_history,
            mem,
            input_safety,
        )
        return emotion, reply, chat_history, mem, session_meta

    messages = [
        {
            "role": "system",
            "content": f"""
{system_prompt}

STRICT:
- Stay in persona the whole time
- No assistant tone
- No generic help-desk language
- Reply like a real person in chat
- Keep it compact, usually 1-2 short sentences
- Reference the user's actual message, not generic advice

User emotion: {emotion}
Style anchor: {style}

SAFETY:
- Do not encourage self-harm, violence, illegal acts, sexual content involving minors, or harassment
- De-escalate risky messages
- If the user is in danger, tell them to seek immediate real-world help
""",
        }
    ]

    messages += build_context(chat_history)
    messages.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.85, "top_p": 0.92},
    )

    reply = response["message"]["content"]
    reply = re.split(r"ASSISTANT|USER|Assistant:|User:", reply)[0]
    reply = clean_text(reply)
    reply = trim_reply(reply)

    if is_bad_response(reply) or not looks_persona_consistent(reply, persona):
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""
{system_prompt}

EXTRA PERSONA LOCK:
- Stay fully in persona
- If you are mentor, do not soften the message
- If you are Ankit, sound like a real friend and not an assistant
- If you are therapist, stay gentle and reflective
""",
                },
                *messages[1:],
            ],
            options={"temperature": 0.95, "top_p": 0.95},
        )
        reply = response["message"]["content"]
        reply = re.split(r"ASSISTANT|USER|Assistant:|User:", reply)[0]
        reply = clean_text(reply)
        reply = trim_reply(reply)

    if len(reply) < 3:
        if persona == "ankit":
            reply = "say that again, bro?"
        elif persona == "mentor":
            reply = "say it clearly."
        else:
            reply = "Could you say a little more?"

    output_safety = analyze_safety(reply)
    if is_unsafe_output(output_safety):
        output_safety["blocked"] = True
        output_safety["guardrail_action"] = "replaced_output"
        reply = build_safe_output_reply(persona, emotion)
        safety = output_safety
    else:
        safety = input_safety

    session_meta = persist_turn(
        user_id,
        persona,
        conversation_id,
        user_input,
        reply,
        emotion,
        confidence,
        chat_history,
        mem,
        safety,
    )

    return emotion, reply, chat_history, mem, session_meta
