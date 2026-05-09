# filename: mark14_improved.py
import re
import json
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import joblib
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# --------------------
# Config
# --------------------
BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "user_profile.json"
CHAT_LOG_FILE = BASE_DIR / "chat_log.json"
EMOTION_MODEL_PATH = BASE_DIR / "models" / "emotion_distilbert"
PERSONALITY_MODEL_PATH = BASE_DIR / "results" / "bigfive_personality_model.pkl"
EMOTION_LABEL_ENCODER_PATH = BASE_DIR / "models" / "emotion_label_encoder.pkl"
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
MAX_FACTS = 30
MAX_EMOTION_HISTORY = 20
MAX_CONTEXT_MESSAGES = 8

# Generation defaults (short, human-like)
GEN_KW_BASE = dict(
    max_new_tokens=80,        # allow natural sentence completion, we will trim after generation
    do_sample=True,
    top_p=0.85,
    temperature=0.7,
    repetition_penalty=1.05,
)

# Confidence threshold for emotion predictions
EMOTION_CONF_THRESH = 0.45

# smoothing window for emotion history
EMOTION_SMOOTH_WINDOW = 3

# --------------------
# Utilities: IO / Logging / Memory
# --------------------
def load_json(path: Path, default):
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(path: Path, obj: Any):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save {path}: {e}")

chat_log_cache = load_json(CHAT_LOG_FILE, [])
def log_chat_entry(text: str, emotion: str, confidence: float, latency: float):
    """Safely append a chat entry to the chat log JSON file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "predicted_emotion": emotion,
        "confidence": confidence,
        "latency_s": round(latency, 3)
    }

    chat_log_cache.append(entry)
    save_json(CHAT_LOG_FILE, chat_log_cache)


def load_memory() -> Dict:
    default = {"facts": [], "emotion_counts": {}, "emotion_history": [], "last_seen": None}
    return load_json(MEMORY_FILE, default)

def load_chat_log() -> List[Dict]:
    """Return the saved chat log (list)."""
    return list(chat_log_cache)


def save_memory(mem: Dict):
    save_json(MEMORY_FILE, mem)

# --------------------
# Model Loading (cached)
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Emotion model
emotion_model = DistilBertForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH).to(device)
emotion_tokenizer = DistilBertTokenizerFast.from_pretrained(EMOTION_MODEL_PATH)
label_encoder = joblib.load(EMOTION_LABEL_ENCODER_PATH)
emotion_model.eval()

# Personality model
try:
    bigfive_model = joblib.load(PERSONALITY_MODEL_PATH)
    print("[INFO] BigFive personality model loaded.")
except Exception:
    bigfive_model = None
    print("[WARN] BigFive personality model not found. Using defaults.")

# Chat model + tokenizer
chat_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
chat_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
    low_cpu_mem_usage=not torch.cuda.is_available(),
    trust_remote_code=True,
)
chat_model.eval()

# --------------------
# Small helpers
# --------------------
META_PATTERNS = re.compile(
    r"(?i)(as\s+an\s+ai|language\s+model|assistant|i\s+do\s+not\s+have\s+feelings|i\s+am\s+an\s+ai)"
)

BAD_ADVICE_PATTERNS = re.compile(
    r"(?i)(here are (some )?tips|you should|you could|i suggest|try to|ways to|steps to|recommend|advice)"
)

EMO_EMOJI = {
    "joy": "😄", "sadness": "😔", "anger": "😤", "fear": "😰", "love": "❤️",
    "optimism": "🙂", "pessimism": "😕", "surprise": "😲", "trust": "🤝", "neutral":"😐"
}

CONTRACTIONS = {
    r"\bis not\b": "isn't", r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't",
    r"\bi am\b": "i'm", r"\bI am\b": "I'm", r"\bI will\b": "I'll", r"\bI would\b": "I'd",
    r"\bcan not\b": "can't", r"\bwill not\b": "won't", r"\bwe are\b": "we're"
}

def apply_contractions(text: str) -> str:
    for pat, repl in CONTRACTIONS.items():
        text = re.sub(pat, repl, text, flags=re.I)
    return text

def sanitize_user_text(t: str) -> str:
    return t.strip()

def is_meta_reply(text: str) -> bool:
    return bool(META_PATTERNS.search(text))

def ban_advice(text: str) -> str:
    # remove advice-like fragments. If removed entirely, fallback to short empathetic lines.
    if BAD_ADVICE_PATTERNS.search(text):
        text = re.sub(BAD_ADVICE_PATTERNS, "", text)
        text = re.sub(r"(?i)(here (are|is).*)", "", text)
    return text.strip()

def trim_to_sentences(text: str, n: int = 2) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if not sentences:
        return text.strip()
    return " ".join(sentences[:n]).strip()

def pick_fallback(emotion: str) -> str:
    # short human-like fallbacks per emotion
    candidates = {
        "sadness": ["that sucks bro 😔", "aw man, that’s rough.", "uuugh that’s tough."],
        "anger": ["ugh that pisses me off too 😤", "that's annoying, I get it."],
        "joy": ["sick!! that's lit 😄", "woah, that's awesome! 🎉"],
        "fear": ["oh man, that sounds stressful 😟", "dang, that sounds rough."],
        "neutral": ["yeah, got you.", "ah okay, tell me more."],
    }
    return random.choice(candidates.get(emotion, ["I hear you."]))

def humanize_reply(raw: str, emotion: str) -> str:
    # 1) Clean and remove theatrical / AI / advice bits
    r = clean_text_basic(raw)
    r = ban_advice(r)
    r = apply_contractions(r)

    # 2) If meta/empty, fallback
    if not r or is_meta_reply(r):
        r = pick_fallback(emotion)

    # 3) Trim to at most 2 sentences
    r = trim_to_sentences(r, 2)

    # 4) Add small human variance (emoji / filler) probabilistically
    if len(r.split()) < 6:
        # short replies — maybe add small filler
        if random.random() < 0.4:
            r = r + " " + random.choice(["😅", "ya", "nah", "fr"])
    else:
        if random.random() < 0.45:
            emo = EMO_EMOJI.get(emotion, "")
            if emo:
                r = r + " " + emo

    return r.strip()

def clean_text_basic(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)        # remove bracketed directions
    text = re.sub(r"User:.*", "", text)        # remove echoes
    text = re.sub(r"\s{2,}", " ", text)        # collapse spaces
    return text.strip()

# --------------------
# Prediction / Inference
# --------------------
def predict_emotion(text: str, mem: Dict) -> Tuple[str, float]:
    """
    Return (label, confidence). Apply smoothing and fallback if confidence low.
    mem is used to store and read emotion_history for smoothing.
    """
    text = text.strip()
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96).to(device)
    with torch.inference_mode():
        logits = emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_i = int(torch.argmax(probs).item())
        confidence = float(probs[pred_i].item())

    try:
        label = label_encoder.inverse_transform([pred_i])[0].lower()
    except Exception:
        label = "neutral"

    # Confidence fallback & smoothing
    hist = mem.get("emotion_history", [])
    hist.append(label)
    mem["emotion_history"] = hist[-MAX_EMOTION_HISTORY:]  # keep limited history

    if confidence < EMOTION_CONF_THRESH:
        # pick the most common in recent history (simple smoothing)
        window = mem["emotion_history"][-EMOTION_SMOOTH_WINDOW:]
        if window:
            smoothed = max(set(window), key=window.count)
            # if smoothed majority exists, prefer it when confidence low
            return smoothed, confidence
        else:
            return "neutral", confidence

    return label, confidence

def infer_personality(text: str) -> Dict[str, float]:
    default_personality = {
        "openness": 0.6,
        "conscientiousness": 0.6,
        "extraversion": 0.65,
        "agreeableness": 0.9,
        "neuroticism": 0.3,
    }
    if bigfive_model is None:
        return default_personality.copy()
    try:
        # handle both predict returning array of floats or dict-like
        res = bigfive_model.predict([text])[0]
        if isinstance(res, (list, tuple)):
            traits = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
            return {t: float(v) for t, v in zip(traits, res)}
        elif isinstance(res, dict):
            return {k: float(res.get(k, 0.5)) for k in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]}
        else:
            # fallback: return defaults
            return default_personality.copy()
    except Exception:
        return default_personality.copy()

from personas import TEAM_PERSONAS


def normalize_persona_key(persona_key: str) -> str:
    normalized = (persona_key or "").strip().lower()
    for key in TEAM_PERSONAS:
        if key.lower() == normalized:
            return key
    return "ankit"


def build_generation_history(chat_history: List[Dict], system_prompt: str) -> List[Dict]:
    recent_messages = [
        msg for msg in chat_history[1:]
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]
    if len(recent_messages) > MAX_CONTEXT_MESSAGES:
        recent_messages = recent_messages[-MAX_CONTEXT_MESSAGES:]
    return [{"role": "system", "content": system_prompt}, *recent_messages]


def build_persona_prompt(persona_key: str, emotion: str, tone_desc: str) -> str:
    persona = TEAM_PERSONAS[normalize_persona_key(persona_key)]
    emoji_rule = "Use emojis occasionally." if persona["emoji"] else "Do not use emojis."

    return (
        f"You are {persona['name']}, a team member in this project. "
        f"Your communication style is {persona['style']}. "
        f"Verbosity: {persona['verbosity']}. "
        f"The user feels {emotion}. "
        f"Match a {tone_desc} tone. "
        "Respond naturally in character. "
        "DO NOT give advice, steps, or solutions. "
        "DO NOT sound like a therapist or teacher. "
        f"{emoji_rule} "
        "Limit responses to 1–2 short sentences."
    )

# --------------------
# Core generation
# --------------------
def generate_response(user_input: str, chat_history: List[Dict], mem: Dict, persona_key: str = "ankit", max_attempts: int = 2):

    """
    Returns: emotion, reply, updated_chat_history, updated_mem
    - chat_history: list of dicts role/content (system at index 0)
    - mem: loaded memory (will be updated in-place)
    """
    start_t = time.perf_counter()
    user_input = sanitize_user_text(user_input)
    if not user_input:
        return "neutral", "say something!", chat_history, mem
    persona_key = normalize_persona_key(persona_key)
    if not chat_history or not isinstance(chat_history[0], dict) or chat_history[0].get("role") != "system":
        chat_history = [{"role": "system", "content": "You are a chill, real human friend."}, *chat_history]

    # 1) Emotion detection (with smoothing)
    emotion, confidence = predict_emotion(user_input, mem)

    # 2) Personality inference (quick)
    personality = infer_personality(user_input)

    # 3) Update memory
    mem.setdefault("emotion_counts", {})
    mem["emotion_counts"][emotion] = mem["emotion_counts"].get(emotion, 0) + 1
    mem.setdefault("facts", [])
    mem["facts"].append(user_input)
    mem["facts"] = mem["facts"][-MAX_FACTS:]
    mem["last_seen"] = datetime.now(timezone.utc).isoformat()

    # Log now (latency will be appended after generation)
    # We'll log later with latency

    # 4) Build tone description from personality (short)
    tone = []
    if personality.get("agreeableness", 0) > 0.8: tone.append("warm and kind")
    if personality.get("extraversion", 0) > 0.7: tone.append("chill and expressive")
    if personality.get("openness", 0) > 0.7: tone.append("thoughtful and curious")
    tone_desc = ", ".join(tone) if tone else "relaxed and natural"

    # 5) prepare system prompt (concise, strong)
    system_prompt = build_persona_prompt(persona_key, emotion, tone_desc)


    generation_history = build_generation_history(chat_history, system_prompt)
    user_message = {"role": "user", "content": f"[Emotion: {emotion}] {user_input}"}
    generation_history.append(user_message)
    chat_history[0] = {"role": "system", "content": system_prompt}
    chat_history.append(user_message)

    # 6) Build prompt and run generation (with a small retry loop)
    attempt = 0
    reply = ""
    while attempt < max_attempts:
        attempt += 1
        try:
            gen_kwargs = {**GEN_KW_BASE, "temperature": min(0.9, GEN_KW_BASE["temperature"] + 0.1 * (attempt - 1))}
            prompt_text = chat_tokenizer.apply_chat_template(generation_history, tokenize=False, add_generation_prompt=True)
            inputs = chat_tokenizer(prompt_text, return_tensors="pt").to(chat_model.device)

            with torch.inference_mode():
                out = chat_model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=chat_tokenizer.eos_token_id,
                    eos_token_id=chat_tokenizer.eos_token_id,
                )
            raw = chat_tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            reply_candidate = humanize_reply(raw, emotion)

            # Avoid meta responses or empty rows
            if (
                is_meta_reply(reply_candidate)
                or BAD_ADVICE_PATTERNS.search(reply_candidate)
                or re.search(r"\b(try|steps|ways|tips|suggest)\b", reply_candidate, re.I)
                or len(reply_candidate.split("\n")) > 1
            ):
                continue


            # Final reply accepted
            reply = reply_candidate
            break
        except Exception as e:
            # log and attempt again
            print(f"[WARN] generation attempt {attempt} failed: {e}")
            time.sleep(0.3)
            continue

    # If still empty, fallback
    if not reply:
        reply = pick_fallback(emotion)

    # 7) finalize: store assistant turn, compute latency, log
    chat_history.append({"role": "assistant", "content": reply})
    latency = time.perf_counter() - start_t
    log_chat_entry(user_input, emotion, confidence, latency)
    save_memory(mem)

    return emotion, reply, chat_history, mem

def cli_main():
    print("Friend Bot (improved). Type 'quit' to exit.")
    mem = load_memory()
    chat_history = [{"role": "system", "content": "You are a chill, real human friend."}]
    persona_key = input(
    "Choose persona (ankit / teammate1 / teammate2): ").strip().lower()

    persona_key = normalize_persona_key(persona_key)

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: catch you later bro 👋")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("Bot: aight bro, talk later 👋")
            break
        try:
            emotion, reply, chat_history, mem = generate_response(
                user, chat_history, mem, persona_key=persona_key
                )
            print(f"[Emotion: {emotion}]")
            print("Bot:", reply, "\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            print("Bot: ugh glitch, try again?\n")

if __name__ == "__main__":
    cli_main()
