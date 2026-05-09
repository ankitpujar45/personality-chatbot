from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"

MEMORY_FILE = MEMORY_DIR / "user_profile.json"
PERSONA_CHAT_FILE = MEMORY_DIR / "persona_chat_logs.json"
CHAT_LOG_FILE = BASE_DIR / "chatbot_conversations.json"
EMOTION_MODEL_DIR = BASE_DIR / "notebook" / "models" / "emotion_distilbert"
EMOTION_LABEL_ENCODER_FILE = BASE_DIR / "notebook" / "models" / "emotion_label_encoder.pkl"

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

MAX_FACTS = 30
MAX_CONTEXT_MESSAGES = 8
