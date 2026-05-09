from persona_prompt import build_prompt
from memory_manager import update_memory
from logger import log_chat

def call_llm(prompt):
    # Replace with OpenAI / Qwen / Ollama
    return f"[Generated Response]\n{prompt}"

def chat(persona_key, persona, query):
    prompt = build_prompt(persona, query)
    response = call_llm(prompt)

    update_memory(persona_key, query, response)
    log_chat(persona_key, query, response)

    return response
