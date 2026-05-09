import torch
import joblib
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    AutoTokenizer, 
    AutoModelForCausalLM
)

emotion_model_path = "models/emotion_distilbert"
emotion_model = DistilBertForSequenceClassification.from_pretrained(emotion_model_path)
emotion_tokenizer = DistilBertTokenizerFast.from_pretrained(emotion_model_path)
label_encoder = joblib.load("models/emotion_label_encoder.pkl")

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
    return label_encoder.inverse_transform([pred_id])[0]


chat_model_name = "Qwen/Qwen1.5-0.5B-Chat"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForCausalLM.from_pretrained(
    chat_model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def generate_response(user_input, chat_history):
    emotion = predict_emotion(user_input)

    chat_history.append({"role": "user", "content": f"[User feels {emotion}] {user_input}"})

    text_prompt = chat_tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = chat_tokenizer(text_prompt, return_tensors="pt").to(chat_model.device)
    outputs = chat_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        pad_token_id=chat_tokenizer.eos_token_id
    )

    reply = chat_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    chat_history.append({"role": "assistant", "content": reply})

    return emotion, reply, chat_history

print(" Emotion-Aware Qwen Chatbot ready! Type 'quit' to exit.\n")

chat_history = [
    {"role": "system", "content": "You are an empathetic AI companion.  - Always respond naturally like a supportive friend.  - Adjust your tone to the user's detected emotion.  • If they are sad → comfort and encourage.  • If they are angry → stay calm and help them relax.  • If they are joyful → celebrate with them.  • If they are fearful → reassure them.  - Keep replies short, conversational, and human-like.  - Sometimes ask gentle follow-up questions to keep the conversation flowing."}
]

while True:
    user_inp = input("You: ")
    if user_inp.lower() == "quit":
        break

    emotion, bot_reply, chat_history = generate_response(user_inp, chat_history)
    print(f"[Detected Emotion: {emotion}]")
    print("Bot:", bot_reply, "\n")
