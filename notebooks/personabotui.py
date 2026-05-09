import gradio as gr
from mark13 import generate_response, load_memory

# Persistent memory (long-term)
mem = load_memory()

# System persona
SYSTEM_PROMPT = "You are a chill, real human friend."

def chat_interface(user_input, ui_history, persona_key):
    try:
        # Build backend chat history from UI history
        backend_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user, bot in ui_history:
            backend_history.append({"role": "user", "content": user})
            backend_history.append({"role": "assistant", "content": bot})

        # Generate response
        emotion, reply, updated_history, _ = generate_response(
            user_input,
            backend_history,
            mem,
            persona_key=persona_key
        )

        # Append ONLY clean text to UI
        ui_history.append((user_input, reply))

        return ui_history, ""

    except Exception as e:
        ui_history.append(("System", f"[ERROR] {e}"))
        return ui_history, ""


def clear_chat():
    return [], ""


with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        """
        # 🧠 Emotion & Personality Chatbot  
        **Psychologically Personalized Conversational Agent**
        """
    )

    persona_selector = gr.Dropdown(
    choices=["ankit", "Ganesh", "Varsha"],
    value="ankit",
    label="Select Team Persona"
    )


    chatbot = gr.Chatbot(
        height=480,
        bubble_full_width=False,
        show_label=True
    )

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type something… (e.g., I'm tired of studying)",
            scale=9
        )
        send_btn = gr.Button("Send", scale=1)

    clear_btn = gr.Button("🧹 Clear Chat")

    send_btn.click(
        chat_interface,
        [user_input, chatbot, persona_selector],
        [chatbot, user_input]
    )

    user_input.submit(
        chat_interface,
        [user_input, chatbot, persona_selector],
        [chatbot, user_input]
    )
    clear_btn.click(clear_chat, None, [chatbot, user_input], queue=False)

    gr.Markdown(
        """
        ---
        ### ℹ️ Notes
        - Emotion is detected automatically per message  
        - Responses adapt to emotional state and personality  
        - Conversation context is preserved across turns  
        """
    )

if __name__ == "__main__":
    demo.launch()
