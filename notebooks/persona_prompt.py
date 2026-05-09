def build_prompt(persona, query):
    return f"""
You are acting as {persona['name']}.

Rules:
- Tone: {persona['tone']}
- Thinking style: {persona['thinking']}
- Response length: {persona['length']}
- Focus: {persona['focus']}
- Avoid: {persona['avoid']}

Answer the following query strictly following the rules:
"{query}"
"""
