import os
from typing import List, Optional

# pip install google-generativeai
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Import Enviromental Variables

def is_configured() -> bool:
    return bool(genai and GEMINI_API_KEY)

def _init():
    if is_configured():
        genai.configure(api_key=GEMINI_API_KEY)

def generate_with_gemini(user_message: str, context_docs: List[str], quiz_summary: str = "") -> Optional[str]:
    """
    RAG: Provide retrieved docs + quiz results to Gemini. No chat memory.
    """
    if not is_configured():
        return None

    _init()

    system_preamble = (
        #Add Preamble to let know Gemini what role is taking. Include response format and 
        #use techniques learnt in class
    )

    quiz_block = f"Quiz results: {quiz_summary}\n" if quiz_summary else ""
    bullet_context = "\n".join(f"- {c}" for c in context_docs[:5])

    prompt = (
        f"{system_preamble}\n\n"
        f"{quiz_block}"
        f"Context (retrieved from training corpus):\n{bullet_context}\n\n"
        f"User: {user_message}\n\n"
        "Assistant:"
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        # Fallback parse for older SDK response shapes
        if hasattr(resp, "candidates") and resp.candidates:
            c = resp.candidates[0]
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                txt = " ".join(getattr(p, "text", "") for p in c.content.parts if getattr(p, "text", ""))
                return txt.strip() or None
        return None
    except Exception as e:
        print(f"[gemini] generation failed: {e}")
        return None
