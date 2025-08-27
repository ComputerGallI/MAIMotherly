import os
from typing import List, Optional

# pip install google-generativeai
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Import Enviromental Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-1.5-flash")

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
        "You are MAI, a warm, concise mental-health buddy. "
        "Be empathetic, practical, and avoid medical diagnoses. "
        "Use the provided context and quiz results when helpful. "
        "Keep answers to 1 to 3 sentence short paragraphs."
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
