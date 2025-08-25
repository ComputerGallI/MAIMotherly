# services/gemini.py
import os
from typing import List, Optional

# If you're using Google's official SDK:
# pip install google-generativeai
try:
    import google.generativeai as genai
except Exception:
    genai = None

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def is_configured() -> bool:
    return bool(genai and GEMINI_API_KEY)

def _init():
    if is_configured():
        genai.configure(api_key=GEMINI_API_KEY)

def generate_with_gemini(user_message: str, context_docs: List[str], quiz_summary: str = "") -> Optional[str]:
    """
    Call Gemini with retrieval-augmented context. Returns a string or None if not available.
    """
    if not is_configured():
        return None

    _init()

    # Build an instruction-grounded prompt
    system_preamble = (
        "You are MAI, a warm, concise mental health buddy. "
        "Be empathetic, practical, and avoid medical claims or diagnoses. "
        "Use the provided context when helpful. Keep answers to 1–3 short paragraphs."
    )
    soft_profile = f"User profile hints: {quiz_summary}\n" if quiz_summary else ""
    context = "\n\n".join([f"- {c}" for c in context_docs[:5]])
    prompt = (
        f"{system_preamble}\n\n"
        f"{soft_profile}"
        f"Context (retrieved):\n{context}\n\n"
        f"User: {user_message}\n\n"
        "Assistant: "
    )

    try:
        # SDK: text-only generation
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        return resp.text.strip() if hasattr(resp, "text") else None
    except Exception as e:
        print(f"[gemini] generation failed: {e}")
        return None
