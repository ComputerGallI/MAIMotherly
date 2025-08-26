import os
import pickle
from pathlib import Path

# Optional deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

RETRIEVER = None          # SentenceTransformer model (optional)
GENERATOR = None          # transformers pipeline (optional)
GEN_TOKENIZER = None      # tokenizer for generator (optional)
FAISS_INDEX = None        # faiss.Index (optional)

KNOWLEDGE_CORPUS = []     # list of strings (or entries)
MODEL_LOADED = False

def _artifacts_path() -> str:
    """
    Resolve artifacts folder:
    1) ARTIFACTS_PATH env var (if set)
    2) <backend_root>/artifacts relative to this file
    """
    env_path = os.getenv("ARTIFACTS_PATH")
    if env_path:
        return env_path
    backend_root = Path(__file__).resolve().parents[1]  # .../backend
    return str(backend_root / "artifacts")

def load_your_trained_models() -> bool:
    """Load your exported artifacts (corpus required; vector/gen optional)."""
    global KNOWLEDGE_CORPUS, MODEL_LOADED
    global FAISS_INDEX, RETRIEVER, GENERATOR, GEN_TOKENIZER

    artifacts = _artifacts_path()
    print(f"[model_loader] Looking for artifacts under: {artifacts}")

    # 1) Knowledge corpus (REQUIRED)
    corpus_path = os.path.join(artifacts, "knowledge_corpus.pkl")
    if not os.path.exists(corpus_path):
        print(f"[model_loader] WARNING: Missing corpus: {corpus_path}")
        KNOWLEDGE_CORPUS = []
        MODEL_LOADED = False
        return False

    try:
        with open(corpus_path, "rb") as f:
            KNOWLEDGE_CORPUS = pickle.load(f)
        MODEL_LOADED = True
        print(f"[model_loader] Loaded corpus with {len(KNOWLEDGE_CORPUS)} entries")
    except Exception as e:
        print(f"[model_loader] ERROR loading corpus: {e}")
        KNOWLEDGE_CORPUS = []
        MODEL_LOADED = False
        return False

    # 2) Vector search (optional): FAISS + SentenceTransformer
    faiss_path = os.path.join(artifacts, "faiss_index.bin")
    retriever_dir = os.path.join(artifacts, "retriever_model")
    if faiss and os.path.exists(faiss_path) and os.path.exists(retriever_dir):
        try:
            from sentence_transformers import SentenceTransformer
            FAISS_INDEX = faiss.read_index(faiss_path)
            RETRIEVER = SentenceTransformer(retriever_dir)
            print("[model_loader] Vector search enabled (FAISS + SentenceTransformer)")
        except Exception as e:
            FAISS_INDEX = None
            RETRIEVER = None
            print(f"[model_loader] WARNING: Failed enabling vector search: {e}")
    else:
        print("[model_loader] Vector search artifacts not found. Using keyword search fallback.")

    # 3) Local generator (optional): transformers seq2seq
    gen_dir = os.path.join(artifacts, "generator_model")
    tok_dir = os.path.join(artifacts, "generator_tokenizer")
    if os.path.exists(gen_dir) and os.path.exists(tok_dir):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            GEN_TOKENIZER = AutoTokenizer.from_pretrained(tok_dir)
            gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_dir)
            GENERATOR = pipeline("text2text-generation", model=gen_model, tokenizer=GEN_TOKENIZER)
            print("[model_loader] Local text generation enabled (transformers)")
        except Exception as e:
            GENERATOR = None
            GEN_TOKENIZER = None
            print(f"[model_loader] WARNING: Failed enabling local generator: {e}")
    else:
        print("[model_loader] Generator artifacts not found. Using templated response fallback.")

    return MODEL_LOADED
