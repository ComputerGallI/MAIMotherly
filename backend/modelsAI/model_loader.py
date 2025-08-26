# modelsAI/model_loader.py
import os
import pickle

# Optional imports (only used if artifacts exist)
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

RETRIEVER = None       # SentenceTransformer model (optional)
GENERATOR = None       # transformers text2text pipeline/model (optional)
GEN_TOKENIZER = None   # tokenizer for generator (optional)
FAISS_INDEX = None     # faiss.Index (optional)

KNOWLEDGE_CORPUS = []  # list[str]
MODEL_LOADED = False   # at least the corpus was loaded

def _artifacts_path() -> str:
    return os.getenv("ARTIFACTS_PATH", "./artifacts")

def _exists(path: str) -> bool:
    return os.path.exists(path)

def load_your_trained_models() -> bool:
    """
    Load artifacts exported from your Colab training:
      - knowledge_corpus.pkl (REQUIRED for any search)
      - faiss_index.bin (optional, enables vector search)
      - retriever_model/ (optional: SentenceTransformer model dir)
      - generator_model/, generator_tokenizer/ (optional: BART model/tokenizer dirs)
    Falls back to keyword search if vector stack not available.
    """
    global KNOWLEDGE_CORPUS, MODEL_LOADED
    global FAISS_INDEX, RETRIEVER, GENERATOR, GEN_TOKENIZER

    artifacts = _artifacts_path()
    print(f"[model_loader] Looking for artifacts under: {artifacts}")

    # 1) Load knowledge corpus (required)
    corpus_path = os.path.join(artifacts, "knowledge_corpus.pkl")
    if not _exists(corpus_path):
        print(f"[model_loader] WARNING: Missing corpus: {corpus_path}")
        MODEL_LOADED = False
        return False

    try:
        with open(corpus_path, "rb") as f:
            KNOWLEDGE_CORPUS = pickle.load(f)
        MODEL_LOADED = True
        print(f"[model_loader] Loaded corpus with {len(KNOWLEDGE_CORPUS)} entries")
    except Exception as e:
        print(f"[model_loader] ERROR loading corpus: {e}")
        MODEL_LOADED = False
        return False

    # 2) Try to load FAISS index + retriever for vector search (optional)
    faiss_path = os.path.join(artifacts, "faiss_index.bin")
    retriever_dir = os.path.join(artifacts, "retriever_model")
    if faiss and _exists(faiss_path) and _exists(retriever_dir):
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

    # 3) Try to load generator (optional)
    gen_dir = os.path.join(artifacts, "generator_model")
    tok_dir = os.path.join(artifacts, "generator_tokenizer")
    if _exists(gen_dir) and _exists(tok_dir):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            GEN_TOKENIZER = AutoTokenizer.from_pretrained(tok_dir)
            gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_dir)
            GENERATOR = pipeline("text2text-generation", model=gen_model, tokenizer=GEN_TOKENIZER)
            print("[model_loader] Text generation enabled (transformers)")
        except Exception as e:
            GENERATOR = None
            GEN_TOKENIZER = None
            print(f"[model_loader] WARNING: Failed enabling generator: {e}")
    else:
        print("[model_loader] Generator artifacts not found. Using templated response fallback.")

    return MODEL_LOADED
