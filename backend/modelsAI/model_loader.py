import os, pickle

KNOWLEDGE_CORPUS = []
MODEL_LOADED = False

def load_your_trained_models():
    """Load your actual trained models"""
    global KNOWLEDGE_CORPUS, MODEL_LOADED
    
    try:
        artifacts_path = os.getenv('ARTIFACTS_PATH', './mai_artifacts')
        print(f"Looking for your trained models in: {artifacts_path}")
        
        corpus_path = f"{artifacts_path}/knowledge_corpus.pkl"
        if os.path.exists(corpus_path):
            with open(corpus_path, 'rb') as f:
                KNOWLEDGE_CORPUS = pickle.load(f)
            print(f"SUCCESS: Loaded {len(KNOWLEDGE_CORPUS)} knowledge entries")
            MODEL_LOADED = True
            return True
        else:
            print(f"ERROR: Knowledge corpus not found: {corpus_path}")
            return False
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return False
