import numpy as np, os, hashlib
from sentence_transformers import SentenceTransformer

def encode_cached(texts, model_name, cache_dir, tag, bs=128):
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5((model_name + tag + str(len(texts))).encode()).hexdigest()[:10]
    path = f"{cache_dir}/{tag}_{h}.npy"
    if os.path.exists(path): return np.load(path)
    m = SentenceTransformer(model_name)
    v = m.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=True)
    np.save(path, v); return v