"""One-time step: embed every item's title with a frozen MiniLM sentence
encoder and cache the result. This is NOT re-run inside the training loop
(only during data prep and index building), so it never becomes a training
bottleneck on 8GB RAM.
"""
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    num_items = mappings["num_items"]
    titles = [mappings["item_titles"].get(i, "") or "" for i in range(num_items)]

    print(f"Encoding {num_items:,} item titles with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        titles, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    np.save("data/processed/item_text_emb.npy", embeddings)
    print("Saved data/processed/item_text_emb.npy", embeddings.shape)


if __name__ == "__main__":
    main()
