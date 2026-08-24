"""Precomputes embeddings for every item in the catalog using the trained
item tower, builds a FAISS inner-product index (vectors are L2-normalized,
so inner product == cosine similarity), and saves everything the MCP
server needs at inference time.
"""
import pickle

import numpy as np
import pandas as pd
import torch  # noqa: F401  -- must import before faiss: avoids an OpenMP
# runtime conflict between torch and faiss that segfaults on macOS otherwise
import faiss

from model import TwoTowerModel

DEVICE = torch.device("cpu")  # index building is a one-time step, CPU is fine


def main():
    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    num_users, num_items = mappings["num_users"], mappings["num_items"]

    item_text_emb = torch.tensor(np.load("data/processed/item_text_emb.npy"), device=DEVICE)

    model = TwoTowerModel(num_users, num_items).to(DEVICE)
    model.load_state_dict(torch.load("models/two_tower_best.pt", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        all_item_idx = torch.arange(num_items, device=DEVICE)
        item_vecs = model.encode_items(all_item_idx, item_text_emb).cpu().numpy().astype(np.float32)

    dim = item_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_vecs)
    faiss.write_index(index, "models/item_index.faiss")
    print(f"Built FAISS index: {num_items:,} items, dim={dim}")

    # also cache trained user vectors for the recommend_for_user tool
    with torch.no_grad():
        all_user_idx = torch.arange(num_users, device=DEVICE)
        user_vecs = model.encode_users(all_user_idx).cpu().numpy().astype(np.float32)
    np.save("models/user_vecs.npy", user_vecs)
    np.save("models/item_vecs.npy", item_vecs)

    # per-user "already seen" items (train+valid+test), so recommend_for_user
    # can filter out items the user has already interacted with
    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    seen = {}
    for df in (train_df, valid_df, test_df):
        for row in df.itertuples():
            seen.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    with open("models/seen_items.pkl", "wb") as f:
        pickle.dump(seen, f)

    # separate content-only index (raw, L2-normalized MiniLM title embeddings) so
    # search_items() can do cold-start semantic search without needing a trained
    # item-ID embedding (i.e. it also works for items the model never saw).
    text_norm = item_text_emb.numpy().astype(np.float32)
    text_norm = text_norm / (np.linalg.norm(text_norm, axis=1, keepdims=True) + 1e-8)
    text_index = faiss.IndexFlatIP(text_norm.shape[1])
    text_index.add(text_norm)
    faiss.write_index(text_index, "models/text_index.faiss")

    print("Saved models/item_index.faiss, text_index.faiss, user_vecs.npy, item_vecs.npy, seen_items.pkl")


if __name__ == "__main__":
    main()
