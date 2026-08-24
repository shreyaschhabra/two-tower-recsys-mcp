"""MCP tool server for the two-tower video-games recommender.

Exposes trained-model inference (recommendation, similarity, cold-start
search, and explanation) as MCP tools any MCP-compatible agent can call.
Run with: .venv/bin/python mcp_server.py
"""
import os

# must be set before numpy/torch/faiss are imported: works around a macOS
# OpenMP runtime conflict between torch and faiss that otherwise segfaults
# faiss search calls (see also src/build_index.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pickle

import numpy as np
from sentence_transformers import SentenceTransformer  # imports torch internally
# must come before faiss: avoids an OpenMP runtime conflict between
# torch/numpy and faiss that segfaults on macOS otherwise
import faiss
from fastmcp import FastMCP

mcp = FastMCP("Two-Tower Video Games Recommender")

with open("data/processed/mappings.pkl", "rb") as f:
    MAPPINGS = pickle.load(f)
USER2IDX = MAPPINGS["user2idx"]
ITEM2IDX = MAPPINGS["item2idx"]
IDX2ITEM = {v: k for k, v in ITEM2IDX.items()}
ITEM_TITLES = MAPPINGS["item_titles"]

USER_VECS = np.load("models/user_vecs.npy")
ITEM_VECS = np.load("models/item_vecs.npy")
ITEM_INDEX = faiss.read_index("models/item_index.faiss")
TEXT_INDEX = faiss.read_index("models/text_index.faiss")

with open("models/seen_items.pkl", "rb") as f:
    SEEN_ITEMS = pickle.load(f)

TEXT_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")


def _item_payload(item_idx: int, score: float) -> dict:
    return {
        "item_id": IDX2ITEM[item_idx],
        "title": ITEM_TITLES.get(item_idx, ""),
        "score": round(float(score), 4),
    }


@mcp.tool()
def recommend_for_user(user_id: str, k: int = 10) -> dict:
    """Return top-k personalized item recommendations for a known user_id,
    ranked by the trained two-tower model. Excludes items the user has
    already interacted with."""
    if user_id not in USER2IDX:
        return {"error": f"unknown user_id '{user_id}' (not in the training catalog)"}

    user_idx = USER2IDX[user_id]
    scores = ITEM_VECS @ USER_VECS[user_idx]
    seen = SEEN_ITEMS.get(user_idx, set())
    ranked = np.argsort(-scores)

    results = []
    for item_idx in ranked:
        if item_idx in seen:
            continue
        results.append(_item_payload(int(item_idx), scores[item_idx]))
        if len(results) >= k:
            break
    return {"user_id": user_id, "recommendations": results}


@mcp.tool()
def similar_items(item_id: str, k: int = 10) -> dict:
    """Return the k items most similar to the given item_id, using the
    trained item-tower embeddings (id embedding + title embedding)."""
    if item_id not in ITEM2IDX:
        return {"error": f"unknown item_id '{item_id}' (not in the catalog)"}

    item_idx = ITEM2IDX[item_id]
    query_vec = ITEM_VECS[item_idx : item_idx + 1]
    scores, indices = ITEM_INDEX.search(query_vec, k + 1)  # +1 to drop self-match

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == item_idx:
            continue
        results.append(_item_payload(int(idx), score))
        if len(results) >= k:
            break
    return {"item_id": item_id, "title": ITEM_TITLES.get(item_idx, ""), "similar_items": results}


@mcp.tool()
def search_items(query_text: str, k: int = 10) -> dict:
    """Cold-start semantic search over item titles using the same MiniLM
    encoder used at training time. Works even for items outside a user's
    interaction history, since it only needs the item's title text."""
    query_vec = TEXT_ENCODER.encode([query_text], convert_to_numpy=True).astype(np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
    scores, indices = TEXT_INDEX.search(query_vec, k)

    results = [_item_payload(int(idx), score) for score, idx in zip(scores[0], indices[0])]
    return {"query": query_text, "results": results}


@mcp.tool()
def explain_recommendation(user_id: str, item_id: str) -> dict:
    """Explain why an item would be recommended to a user: the model
    similarity score, plus which of the user's past items are most
    similar to it (the closest analogues driving the score)."""
    if user_id not in USER2IDX:
        return {"error": f"unknown user_id '{user_id}'"}
    if item_id not in ITEM2IDX:
        return {"error": f"unknown item_id '{item_id}'"}

    user_idx = USER2IDX[user_id]
    item_idx = ITEM2IDX[item_id]
    score = float(ITEM_VECS[item_idx] @ USER_VECS[user_idx])

    seen = sorted(SEEN_ITEMS.get(user_idx, set()) - {item_idx})
    if not seen:
        return {"user_id": user_id, "item_id": item_id, "score": round(score, 4), "driven_by": []}

    seen_vecs = ITEM_VECS[seen]
    target_vec = ITEM_VECS[item_idx]
    sims = seen_vecs @ target_vec
    top = np.argsort(-sims)[:5]

    driven_by = [
        {**_item_payload(seen[i], sims[i])}
        for i in top
    ]
    return {
        "user_id": user_id,
        "item_id": item_id,
        "title": ITEM_TITLES.get(item_idx, ""),
        "score": round(score, 4),
        "driven_by": driven_by,
    }


if __name__ == "__main__":
    mcp.run()
