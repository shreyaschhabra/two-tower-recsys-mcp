import json
import pickle

import numpy as np
import pandas as pd
import torch

from metrics import evaluate_ranking
from model import TwoTowerModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
K = 10


def main():
    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    num_users, num_items = mappings["num_users"], mappings["num_items"]

    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    item_text_emb = torch.tensor(np.load("data/processed/item_text_emb.npy"), device=DEVICE)

    model = TwoTowerModel(num_users, num_items).to(DEVICE)
    model.load_state_dict(torch.load("models/two_tower_best.pt", map_location=DEVICE))
    model.eval()

    # exclude everything the user has already interacted with (train + valid) from test ranking
    exclude = {}
    for df in (train_df, valid_df):
        for row in df.itertuples():
            exclude.setdefault(row.user_idx, set()).add(row.item_idx)

    test_pairs = list(zip(test_df["user_idx"], test_df["item_idx"]))
    metrics = evaluate_ranking(model, item_text_emb, test_pairs, exclude, num_items, DEVICE, k=K)

    print("Test set metrics (full-catalog ranking, leave-one-out protocol):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open("models/test_results.json", "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "num_users": num_users,
                "num_items": num_items,
                "n_train_interactions": len(train_df),
                "n_valid_interactions": len(valid_df),
                "n_test_interactions": len(test_df),
                "k": K,
                "protocol": "leave-one-out, full-catalog ranking (no sampled negatives)",
            },
            f,
            indent=2,
        )
    print("Saved models/test_results.json")


if __name__ == "__main__":
    main()
