import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from metrics import evaluate_ranking
from model import TwoTowerModel

DEVICE = torch.device("cpu")  # this two-tower model is tiny; MPS dispatch overhead
# per op is slower than plain CPU at this scale (many small embedding/MLP ops)
BATCH_SIZE = 256
EPOCHS = 150  # raised: run at 60 was still improving with no plateau, never hit early-stop
LR = 1e-3
TEMPERATURE = 0.07
PATIENCE = 10
K = 10
RESUME = True


def build_exclude_map(*dfs):
    excl = {}
    for df in dfs:
        for row in df.itertuples():
            excl.setdefault(row.user_idx, set()).add(row.item_idx)
    return excl


def main():
    print(f"Device: {DEVICE}")

    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    num_users, num_items = mappings["num_users"], mappings["num_items"]
    print(f"num_users={num_users:,} num_items={num_items:,}")

    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")

    item_text_emb = torch.tensor(np.load("data/processed/item_text_emb.npy"), device=DEVICE)

    train_users = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    train_items = torch.tensor(train_df["item_idx"].values, dtype=torch.long)
    ds = TensorDataset(train_users, train_items)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    valid_pairs = list(zip(valid_df["user_idx"], valid_df["item_idx"]))
    exclude_for_valid = build_exclude_map(train_df)

    model = TwoTowerModel(num_users, num_items).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_recall, best_epoch, patience_ctr, start_epoch = -1.0, -1, 0, 1
    history = []
    if RESUME and os.path.exists("models/two_tower_best.pt") and os.path.exists("models/train_history.csv"):
        model.load_state_dict(torch.load("models/two_tower_best.pt", map_location=DEVICE))
        prev = pd.read_csv("models/train_history.csv")
        history = prev.to_dict("records")
        best_row = prev.loc[prev[f"Recall@{K}"].idxmax()]
        best_recall, best_epoch = float(best_row[f"Recall@{K}"]), int(best_row["epoch"])
        start_epoch = int(prev["epoch"].max()) + 1
        print(f"Resumed from checkpoint: best_epoch={best_epoch} best_recall={best_recall:.4f}, continuing from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for user_idx, item_idx in loader:
            user_idx, item_idx = user_idx.to(DEVICE), item_idx.to(DEVICE)
            batch_text_emb = item_text_emb[item_idx]

            u, v = model(user_idx, item_idx, batch_text_emb)
            logits = (u @ v.T) / TEMPERATURE  # in-batch sampled softmax negatives
            labels = torch.arange(logits.size(0), device=DEVICE)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * user_idx.size(0)

        train_loss = total_loss / len(ds)
        val_metrics = evaluate_ranking(
            model, item_text_emb, valid_pairs, exclude_for_valid, num_items, DEVICE, k=K
        )
        dt = time.time() - t0
        print(
            f"epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val Recall@{K} {val_metrics[f'Recall@{K}']:.4f} | "
            f"val NDCG@{K} {val_metrics[f'NDCG@{K}']:.4f} | {dt:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics[f"Recall@{K}"] > best_recall:
            best_recall = val_metrics[f"Recall@{K}"]
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), "models/two_tower_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val Recall@{K}={best_recall:.4f})")
                break

    pd.DataFrame(history).to_csv("models/train_history.csv", index=False)
    print(f"Best val Recall@{K}={best_recall:.4f} at epoch {best_epoch}. Saved models/two_tower_best.pt")


if __name__ == "__main__":
    main()
