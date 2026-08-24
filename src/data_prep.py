"""
Loads raw Amazon Reviews 2023 (Video_Games) data, applies 5-core filtering,
builds user/item ID mappings, and creates a leave-last-two-out temporal
train/valid/test split (standard sequential-recommendation protocol:
each user's last interaction -> test, second-to-last -> valid, rest -> train).
"""
import pickle

import pandas as pd

CATEGORY = "Video_Games"
RAW_REVIEWS = f"data/raw/raw/review_categories/{CATEGORY}.jsonl"
RAW_META = f"data/raw/raw/meta_categories/meta_{CATEGORY}.jsonl"
OUT_DIR = "data/processed"
MIN_INTERACTIONS = 5  # 5-core filtering threshold
CHUNKSIZE = 200_000  # stream-parse to keep peak memory low on 8GB RAM


def load_reviews() -> pd.DataFrame:
    keep_cols = ["user_id", "parent_asin", "rating", "timestamp"]
    chunks = []
    reader = pd.read_json(RAW_REVIEWS, lines=True, chunksize=CHUNKSIZE)
    for chunk in reader:
        chunks.append(chunk[keep_cols])
    return pd.concat(chunks, ignore_index=True)


def k_core_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users/items with fewer than k interactions until stable."""
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["parent_asin"].value_counts()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        new_df = df[df["user_id"].isin(keep_users) & df["parent_asin"].isin(keep_items)]
        if len(new_df) == len(df):
            return new_df
        df = new_df


def leave_last_two_out(df: pd.DataFrame):
    df = df.sort_values(["user_id", "timestamp"])
    df["rank"] = df.groupby("user_id").cumcount(ascending=False)  # 0 = most recent
    test = df[df["rank"] == 0]
    valid = df[df["rank"] == 1]
    train = df[df["rank"] >= 2]
    return (
        train.drop(columns="rank").reset_index(drop=True),
        valid.drop(columns="rank").reset_index(drop=True),
        test.drop(columns="rank").reset_index(drop=True),
    )


def main():
    print("Loading raw reviews...")
    reviews = load_reviews()
    print(f"  raw reviews: {len(reviews):,}, users: {reviews['user_id'].nunique():,}, items: {reviews['parent_asin'].nunique():,}")

    print(f"Applying {MIN_INTERACTIONS}-core filtering...")
    filtered = k_core_filter(reviews, MIN_INTERACTIONS)
    print(f"  after filtering: {len(filtered):,}, users: {filtered['user_id'].nunique():,}, items: {filtered['parent_asin'].nunique():,}")

    users = sorted(filtered["user_id"].unique())
    items = sorted(filtered["parent_asin"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {a: i for i, a in enumerate(items)}
    filtered["user_idx"] = filtered["user_id"].map(user2idx)
    filtered["item_idx"] = filtered["parent_asin"].map(item2idx)

    print("Splitting (leave-last-two-out per user)...")
    train, valid, test = leave_last_two_out(filtered)
    print(f"  train: {len(train):,}  valid: {len(valid):,}  test: {len(test):,}")

    print("Attaching item titles from metadata...")
    meta_chunks = []
    reader = pd.read_json(RAW_META, lines=True, chunksize=CHUNKSIZE)
    for chunk in reader:
        chunk = chunk[["parent_asin", "title"]]
        meta_chunks.append(chunk[chunk["parent_asin"].isin(item2idx)])
    meta = pd.concat(meta_chunks, ignore_index=True)
    item_titles = {item2idx[row.parent_asin]: (row.title or "") for row in meta.itertuples()}
    # any item without metadata gets an empty title (handled downstream)
    for idx in item2idx.values():
        item_titles.setdefault(idx, "")

    train.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
    valid.to_parquet(f"{OUT_DIR}/valid.parquet", index=False)
    test.to_parquet(f"{OUT_DIR}/test.parquet", index=False)

    with open(f"{OUT_DIR}/mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "user2idx": user2idx,
                "item2idx": item2idx,
                "item_titles": item_titles,
                "num_users": len(users),
                "num_items": len(items),
            },
            f,
        )

    print("Done. Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
