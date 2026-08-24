import math

import torch


@torch.no_grad()
def evaluate_ranking(
    model,
    item_text_emb: torch.Tensor,
    eval_pairs,
    exclude_items_per_user: dict,
    num_items: int,
    device,
    k: int = 10,
    user_batch_size: int = 2000,
):
    """Full-catalog ranking evaluation (leave-one-out protocol), vectorized
    over batches of users so it's fast enough to run after every training
    epoch even with ~100K eval users and a ~26K-item catalog.

    For each (user_idx, true_item_idx) pair, ranks the ENTIRE item catalog
    (minus items the user already interacted with earlier) and checks
    where the true held-out item lands. No sampled negatives (which are
    known to inflate offline RecSys metrics) — this is the unbiased protocol.
    """
    model.eval()
    all_item_idx = torch.arange(num_items, device=device)
    item_vecs = model.encode_items(all_item_idx, item_text_emb.to(device))  # [I, D]

    users = torch.tensor([u for u, _ in eval_pairs], device=device, dtype=torch.long)
    true_items = torch.tensor([i for _, i in eval_pairs], device=device, dtype=torch.long)
    n = users.size(0)

    total_hit, total_ndcg = 0.0, 0.0
    for start in range(0, n, user_batch_size):
        end = min(start + user_batch_size, n)
        batch_users = users[start:end]
        batch_true = true_items[start:end]
        b = batch_users.size(0)

        user_vecs = model.encode_users(batch_users)  # [b, D]
        scores = user_vecs @ item_vecs.T  # [b, I]

        # mask out each user's already-seen items (vectorized via a sparse index list)
        rows, cols = [], []
        for i, u in enumerate(batch_users.tolist()):
            excl = exclude_items_per_user.get(u)
            if excl:
                rows.extend([i] * len(excl))
                cols.extend(excl)
        if rows:
            scores[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = -1e9

        true_scores = scores.gather(1, batch_true.view(-1, 1))  # [b, 1]
        ranks = (scores > true_scores).sum(dim=1) + 1  # [b], 1-indexed

        hits = (ranks <= k).float()
        ndcgs = torch.where(ranks <= k, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float))

        total_hit += hits.sum().item()
        total_ndcg += ndcgs.sum().item()

    return {
        f"Recall@{k}": total_hit / n,  # == HitRate@k under leave-one-out (single relevant item/user)
        f"HitRate@{k}": total_hit / n,
        f"NDCG@{k}": total_ndcg / n,
        "n_eval_users": n,
    }
