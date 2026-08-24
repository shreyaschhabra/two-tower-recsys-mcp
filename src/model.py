import torch
import torch.nn as nn

ID_EMB_DIM = 64
HIDDEN_DIM = 128
OUT_DIM = 64
TEXT_EMB_DIM = 384  # all-MiniLM-L6-v2 output dim


class UserTower(nn.Module):
    def __init__(self, num_users: int):
        super().__init__()
        self.id_embedding = nn.Embedding(num_users, ID_EMB_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(ID_EMB_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUT_DIM),
        )

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.id_embedding(user_idx))


class ItemTower(nn.Module):
    def __init__(self, num_items: int):
        super().__init__()
        self.id_embedding = nn.Embedding(num_items, ID_EMB_DIM)
        self.text_proj = nn.Linear(TEXT_EMB_DIM, ID_EMB_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(ID_EMB_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUT_DIM),
        )

    def forward(self, item_idx: torch.Tensor, item_text_emb: torch.Tensor) -> torch.Tensor:
        id_e = self.id_embedding(item_idx)
        text_e = self.text_proj(item_text_emb)
        return self.mlp(torch.cat([id_e, text_e], dim=-1))


class TwoTowerModel(nn.Module):
    """User tower: learned ID embedding -> MLP.
    Item tower: learned ID embedding + frozen MiniLM title embedding -> MLP.
    Both towers output L2-normalized vectors scored via dot product (cosine similarity).
    """

    def __init__(self, num_users: int, num_items: int):
        super().__init__()
        self.user_tower = UserTower(num_users)
        self.item_tower = ItemTower(num_items)

    def encode_users(self, user_idx: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.user_tower(user_idx), dim=-1)

    def encode_items(self, item_idx: torch.Tensor, item_text_emb: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.item_tower(item_idx, item_text_emb), dim=-1)

    def forward(self, user_idx, item_idx, item_text_emb):
        return self.encode_users(user_idx), self.encode_items(item_idx, item_text_emb)
