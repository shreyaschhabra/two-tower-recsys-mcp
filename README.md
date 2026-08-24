# Amazon Neural Recommender — Two-Tower Retrieval, Served over MCP

A deep-learning two-tower recommendation model trained on Amazon's own 2023
review corpus, served as an [MCP](https://modelcontextprotocol.io) (Model
Context Protocol) tool server, with a Streamlit chat frontend that lets a
Gemini agent call those tools on your behalf.

This README walks through the whole pipeline end to end: what the model is,
how it was trained, how well it actually performs (measured, not estimated),
how the MCP server exposes it, and how to run or deploy the frontend.

---

## 1. What this is

Two-tower models are the standard architecture behind large-scale industrial
recommender systems (this pattern — separate "towers" that embed a user and
an item into the same vector space, trained so relevant pairs land close
together — is the same shape used in production by YouTube, Pinterest, and
Amazon's own retrieval systems). This project implements one from scratch,
trains it on real Amazon interaction data, and wraps it for agentic use via
MCP instead of a typical REST API.

**Why MCP instead of a REST API?** MCP is the protocol Anthropic introduced
for connecting LLM agents to tools and data. Wrapping a trained model as MCP
tools (rather than, say, a Flask endpoint) means any MCP-compatible agent —
Claude Desktop, this project's own Streamlit+Gemini frontend, or any other
MCP client — can call `recommend_for_user`, `similar_items`, etc. directly,
with the LLM deciding when and how to invoke them based on natural language.

## 2. Architecture

- **User tower**: a learned user-ID embedding (64-dim) → 2-layer MLP → 64-dim output.
- **Item tower**: a learned item-ID embedding (64-dim) concatenated with a
  frozen `all-MiniLM-L6-v2` sentence embedding of the product title (384-dim,
  projected to 64-dim) → 2-layer MLP → 64-dim output. The frozen text
  embedding is what gives the model cold-start capability — it can place an
  item sensibly in vector space even with zero interaction history, purely
  from its title.
- Both towers output L2-normalized vectors; similarity is a dot product
  (equivalently, cosine similarity).
- **Training loss**: in-batch sampled softmax — for a batch of B (user, item)
  positive pairs, every other item in the batch acts as a negative for every
  user, and cross-entropy is applied over the resulting B×B similarity
  matrix. This is the standard, compute-efficient way to train retrieval
  towers without explicit negative sampling.
- **Serving**: item embeddings are precomputed once and indexed in
  [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`) for
  fast nearest-neighbor retrieval. A second FAISS index, built over the raw
  (untrained) MiniLM title embeddings, enables cold-start text search that
  works independently of the trained collaborative signal.

```
        ┌────────────┐                          ┌────────────┐
        │  User ID   │                          │  Item ID   │
        └─────┬──────┘                          └─────┬──────┘
              │ embed(64)                              │ embed(64)
              ▼                                         ▼
        ┌────────────┐                    ┌──────────────────────────┐
        │  MLP (128) │                    │  Item title → MiniLM(384) │
        └─────┬──────┘                    └─────────────┬─────────────┘
              │                                          │ project(64)
              │                                          ▼
              │                                   concat(128) → MLP(128)
              ▼                                          ▼
        user vector (64, L2-norm)          item vector (64, L2-norm)
              └──────────────┬───────────────────────────┘
                              ▼
                    dot product = relevance score
```

## 3. Dataset

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(UC San Diego McAuley Lab), `Video_Games` category — raw reviews + item
metadata, downloaded directly from HuggingFace.

| Step | Count |
|---|---|
| Raw reviews | 4,624,615 |
| Raw users / items | 2,766,656 / 137,249 |
| After 5-core filtering (users & items with ≥5 interactions) | 857,505 interactions |
| Users / items (post-filter) | 98,906 / 26,354 |
| Train / Valid / Test interactions | 659,693 / 98,906 / 98,906 |

**Split protocol — leave-last-two-out per user, sorted by timestamp**: each
user's most recent interaction → test, second-most-recent → validation, the
rest → train. This is a *temporal* split, so the model is evaluated on
predicting genuinely future behavior relative to what it trained on, not on
randomly held-out interactions (which would leak future information into
training and inflate the numbers).

## 4. Evaluation (real, measured numbers)

Evaluation uses **full-catalog ranking** — every candidate is scored against
all 26,354 items, not a small sampled subset of negatives. Sampled-negative
evaluation (common in older RecSys papers, e.g. ranking against only 99
random negatives) is known to inflate offline metrics substantially, so this
is the harder, more honest protocol. Each user's already-seen items are
excluded from their own candidate ranking.

**Test set — 98,906 users, each user's held-out final interaction:**

| Metric | Value |
|---|---|
| Recall@10 | **1.40%** |
| NDCG@10 | **0.70%** |
| HitRate@10 | **1.40%** (identical to Recall@10 under leave-one-out: exactly one relevant item per user) |

For context: random chance on a 26,354-item catalog with k=10 is
10/26,354 = 0.038%. The trained model is **~37x better than random** under
full-catalog ranking.

Validation Recall@10 peaked at 2.43% (epoch 142/150) during training — the
test number is lower because the test interaction is each user's *furthest*
interaction into the future relative to their training history, which is
intrinsically the harder prediction. That gap is expected behavior for a
temporal split, not a bug. The **test number (1.40%) is the one that should
be quoted anywhere** — validation was used only to pick the best checkpoint
during training, so reporting it as a final result would be a form of
cherry-picking.

Full training curve: [`models/train_history.csv`](models/train_history.csv).
Raw results: [`models/test_results.json`](models/test_results.json).

## 5. MCP tools (`mcp_server.py`)

| Tool | Description |
|---|---|
| `recommend_for_user(user_id, k)` | Top-k personalized recommendations, excludes items the user already interacted with |
| `similar_items(item_id, k)` | Item-to-item similarity via the trained item-tower embeddings |
| `search_items(query_text, k)` | Cold-start semantic search over item titles (MiniLM only — works for items the collaborative model has weak signal on) |
| `explain_recommendation(user_id, item_id)` | Similarity score plus the user's past items most similar to the target, for interpretability |

## 6. Frontend (`streamlit_app.py`)

A chat UI in the same style as [weather-mcp-server](https://github.com/shreyaschhabra/weather-mcp-server):
it spins up the MCP server as a subprocess over stdio, fetches its tool
schemas, converts them to Gemini function-calling declarations, and runs an
agentic loop — Gemini decides which of the 4 tools to call (if any) based on
your message, the tool executes against the real trained model, and the
result is fed back for a final natural-language response. The sidebar shows
each tool's description plus a one-click example using real IDs from the
trained catalog, and an expander with the model's evaluation stats.

## 7. Running locally

```bash
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt

# one-time: reproduce the trained model from scratch
.venv/bin/python src/data_prep.py                  # downloads + filters the dataset
.venv/bin/python src/precompute_text_embeddings.py
.venv/bin/python src/train.py                       # ~150 epochs, ~40s/epoch on an M2 CPU
.venv/bin/python src/evaluate.py                    # writes models/test_results.json
.venv/bin/python src/build_index.py                 # builds FAISS indices for serving

# run the MCP server standalone (stdio transport)
.venv/bin/python mcp_server.py

# or run the chat frontend (spawns the MCP server itself)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in your key
.venv/bin/streamlit run streamlit_app.py
```

If `.streamlit/secrets.toml` (or a `GEMINI_API_KEY` environment variable)
isn't set, the app falls back to asking for a key in the sidebar at runtime.

### macOS note

`faiss` and `torch` conflict over OpenMP runtime initialization on macOS,
which segfaults FAISS search calls unless `torch`/`numpy` are imported
*before* `faiss`, with `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`
set. Both are already handled inside `mcp_server.py` and `src/build_index.py`.

## 8. Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (public or private — Community Cloud can deploy either for a personal account).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, and point it at this repo with `streamlit_app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
   This is the same mechanism `.streamlit/secrets.toml` uses locally — the
   key lives only in Streamlit's secrets store, never in the repo or git
   history, and the app reads it automatically so visitors never need to
   type a key in.
4. Deploy. First boot will be slow (~1-2 min) while it downloads the MiniLM
   model and loads the FAISS indices; subsequent loads are fast.

**Note on repo size**: `models/` (~110MB: the trained checkpoint + FAISS
indices) is committed so the deployed app doesn't need to retrain on every
cold start. `data/raw/` (~2.9GB of raw HuggingFace downloads) is gitignored
and only needed if you want to reproduce training from scratch.

## 9. Tech stack

Python, PyTorch, FAISS, Sentence-Transformers (MiniLM), FastMCP, MCP Python SDK, Google Gemini API, Streamlit, pandas, HuggingFace `datasets`/`huggingface_hub`.
