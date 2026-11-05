# Hybrid Multimodal RAG over Document Pages

Ask a question about a set of documents and get an answer that cites the exact page it came from. The system ingests
PDF, PowerPoint, Word, images, Markdown and text, indexes every page both as text and as an image, and is evaluated on
a 415-page textbook from the ViDoRe V3 benchmark.


- **Baseline:** hybrid retrieval (BGE-M3 dense + BM25 sparse, merged with RRF).
- **Improvement:** the same pipeline plus a cross-encoder reranker (bge-reranker-v2-m3) over the top 20 pages.
- **Answers:** Claude Opus 5.5 reads the top pages (text and image) and answers with page citations.

## Repository layout

```
pipeline/       ingestion, chunking, embeddings, retrieval, answering, eval checks, LLM judge
server/         FastAPI backend (the query API)
scripts/        dataset download, dataset index build, retrieval and answer evals
modal_app.py    GPU deployment on Modal: model download, index build, evals, API
supabase/       database schema
web/            Next.js dashboard (deployed on Vercel)
multimodal_rag.ipynb   the original Colab prototype of the whole pipeline
```

## Installation

### 1. Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 20 or newer
- Accounts: [Anthropic](https://console.anthropic.com), [Modal](https://modal.com), [Qdrant Cloud](https://cloud.qdrant.io),
  [Supabase](https://supabase.com). Optional: [Langfuse](https://langfuse.com) for traces, OpenAI for the LLM judge,
  Vercel for hosting the dashboard.

### 2. Install

```bash
git clone https://github.com/sharathkum05/RAG-Autopilot.git
cd RAG-Autopilot
uv sync
cd web && npm install && cd ..
```

### 3. Configure keys

```bash
cp .env.example .env              # backend keys: Anthropic, Qdrant, Supabase, Langfuse, optional OpenAI and HF_TOKEN
cp web/.env.example web/.env.local
```

In `web/.env.local`, set `BACKEND_URL` (the Modal API URL from step 6) and `APP_SECRET` (any long random string).

### 4. Set up the database

Paste `supabase/schema.sql` into the Supabase SQL editor and run it, then:

```bash
uv run python scripts/init_supabase.py      # checks the tables and creates the page-image bucket
```

### 5. Get the dataset

```bash
uv run python scripts/download_vidore_v3.py "raw dataset v3" computer_science
uv run python scripts/make_subset_v3.py "raw dataset v3/computer_science" Introduction_to_Python_Programming "raw dataset v3/python_textbook"
```

The first command downloads the computer-science domain; drop `computer_science` to download all eight public domains
(about 8.5 GB). The second keeps one textbook: 415 pages and 79 English queries with page labels and reference answers.

### 6. Deploy the backend on Modal

```bash
uv run modal setup                                              # log in once
uv run modal secret create matt-secrets --from-dotenv .env      # your keys, available to every Modal function
uv run modal run modal_app.py::download_models                  # once: models into a Modal volume
uv run modal volume put matt-data "raw dataset v3/python_textbook" /python_textbook
uv run modal run modal_app.py::main                             # build the index on a GPU (about 7 minutes)
uv run modal deploy modal_app.py                                # serve the API; prints its URL
```

Put the printed URL in `web/.env.local` as `BACKEND_URL`.

### 7. Run the dashboard

```bash
cd web && npm run dev            # http://localhost:3000
```

To host it on Vercel, import the repository with **Root Directory** set to `web` and add `BACKEND_URL` and
`APP_SECRET` as environment variables.

**Running without Modal.** The backend also runs locally with `uv run uvicorn server.app:app --port 8000` and
`BACKEND_URL=http://127.0.0.1:8000`. The text path works on a laptop; the page-image model needs a GPU beyond a few pages.

## Evaluation

Every run writes one row to Supabase, and the dashboard's **Retrieval Evals** and **Answer Evals** tabs read those rows.

### Retrieval evals

Scores each retrieval config against the benchmark's graded page labels on all 79 queries (nDCG@5, Recall@5,
MRR@10, hit@5) and records latency. No LLM calls; about a minute per config.

```bash
uv run modal run modal_app.py::evals                                          # every config
uv run modal run modal_app.py::evals --configs text_hybrid,text_hybrid_rerank # baseline and improvement only
```

### Answer evals

Answers the same seeded sample of questions with each config, exactly as the app does, then grades every answer twice:

- **Deterministic checks**, no model in the score: cites a correct page, citations are valid, key terms and numbers
  from the reference answer are present, yes/no answers are right. An answer passes at a score of 0.6.
- **LLM judge** (OpenAI GPT-5 mini, a different model family from the answering model): relevance, groundedness,
  correctness and coherence, each scored 1 to 5. Needs `OPENAI_API_KEY` in the Modal secret; without it only the
  deterministic checks run.

```bash
uv run modal run modal_app.py::answer_evals --config text_hybrid --sample 30 --seed 42
uv run modal run modal_app.py::answer_evals --config text_hybrid_rerank --sample 30 --seed 42
```

About 3 minutes and $2 to $3 of Claude usage per config for 30 questions. Drop `--sample` to run all 79.

The app now reads the top 5 pages per answer. The results below were measured with 3; add `--n-pages 3` to
reproduce them.

```bash
uv run python scripts/run_answer_evals.py --rescore-run <run id>   # re-grade a stored run after a rubric change, no new answers
uv run modal run modal_app.py::answer_evals --judge-run <run id>   # add judge scores to a stored run
```

### Results (30 questions, 3 pages per answer)

| Metric | Baseline | With reranker |
|---|---|---|
| Pass rate | 93.3% | **96.7%** |
| Grounded answers (cites a correct page) | 90.0% | **96.7%** |
| Mean score | 0.854 | **0.896** |
| nDCG@5 (79 queries) | 68.2% | **72.9%** |
| LLM judge (out of 5) | 4.81 | 4.83 |
| Retrieval time (p50) | 0.3 s | 2.3 s |
| Cost per answer | $0.059 | $0.059 |

### Retrieval configs

| Config | What it does |
|---|---|
| `text_hybrid` | Baseline: BGE-M3 dense + BM25, RRF |
| `text_hybrid_rerank` | Improvement: baseline + cross-encoder reranker over the top 20 pages |
| `fused` | Text list + ColQwen2.5 page-image list, RRF (default for Playground uploads) |
| `fused_rerank`, `visual`, `bm25` | Measured for comparison |

## Costs and cold starts

The backend scales to zero after 10 minutes idle, so the first question after that waits about 30 seconds for the
models to load. `WARM_WINDOWS` in `modal_app.py` keeps one container warm during set hours (about $0.88 per hour on
an L4); `KEEP_WARM = 1` keeps one warm all the time.
