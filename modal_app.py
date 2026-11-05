"""Matt on Modal: the same `pipeline` + `server` code as the local backend, with a GPU, LibreOffice and pre-downloaded models.

    uv run modal run modal_app.py::download_models          # once: fill the model volume (bge-m3, ColQwen2.5, reranker, BM25, Docling, EasyOCR)
    uv run modal run modal_app.py::main                      # build the dataset index (415 textbook pages) on the GPU
    uv run modal run modal_app.py::evals                     # retrieval evals, every config, 79 English queries
    uv run modal run modal_app.py::answer_evals --config text_hybrid_rerank --sample 30   # answer evals: retrieve + Claude answer + deterministic checks (+ judge)
    uv run modal deploy modal_app.py                          # serve the FastAPI app; set BACKEND_URL in the web app to the printed URL

Secrets: `modal secret create matt-secrets --from-dotenv .env` (Anthropic, Qdrant, Supabase, Langfuse, optional HF_TOKEN).
"""
from __future__ import annotations

import modal

APP_NAME = "matt"
MODELS_DIR, DATA_DIR = "/models", "/data"
GPU = ["L4", "A10G", "L40S", "A100-40GB"]   # batch jobs: first available wins; all run ColQwen in bf16
WEB_GPU = "L4"           # the query API is pinned to L4: predictable cost (~$0.88/h warm); waits for an L4 rather than falling back
KEEP_WARM = 0            # 1 keeps one L4 container resident (no cold start, ~$0.80/h); 0 scales to zero after `scaledown_window`

models_vol = modal.Volume.from_name("matt-models", create_if_missing=True)
data_vol = modal.Volume.from_name("matt-data", create_if_missing=True)
secrets = [modal.Secret.from_name("matt-secrets")]

ENV = {
    "HF_HOME": f"{MODELS_DIR}/hf",
    "FASTEMBED_CACHE_PATH": f"{MODELS_DIR}/fastembed",
    "DOCLING_ARTIFACTS_PATH": f"{MODELS_DIR}/docling",
    "EASYOCR_MODULE_PATH": f"{MODELS_DIR}/docling/EasyOcr",
    "TORCH_HOME": f"{MODELS_DIR}/torch",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "DEVICE": "cuda",
    "CORS_ORIGINS": '["*"]',
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libreoffice-writer", "libreoffice-impress", "libgl1", "libglib2.0-0", "fonts-dejavu-core")
    .uv_sync()
    .env(ENV)
    .add_local_python_source("pipeline", "server", "scripts")
)

app = modal.App(APP_NAME, image=image)


@app.function(volumes={MODELS_DIR: models_vol}, timeout=3600, cpu=4, memory=16384)
def download_models() -> None:
    """Fill the model volume once so no container ever downloads at request time."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    for repo in ("BAAI/bge-m3", "vidore/colqwen2.5-base", "vidore/colqwen2.5-v0.2", "BAAI/bge-reranker-v2-m3"):
        print("hf:", repo, flush=True)
        snapshot_download(repo)
    print("fastembed: Qdrant/bm25", flush=True)
    from fastembed import SparseTextEmbedding

    SparseTextEmbedding("Qdrant/bm25")
    print("docling + easyocr", flush=True)
    from docling.utils.model_downloader import download_models as docling_download

    docling_download(output_dir=Path(ENV["DOCLING_ARTIFACTS_PATH"]), with_code_formula=False, with_picture_classifier=False, with_easyocr=True)
    models_vol.commit()
    print("model volume ready", flush=True)


@app.function(gpu=GPU, volumes={MODELS_DIR: models_vol, DATA_DIR: data_vol}, secrets=secrets, timeout=3 * 3600)
def build_index(limit: int = 0, visual: bool = True) -> None:
    """Dataset load (ViDoRe V3 Python textbook): text pass + ColQwen page pass on the GPU. Resumable."""
    import sys

    sys.argv = ["ingest_dataset", f"{DATA_DIR}/python_textbook"] + (["--limit", str(limit)] if limit else []) + ([] if visual else ["--no-visual"])
    from scripts.ingest_dataset import main

    main()


@app.function(gpu=GPU, volumes={MODELS_DIR: models_vol, DATA_DIR: data_vol}, secrets=secrets, timeout=2 * 3600)
def run_evals(configs: str = "", limit: int = 0, k: int = 10) -> None:
    """Retrieval evals on the textbook queries for every config; one eval_runs row per config."""
    import sys

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    sys.argv = ["run_evals", f"{DATA_DIR}/python_textbook", "--k", str(k), "--hardware", f"Modal {gpu}"] + (["--configs", configs] if configs else []) + (["--limit", str(limit)] if limit else [])
    from scripts.run_evals import main

    main()


@app.function(gpu=GPU, volumes={MODELS_DIR: models_vol, DATA_DIR: data_vol}, secrets=secrets, timeout=2 * 3600)
def run_answer_evals(config: str = "", limit: int = 0, sample: int = 0, seed: int = 42, workers: int = 4, judge: bool = True, judge_run: str = "", n_pages: int = 0) -> None:
    """Answer evals for one config: retrieval + Claude answers + deterministic checks (+ OpenAI judge when the key is set).
    With `judge_run`, only adds judge scores to that stored run (no GPU work, cents)."""
    import sys

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if judge_run:
        sys.argv = ["run_answer_evals", "--judge-run", judge_run, "--workers", str(workers)]
    else:
        sys.argv = ["run_answer_evals", f"{DATA_DIR}/python_textbook", "--workers", str(workers), "--hardware", f"Modal {gpu}"] \
            + (["--config", config] if config else []) + (["--limit", str(limit)] if limit else []) + (["--sample", str(sample), "--seed", str(seed)] if sample else []) + ([] if judge else ["--no-judge"]) + (["--n-pages", str(n_pages)] if n_pages else [])
    from scripts.run_answer_evals import main

    main()


@app.local_entrypoint()
def main(limit: int = 0, visual: bool = True) -> None:
    """`modal run modal_app.py::main --limit 0` builds the dataset index."""
    build_index.remote(limit=limit, visual=visual)


@app.local_entrypoint()
def evals(configs: str = "", limit: int = 0, k: int = 10) -> None:
    """`modal run modal_app.py::evals` scores every retrieval config on the textbook queries."""
    run_evals.remote(configs=configs, limit=limit, k=k)


@app.local_entrypoint()
def answer_evals(config: str = "", limit: int = 0, sample: int = 0, seed: int = 42, workers: int = 4, judge: bool = True, judge_run: str = "", n_pages: int = 0) -> None:
    """`modal run modal_app.py::answer_evals --config text_hybrid_rerank --sample 30` answers the sampled benchmark queries and scores the answers;
    `--judge-run <id>` later adds the OpenAI judge to a stored run for cents."""
    run_answer_evals.remote(config=config, limit=limit, sample=sample, seed=seed, workers=workers, judge=judge, judge_run=judge_run, n_pages=n_pages)


@app.cls(gpu=WEB_GPU, volumes={MODELS_DIR: models_vol, DATA_DIR: data_vol}, secrets=secrets, scaledown_window=600, timeout=900, min_containers=KEEP_WARM)
@modal.concurrent(max_inputs=8)
class Web:
    @modal.enter()
    def load(self) -> None:
        """Load every model into GPU memory before the container takes traffic."""
        import time

        t0 = time.time()
        from pipeline.embed_text import text_embedder
        from pipeline.embed_visual import visual_embedder
        from pipeline.retrieval import reranker

        text_embedder()
        visual_embedder()
        reranker()
        import os
        os.environ["MATT_MODEL_LOAD_S"] = f"{time.time() - t0:.1f}"        # shown in the Runtime panel as the cold-start cost
        os.environ["MATT_SCALEDOWN_S"] = "600"
        print(f"models loaded in {time.time() - t0:.1f}s", flush=True)

    @modal.asgi_app()
    def api(self):
        from server.app import app as fastapi_app

        return fastapi_app


# Demo warm window: one Web container stays warm (no cold start) inside these windows, US Central time, and the app
# scales to zero outside them. Checked every 15 minutes; once the last window ends nothing stays warm.
# Cost while warm: about $0.88/h (Web is pinned to an L4; GPU + CPU + memory), ~$17 for the two windows below.
WARM_TZ = "America/Chicago"
WARM_WINDOWS = [
    ("2026-11-05 09:00", "2026-11-05 19:00"),
    ("2026-11-06 09:00", "2026-11-06 18:00"),
]


def warm_target(now) -> int:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(WARM_TZ)
    inside = any(datetime.fromisoformat(a).replace(tzinfo=tz) <= now < datetime.fromisoformat(b).replace(tzinfo=tz)
                 for a, b in WARM_WINDOWS)
    return 1 if inside else KEEP_WARM


@app.function(image=modal.Image.debian_slim(python_version="3.12"), schedule=modal.Cron("*/15 * * * *", timezone=WARM_TZ), timeout=120)
def warm_window() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    target = warm_target(datetime.now(ZoneInfo(WARM_TZ)))
    modal.Cls.from_name(APP_NAME, "Web")().update_autoscaler(min_containers=target)
    print(f"Web min_containers -> {target}", flush=True)
