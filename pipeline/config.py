"""Settings from environment variables (or a .env file next to the project). Nothing is stored locally except model caches."""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Services
    anthropic_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_bucket: str = "pages"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    ntfy_topic: str = ""            # phone alerts on visits, questions and uploads (pipeline/notify.py); off when empty
    hf_token: str = ""                # optional: faster, rate-limit-free model downloads; required only for gated models
    openai_api_key: str = ""          # optional: the frozen answer-eval judge (second opinion next to the deterministic checks)
    judge_model: str = "gpt-5-mini"

    # Models
    answer_model: str = "claude-opus-5-5"
    text_model: str = "BAAI/bge-m3"
    bm25_model: str = "Qdrant/bm25"
    colqwen_model: str = "vidore/colqwen2.5-v0.2"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"              # auto | cuda | mps | cpu
    visual_enabled: bool = True       # ColQwen page embeddings (slow on CPU; fine on MPS for small files, fast on Modal)
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])

    # Chunking + retrieval
    chunk_words: int = 300
    table_rows_per_part: int = 25
    top_k_per_path: int = 50
    fused_k: int = 20
    final_k: int = 5
    answer_pages: int = 5           # pages (text + image) the answer model reads, out of the final_k shown as evidence
    rrf_k: int = 60
    pages_collection: str = "matt_pages"
    chunks_collection: str = "matt_chunks"

    # Server
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    page_image_width: int = 1200      # display copy uploaded to storage; originals are embedded at full size

    @property
    def visual_dims(self) -> int:
        return 128


@lru_cache
def settings() -> Settings:
    s = Settings()
    if s.hf_token:
        import os
        os.environ["HF_TOKEN"] = s.hf_token
    return s
