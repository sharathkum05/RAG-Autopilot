"""Step 4b: ColQwen2.5 page embeddings. ~780 × 128-d multivectors per page plus one pooled vector for a fast first stage."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from pipeline.config import settings
from pipeline.parsing import resolve_device


class VisualEmbedder:
    def __init__(self) -> None:
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        s = settings()
        self.device = resolve_device()
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.torch = torch
        self.model = ColQwen2_5.from_pretrained(s.colqwen_model, torch_dtype=dtype, device_map=self.device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(s.colqwen_model)
        self.batch_size = 4 if self.device == "cuda" else 1

    def _unpad(self, emb, mask) -> list[np.ndarray]:
        return [e[m].float().cpu().numpy().astype(np.float32) for e, m in zip(emb, mask.bool())]

    def pages(self, images: list[Image.Image]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Returns (multivectors, pooled) per image. Falls back to float32 if fp16 produced non-finite values."""
        multi: list[np.ndarray] = []
        pooled: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch_imgs = [im.convert("RGB") for im in images[i:i + self.batch_size]]
            batch = self.processor.process_images(batch_imgs).to(self.model.device)
            with self.torch.inference_mode():
                emb = self.model(**batch)
            for v in self._unpad(emb, batch["attention_mask"]):
                if not np.isfinite(v).all():
                    v = np.nan_to_num(v)
                multi.append(v)
                m = v.mean(0)
                pooled.append(m / (np.linalg.norm(m) + 1e-9))
            del emb, batch
        return multi, pooled

    def query(self, q: str) -> tuple[np.ndarray, np.ndarray]:
        batch = self.processor.process_queries([q]).to(self.model.device)
        with self.torch.inference_mode():
            emb = self.model(**batch)
        v = self._unpad(emb, batch["attention_mask"])[0]
        m = v.mean(0)
        return v, m / (np.linalg.norm(m) + 1e-9)


@lru_cache
def visual_embedder() -> VisualEmbedder:
    return VisualEmbedder()
