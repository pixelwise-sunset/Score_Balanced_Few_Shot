from __future__ import annotations

import asyncio
from typing import List

import numpy as np
import torch
from pydantic import Field, PrivateAttr
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class HFMeanPoolTextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 512,
        pooling: str = "mean",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size)

        with torch.no_grad():
            for start in iterator:
                batch = [str(text) for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                model_out = self.model(**encoded)
                if self.pooling == "cls":
                    pooled = model_out.last_hidden_state[:, 0]
                else:
                    pooled = self._mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu().numpy())

        if not outputs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)


class LlamaIndexHFMeanPoolEmbedding:
    """Factory wrapper to avoid importing LlamaIndex in non-RAG code paths."""

    @staticmethod
    def create(model_name: str, max_length: int = 512, embed_batch_size: int = 32):
        from llama_index.core.embeddings import BaseEmbedding

        class _Embedding(BaseEmbedding):
            model_name: str = Field(default="")
            max_length: int = Field(default=512)
            _encoder: HFMeanPoolTextEncoder = PrivateAttr()

            def __init__(self, **kwargs):
                super().__init__(embed_batch_size=embed_batch_size, **kwargs)
                self._encoder = HFMeanPoolTextEncoder(
                    self.model_name,
                    max_length=self.max_length,
                )

            def _encode_one(self, text: str) -> List[float]:
                return self._encoder.encode(
                    [text],
                    batch_size=1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].tolist()

            def _encode_many(self, texts: list[str]) -> list[list[float]]:
                return self._encoder.encode(
                    texts,
                    batch_size=self.embed_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

            def _get_query_embedding(self, query: str) -> List[float]:
                return self._encode_one(query)

            async def _aget_query_embedding(self, query: str) -> List[float]:
                return await asyncio.to_thread(self._get_query_embedding, query)

            def _get_text_embedding(self, text: str) -> List[float]:
                return self._encode_one(text)

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                return self._encode_many(texts)

        return _Embedding(model_name=model_name, max_length=max_length)


class HFCLSPoolTextEncoder(HFMeanPoolTextEncoder):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            pooling="cls",
        )
