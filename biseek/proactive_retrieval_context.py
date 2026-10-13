"""Causal question retrieval for proposal-compilation context.

This module is an opt-in adapter.  It does not alter proposal matching: it only
adds a small number of question-retrieved historical frames to the one-time 8B
proposal compiler input at query registration.
"""

from __future__ import annotations

from collections import OrderedDict
import time
from typing import Any

import torch
import torch.nn.functional as F


def causal_retrieval_chunks(
    *,
    sample_timestamps: list[float],
    registration_timestamp: float,
    first_recent_position: int,
    chunk_frames: int,
    max_vectors: int,
) -> tuple[list[list[int]], int]:
    """Return completed, fixed-grid historical chunks under a FIFO capacity."""

    if chunk_frames <= 0:
        raise ValueError("proposal_context_retrieval_chunk_frames must be positive")
    if max_vectors <= 0:
        raise ValueError("proposal_context_retrieval_max_vectors must be positive")
    eligible_positions = [
        index
        for index, value in enumerate(sample_timestamps)
        if index < int(first_recent_position)
        and float(value) <= float(registration_timestamp) + 1e-6
    ]
    chunks = [
        eligible_positions[start : start + chunk_frames]
        for start in range(0, len(eligible_positions), chunk_frames)
        if len(eligible_positions[start : start + chunk_frames]) == chunk_frames
    ]
    total = len(chunks)
    return chunks[-max_vectors:], total


def top_retrieval_chunk_indices(scores: torch.Tensor, top_k: int) -> list[int]:
    """Stable top-k indices, returned in score order for auditable selection."""

    values = scores.detach().float().cpu().reshape(-1).tolist()
    if top_k <= 0:
        raise ValueError("proposal_context_retrieval_top_k must be positive")
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[
        : min(top_k, len(values))
    ]


class QuestionRetrievalProposalContext:
    """Build recent-plus-retrieved causal context for an 8B proposal compiler."""

    def __init__(self, *, encoder: Any, cfg: dict[str, Any]) -> None:
        self.encoder = encoder
        self.chunk_frames = int(cfg.get("proposal_context_retrieval_chunk_frames", 4))
        self.top_k = int(cfg.get("proposal_context_retrieval_top_k", 1))
        self.max_vectors = int(cfg.get("proposal_context_retrieval_max_vectors", 128))
        self.cache_size = int(cfg.get("proposal_context_retrieval_cache_size", 4096))
        expected_chunk_frames = int(cfg.get("memory_chunk_frames", self.chunk_frames))
        if self.chunk_frames != expected_chunk_frames:
            raise ValueError(
                "proposal_context_retrieval_chunk_frames must equal the 2B "
                f"encoder chunk size ({expected_chunk_frames})"
            )
        if self.top_k <= 0:
            raise ValueError("proposal_context_retrieval_top_k must be positive")
        if self.max_vectors <= 0:
            raise ValueError("proposal_context_retrieval_max_vectors must be positive")
        if self.cache_size <= 0:
            raise ValueError("proposal_context_retrieval_cache_size must be positive")
        self._chunk_vector_cache: OrderedDict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = OrderedDict()
        self.peak_cache_bytes = 0

    @property
    def cache_bytes(self) -> int:
        return int(
            sum(value.numel() * value.element_size() for value in self._chunk_vector_cache.values())
        )

    def _chunk_vector(
        self, *, sampler: Any, positions: list[int]
    ) -> tuple[torch.Tensor, bool]:
        key = (str(sampler.path), tuple(int(value) for value in positions))
        cached = self._chunk_vector_cache.get(key)
        if cached is not None:
            self._chunk_vector_cache.move_to_end(key)
            return cached, True
        frames = sampler.decode_positions(positions)
        vector = F.normalize(
            self.encoder.encode_chunk(frames).detach().float().cpu().reshape(-1),
            dim=0,
        )
        self._chunk_vector_cache[key] = vector
        while len(self._chunk_vector_cache) > self.cache_size:
            self._chunk_vector_cache.popitem(last=False)
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        return vector, False

    def build(
        self,
        *,
        sampler: Any,
        target_question: str,
        registration_timestamp: float,
        recent_positions: list[int],
    ) -> dict[str, Any]:
        if not recent_positions:
            raise ValueError("retrieval-augmented proposal context needs recent frames")
        started = time.perf_counter()
        chunks, total_eligible = causal_retrieval_chunks(
            sample_timestamps=sampler.sample_timestamps,
            registration_timestamp=float(registration_timestamp),
            first_recent_position=int(recent_positions[0]),
            chunk_frames=self.chunk_frames,
            max_vectors=self.max_vectors,
        )
        query_vector, query_seconds = self.encoder.query_vector(target_question)
        query = F.normalize(query_vector.detach().float().cpu().reshape(-1), dim=0)
        vectors: list[torch.Tensor] = []
        cache_hits = 0
        for positions in chunks:
            vector, cache_hit = self._chunk_vector(
                sampler=sampler,
                positions=positions,
            )
            vectors.append(vector)
            cache_hits += int(cache_hit)
        if getattr(self.encoder, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize(self.encoder.device)
        if vectors:
            scores = torch.mv(torch.stack(vectors), query)
            selected_indices = top_retrieval_chunk_indices(scores, self.top_k)
        else:
            scores = torch.empty(0, dtype=torch.float32)
            selected_indices = []
        selected_chunks = [chunks[index] for index in selected_indices]
        retrieved_positions = sorted(
            {position for chunk in selected_chunks for position in chunk}
        )
        context_positions = sorted(set(retrieved_positions).union(recent_positions))
        retrieved_set = set(retrieved_positions)
        recent_set = set(recent_positions)
        timestamps = sampler.timestamps(context_positions)
        labels = [
            (
                f"Question-retrieved causal history at {timestamp:.3f} seconds; "
                "use only as background context:"
                if position in retrieved_set and position not in recent_set
                else f"Recent causal registration context at {timestamp:.3f} seconds:"
            )
            for position, timestamp in zip(context_positions, timestamps)
        ]
        return {
            "positions": context_positions,
            "timestamps": timestamps,
            "labels": labels,
            "metrics": {
                "enabled": True,
                "query_source": "standing_target_question_only",
                "selection_policy": "cosine_top_k_over_fifo_causal_chunks",
                "ground_truth_visible_to_retriever": False,
                "registration_timestamp_seconds": float(registration_timestamp),
                "chunk_frames": self.chunk_frames,
                "top_k_chunks": self.top_k,
                "max_memory_vectors": self.max_vectors,
                "eligible_chunks_before_fifo": total_eligible,
                "indexed_chunks": len(chunks),
                "chunk_vector_cache_hits": cache_hits,
                "chunk_encoder_2b_calls": len(chunks) - cache_hits,
                "query_encoder_2b_calls": 1,
                "retrieval_2b_calls": 1 + len(chunks) - cache_hits,
                "query_embedding_seconds": float(query_seconds),
                "retrieval_total_seconds": time.perf_counter() - started,
                "persistent_chunk_vector_cache_bytes": self.cache_bytes,
                "peak_chunk_vector_cache_bytes": self.peak_cache_bytes,
                "candidate_chunk_positions": chunks,
                "candidate_chunk_scores": [float(value) for value in scores.tolist()],
                "selected_candidate_indices": selected_indices,
                "selected_chunk_positions": selected_chunks,
                "selected_chunk_scores": [
                    float(scores[index].item()) for index in selected_indices
                ],
                "retrieved_positions": retrieved_positions,
                "recent_positions": list(recent_positions),
                "compiler_context_positions": context_positions,
            },
        }
