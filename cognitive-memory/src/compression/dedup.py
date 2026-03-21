"""Semantic deduplication of reasoning nodes before storage."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schemas.reasoning import NodeType, ReasoningNode
from ..store.vector_store import MemoryStore

_DEFAULT_THRESHOLD = 0.85


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


@dataclass
class DeduplicationResult:
    stored: list[ReasoningNode] = field(default_factory=list)
    merged: list[tuple[ReasoningNode, dict[str, Any]]] = field(default_factory=list)   # (new, existing)
    dropped: list[ReasoningNode] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"stored={len(self.stored)}, "
            f"merged={len(self.merged)}, "
            f"dropped={len(self.dropped)}"
        )


class SemanticDeduplicator:
    """
    Before storing new nodes, check if near-identical nodes already exist.

    Strategy:
    - If similarity > threshold: the new node is a near-duplicate.
      - If the new node has higher confidence → replace existing (merge).
      - Otherwise → drop the new node (existing is better).
    - If similarity ≤ threshold: store as new.
    """

    def __init__(self, store: MemoryStore, similarity_threshold: float = _DEFAULT_THRESHOLD):
        self.store = store
        self.threshold = similarity_threshold

    def deduplicate(
        self,
        new_nodes: list[ReasoningNode],
        project_id: str,
        session_id: str,
    ) -> DeduplicationResult:
        result = DeduplicationResult()

        if not new_nodes:
            return result

        # Embed new nodes
        texts = [f"{n.node_type.value}: {n.summary}" for n in new_nodes]
        new_embeddings = self.store.embed(texts)

        # Fetch existing embeddings for this project
        existing_ids, existing_embeddings = self.store.get_embeddings_for_nodes(project_id)

        if not existing_ids:
            # Nothing stored yet — all new nodes go straight in
            result.stored.extend(new_nodes)
            return result

        for node, new_emb in zip(new_nodes, new_embeddings):
            best_sim = 0.0
            best_idx = -1
            for idx, ex_emb in enumerate(existing_embeddings):
                sim = _cosine_similarity(new_emb, ex_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_sim >= self.threshold:
                # Near-duplicate found
                existing_meta = self.store.nodes_col.get(
                    ids=[existing_ids[best_idx]], include=["metadatas"]
                )
                existing_node_meta = existing_meta["metadatas"][0] if existing_meta["metadatas"] else {}

                existing_confidence = existing_node_meta.get("confidence", 0.0)
                if node.confidence > existing_confidence:
                    # New is better — will overwrite (store handles upsert)
                    result.merged.append((node, existing_node_meta))
                    result.stored.append(node)
                else:
                    result.dropped.append(node)
            else:
                result.stored.append(node)

        return result
