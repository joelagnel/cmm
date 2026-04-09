"""Profile quality metrics — staleness and redundancy checks.

Runs automatically as part of the session-stop hook after every profile
rebuild. No API key needed — all checks use local file inspection and
cosine similarity on stored embeddings.

Metrics:
    1. Staleness: how many file paths referenced in insights/pitfalls
       no longer exist in the repo?
    2. Redundancy: pairwise cosine similarity among profile entries
       — are any above 0.85 (near-duplicates that should be merged)?
    3. Coverage: ratio of sessions that contributed to the profile
       vs total ingested sessions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..schemas.memory import CognitiveProfile


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def check_staleness(
    profile: CognitiveProfile,
    project_dir: str | Path,
) -> dict[str, Any]:
    """Check how many file paths referenced in the profile still exist.

    Scans insight components and pitfall/strategy text for paths that
    look like file references (contain / or . with common extensions).
    """
    import re

    project_dir = Path(project_dir)
    # Extract path-like tokens from all profile text
    all_text = " ".join(
        [i.insight + " " + i.component for i in profile.architectural_insights]
        + [p.description + " " + (p.resolution_strategy or "") for p in profile.pitfalls]
        + [s.trigger + " " + " ".join(s.steps) for s in profile.diagnostic_strategies]
    )

    # Match things that look like file paths
    path_pattern = re.compile(
        r'(?:^|[\s`"\'])([a-zA-Z0-9_./\-]+(?:\.(?:py|js|ts|json|yaml|yml|toml|md|sql|sh|css|html))\b)',
        re.MULTILINE,
    )
    referenced_paths = set(m.group(1) for m in path_pattern.finditer(all_text))

    existing = 0
    missing = 0
    missing_paths: list[str] = []

    for ref in referenced_paths:
        # Try to resolve relative to project_dir
        candidate = project_dir / ref
        if candidate.exists():
            existing += 1
        else:
            missing += 1
            missing_paths.append(ref)

    total = existing + missing
    staleness_ratio = missing / total if total > 0 else 0.0

    return {
        "total_references": total,
        "existing": existing,
        "missing": missing,
        "missing_paths": missing_paths,
        "staleness_ratio": round(staleness_ratio, 3),
    }


def check_redundancy(
    profile: CognitiveProfile,
    embed_fn: Any = None,
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Check pairwise similarity among profile entries.

    If embed_fn is provided, uses it to embed the texts. Otherwise
    falls back to a simple word-overlap heuristic (less accurate but
    no API needed).
    """
    # Collect all profile entry texts
    texts: list[tuple[str, str]] = []  # (label, text)
    for i, ins in enumerate(profile.architectural_insights):
        texts.append((f"insight:{i}", ins.insight))
    for i, p in enumerate(profile.pitfalls):
        texts.append((f"pitfall:{i}", p.description))
    for i, s in enumerate(profile.diagnostic_strategies):
        texts.append((f"strategy:{i}", s.trigger))

    if len(texts) < 2:
        return {"total_entries": len(texts), "redundant_pairs": [], "redundancy_ratio": 0.0}

    # Embed if possible, otherwise fall back to word overlap
    if embed_fn:
        raw_texts = [t for _, t in texts]
        embeddings = embed_fn(raw_texts)
    else:
        embeddings = None

    redundant_pairs: list[dict] = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if embeddings:
                sim = _cosine_sim(embeddings[i], embeddings[j])
            else:
                # Word overlap fallback
                words_i = set(texts[i][1].lower().split())
                words_j = set(texts[j][1].lower().split())
                union = words_i | words_j
                sim = len(words_i & words_j) / len(union) if union else 0.0

            if sim >= threshold:
                redundant_pairs.append({
                    "entry_a": texts[i][0],
                    "entry_b": texts[j][0],
                    "similarity": round(sim, 3),
                    "text_a": texts[i][1][:100],
                    "text_b": texts[j][1][:100],
                })

    total_pairs = len(texts) * (len(texts) - 1) // 2
    redundancy_ratio = len(redundant_pairs) / total_pairs if total_pairs > 0 else 0.0

    return {
        "total_entries": len(texts),
        "redundant_pairs": redundant_pairs,
        "redundancy_ratio": round(redundancy_ratio, 3),
    }


def check_coverage(profile: CognitiveProfile, total_sessions: int) -> dict[str, Any]:
    """What fraction of ingested sessions contributed to the profile?"""
    if total_sessions == 0:
        return {"profile_sessions": profile.session_count, "total_sessions": 0, "coverage_ratio": 0.0}
    ratio = profile.session_count / total_sessions
    return {
        "profile_sessions": profile.session_count,
        "total_sessions": total_sessions,
        "coverage_ratio": round(min(ratio, 1.0), 3),
    }


def run_quality_checks(
    profile: CognitiveProfile,
    project_dir: str | Path,
    total_sessions: int = 0,
    embed_fn: Any = None,
) -> dict[str, Any]:
    """Run all quality checks and return a combined report."""
    return {
        "staleness": check_staleness(profile, project_dir),
        "redundancy": check_redundancy(profile, embed_fn=embed_fn),
        "coverage": check_coverage(profile, total_sessions),
    }
