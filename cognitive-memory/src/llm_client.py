"""Centralized LLM client using LiteLLM.

Supports two providers:
    - Direct Anthropic API (ANTHROPIC_API_KEY set)
    - Amazon Bedrock (AWS credentials set)

An explicit CMM_LLM_MODEL env var overrides all auto-detection.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import litellm

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Default model names (without provider prefix)
DEFAULT_EXTRACTION_MODEL: str = "claude-sonnet-4-5"
DEFAULT_PROFILE_MODEL: str = "claude-sonnet-4-6"

# Amazon Bedrock inference profile mapping (newer models require inference profiles)
_BEDROCK_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
}

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


# ── Private helpers ────────────────────────────────────────────────────────


def _detect_provider() -> str:
    """Detect the LLM provider from environment variables.

    Returns:
        "anthropic" if ANTHROPIC_API_KEY is set,
        "bedrock" if AWS credentials are set,
        "anthropic" as fallback.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        return "bedrock"

    # Default to anthropic — litellm will raise a clear error if no key
    return "anthropic"


def _resolve_model(
    model_override: Optional[str] = None,
    default_model: str = DEFAULT_EXTRACTION_MODEL,
) -> str:
    """Resolve the full LiteLLM model string.

    Priority:
        1. model_override parameter (if provided)
        2. CMM_LLM_MODEL env var (if set)
        3. Auto-detect provider + default_model

    Args:
        model_override: explicit model string (e.g. "bedrock/anthropic.claude-sonnet-4-5-v1")
        default_model: base model name without provider prefix

    Returns:
        Full LiteLLM model string like "anthropic/claude-sonnet-4-5"
        or "bedrock/anthropic.claude-sonnet-4-5-v1"
    """
    # Explicit override takes priority
    if model_override:
        return model_override

    # Env var override
    env_model = os.environ.get("CMM_LLM_MODEL")
    if env_model:
        return env_model

    # Auto-detect from credentials
    provider = _detect_provider()

    if provider == "bedrock":
        bedrock_id = _BEDROCK_MODEL_MAP.get(default_model, default_model)
        return f"bedrock/{bedrock_id}"

    return f"anthropic/{default_model}"


# ── Public functions ───────────────────────────────────────────────────────


async def llm_complete(
    system: str,
    user_content: str,
    max_tokens: int = 4000,
    model_override: Optional[str] = None,
    default_model: str = DEFAULT_EXTRACTION_MODEL,
) -> str:
    """Send a completion request via LiteLLM and return the text response.

    Args:
        system: system prompt
        user_content: user message content
        max_tokens: maximum tokens in the response
        model_override: explicit LiteLLM model string (skips auto-detect)
        default_model: base model name used when auto-detecting provider

    Returns:
        The text content of the LLM response.

    Raises:
        Exception: if the LLM call fails after LiteLLM retries.
    """
    model = _resolve_model(
        model_override=model_override,
        default_model=default_model,
    )
    logger.debug("LLM request: model=%s, max_tokens=%d", model, max_tokens)

    response = await litellm.acompletion(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )

    text = response.choices[0].message.content
    logger.debug("LLM response: %d chars", len(text) if text else 0)
    return text


def count_tokens_for_text(
    text: str,
    model_override: Optional[str] = None,
    default_model: str = DEFAULT_EXTRACTION_MODEL,
) -> int:
    """Count tokens for a piece of text using LiteLLM's token counter.

    This is a local estimation (no API call), making it faster and free
    compared to the Anthropic SDK's count_tokens endpoint.

    Args:
        text: the text to count tokens for
        model_override: explicit LiteLLM model string
        default_model: base model name for auto-detection

    Returns:
        Estimated token count.
    """
    model = _resolve_model(
        model_override=model_override,
        default_model=default_model,
    )
    try:
        count = litellm.token_counter(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return int(count)
    except Exception:
        # Fallback: rough estimate of 4 chars per token
        return max(1, len(text) // 4)
