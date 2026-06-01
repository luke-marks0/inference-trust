"""
Model name registry and conversion utilities.

Maps HuggingFace model names to provider-specific model names (Fireworks, OpenRouter).
"""

# HuggingFace name -> Fireworks verification target.
# Typically uses accounts/fireworks/models/... but can also be
# accounts/<account>/deployments/<deployment-id> for on-demand deployments.
FIREWORKS_MODEL_REGISTRY: dict[str, str] = {
    # Llama models
    "meta-llama/Llama-3.3-70B-Instruct": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    # Kimi models
    "moonshotai/Kimi-K2-Thinking": "accounts/fireworks/models/kimi-k2-thinking",
    "moonshotai/Kimi-K2.5": "accounts/fireworks/models/kimi-k2p5",
    "moonshotai/Kimi-K2.6": "accounts/fireworks/models/kimi-k2p6",
    # Qwen models
    "Qwen/Qwen2.5-72B-Instruct": "accounts/fireworks/models/qwen2p5-72b-instruct",
    "Qwen/Qwen3-8B": "accounts/fireworks/models/qwen3-8b",
    "Qwen/Qwen3-32B": "accounts/fireworks/models/qwen3-32b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    # DeepSeek models
    "deepseek-ai/DeepSeek-V4-Flash": "accounts/fireworks/models/deepseek-v4-flash",
    "deepseek-ai/DeepSeek-V3.2": "accounts/fireworks/models/deepseek-v3p2",
    # MiniMax models
    "MiniMaxAI/MiniMax-M2.1": "accounts/fireworks/models/minimax-m2p1",
    "MiniMaxAI/MiniMax-M2.5": "accounts/fireworks/models/minimax-m2p5",
    "MiniMaxAI/MiniMax-M2.7": "accounts/fireworks/models/minimax-m2p7",
    # Z.ai models
    "zai-org/GLM-4.6": "accounts/fireworks/models/glm-4p6",
    "zai-org/GLM-4.7": "accounts/fireworks/models/glm-4p7",
    # OpenAI OSS models
    "openai/gpt-oss-20b": "accounts/fireworks/models/gpt-oss-20b",
    "openai/gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
}

# HuggingFace name -> OpenRouter name (only for models that differ from hf_name.lower())
OPENROUTER_MODEL_REGISTRY: dict[str, str] = {
    "Qwen/Qwen3-32B": "qwen/qwen3-32b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen/qwen3-235b-a22b-2507",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "qwen/qwen3-coder",
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek/deepseek-v4-flash",
    "deepseek-ai/DeepSeek-V3.2": "deepseek/deepseek-v3.2",
    "MiniMaxAI/MiniMax-M2.1": "minimax/minimax-m2.1",
    "MiniMaxAI/MiniMax-M2.5": "minimax/minimax-m2.5",
    "MiniMaxAI/MiniMax-M2.7": "minimax/minimax-m2.7",
    "moonshotai/Kimi-K2-Instruct": "moonshotai/kimi-k2",
    "moonshotai/Kimi-K2.5": "moonshotai/kimi-k2.5",
    "zai-org/GLM-4.6": "z-ai/glm-4.6",
    "zai-org/GLM-4.7": "z-ai/glm-4.7",
}


def _iter_registered_hf_names() -> list[str]:
    """Return all registered canonical HuggingFace model names in stable order."""
    names = list(FIREWORKS_MODEL_REGISTRY.keys())
    for name in OPENROUTER_MODEL_REGISTRY.keys():
        if name not in FIREWORKS_MODEL_REGISTRY:
            names.append(name)
    return names


def register_fireworks_model(hf_name: str, fireworks_name: str) -> None:
    """
    Register a HuggingFace to Fireworks model name mapping.

    Args:
        hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        fireworks_name: Fireworks verification target. Can be a serverless model
            path (e.g., "accounts/fireworks/models/llama-v3p1-8b-instruct") or
            an on-demand deployment path
            (e.g., "accounts/<account>/deployments/<deployment-id>").
    """
    FIREWORKS_MODEL_REGISTRY[hf_name] = fireworks_name


def register_openrouter_model(hf_name: str, openrouter_name: str) -> None:
    """
    Register a HuggingFace to OpenRouter model name mapping.

    Only needed for models where the OpenRouter name differs from hf_name.lower().

    Args:
        hf_name: HuggingFace model name (e.g., "Qwen/Qwen3-235B-A22B-Instruct-2507")
        openrouter_name: OpenRouter model name (e.g., "qwen/qwen3-235b-a22b-2507")
    """
    OPENROUTER_MODEL_REGISTRY[hf_name] = openrouter_name


def get_openrouter_name(hf_name: str) -> str:
    """
    Get the OpenRouter model name for a HuggingFace model.

    If the model is in OPENROUTER_MODEL_REGISTRY, returns the registered name.
    Otherwise, returns hf_name.lower() as the default.

    Args:
        hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        OpenRouter model name (e.g., "meta-llama/llama-3.1-8b-instruct")
    """
    if hf_name in OPENROUTER_MODEL_REGISTRY:
        return OPENROUTER_MODEL_REGISTRY[hf_name]
    return hf_name.lower()


def resolve_hf_name(model_name: str) -> str:
    """Resolve model aliases/openrouter names to a canonical HuggingFace name.

    Resolution order:
    1. Exact registered HuggingFace name
    2. Case-insensitive match against registered HuggingFace names
    3. Match against registered OpenRouter model names
    4. Fall back to the original value
    """
    if model_name in FIREWORKS_MODEL_REGISTRY or model_name in OPENROUTER_MODEL_REGISTRY:
        return model_name

    lowered = model_name.lower()

    for hf_name in _iter_registered_hf_names():
        if hf_name.lower() == lowered:
            return hf_name

    for hf_name in _iter_registered_hf_names():
        if get_openrouter_name(hf_name) == lowered:
            return hf_name

    return model_name


def get_fireworks_name(model_name: str) -> str:
    """Get Fireworks model name from a HuggingFace/OpenRouter model identifier."""
    resolved = resolve_hf_name(model_name)
    if resolved not in FIREWORKS_MODEL_REGISTRY:
        raise ValueError(
            f"Model {model_name} not in FIREWORKS_MODEL_REGISTRY. "
            f"Use register_fireworks_model() to add it first."
        )
    return FIREWORKS_MODEL_REGISTRY[resolved]


def guess_fireworks_name(hf_name: str) -> str:
    """
    Attempt to convert a HuggingFace model name to Fireworks format using heuristics.

    This is a best-effort guess - always verify against actual Fireworks availability.
    The heuristics applied:
    1. Extract model name after the org/ prefix
    2. Convert to lowercase
    3. Replace dots with 'p' (e.g., "3.1" -> "3p1")

    Args:
        hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        Guessed Fireworks model name (e.g., "accounts/fireworks/models/llama-3p1-8b-instruct")
    """
    # Extract just the model name (after org/)
    if "/" in hf_name:
        model_name = hf_name.split("/", 1)[1]
    else:
        model_name = hf_name

    # Lowercase
    model_name = model_name.lower()

    # Replace dots with 'p'
    model_name = model_name.replace(".", "p")

    return f"accounts/fireworks/models/{model_name}"
