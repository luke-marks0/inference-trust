import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import openai
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer
from tqdm import tqdm

from token_difr.common import (
    TokenSequence,
    construct_prompts,
    encode_thinking_response,
    render_conversation_for_tokenization,
)
from token_difr.model_registry import get_openrouter_name

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
OPENROUTER_PROVIDER_ALIASES: dict[str, str] = {
    # Canonical hyphenated IDs.
    "atlas-cloud": "atlas-cloud",
    "google-vertex": "google-vertex",
    # Common compact/spacing aliases observed in registry payloads.
    "atlascloud": "atlas-cloud",
    "google": "google-vertex",
    "googlevertex": "google-vertex",
}


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _extract_reasoning_from_details(details: Any) -> str:
    """Extract plain reasoning text from OpenRouter reasoning_details payloads."""
    if not isinstance(details, list):
        return ""

    parts: list[str] = []
    for entry in details:
        if isinstance(entry, str):
            parts.append(entry)
            continue
        if not isinstance(entry, dict):
            continue

        text = entry.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        if isinstance(text, dict):
            content = text.get("content")
            if isinstance(content, str):
                parts.append(content)
                continue

        content = entry.get("content")
        if isinstance(content, str):
            parts.append(content)

    return "".join(parts)


def _extract_reasoning_text(message: Any) -> str:
    """Extract reasoning text from OpenRouter message extensions."""
    candidates = (
        getattr(message, "reasoning", None),
        getattr(message, "reasoning_content", None),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = model_extra.get(key)
            if isinstance(value, str) and value:
                return value

        details_text = _extract_reasoning_from_details(model_extra.get("reasoning_details"))
        if details_text:
            return details_text

    details = getattr(message, "reasoning_details", None)
    details_text = _extract_reasoning_from_details(details)
    if details_text:
        return details_text

    return ""


def _fetch_registry() -> list[dict]:
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "token-difr-openrouter-registry"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unexpected OpenRouter registry response format")


def _fetch_endpoints(model_id: str) -> list[dict]:
    request = urllib.request.Request(
        OPENROUTER_ENDPOINTS_URL.format(model_id=model_id),
        headers={"User-Agent": "token-difr-openrouter-registry"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
        if isinstance(data, dict) and "endpoints" in data and isinstance(data["endpoints"], list):
            return data["endpoints"]
        if isinstance(data, list):
            return data
        raise ValueError("Unexpected OpenRouter endpoints data format")
    if isinstance(payload, list):
        return payload
    raise ValueError("Unexpected OpenRouter endpoints response format")


def _normalize_provider_entry(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("provider", "name", "id"):
            if key in entry:
                return entry[key]
    return None


def _canonicalize_provider_base(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not cleaned:
        return ""
    compact = cleaned.replace("-", "")
    if cleaned in OPENROUTER_PROVIDER_ALIASES:
        return OPENROUTER_PROVIDER_ALIASES[cleaned]
    if compact in OPENROUTER_PROVIDER_ALIASES:
        return OPENROUTER_PROVIDER_ALIASES[compact]
    return compact


def _normalize_provider_slug(name: str) -> str:
    cleaned = name.strip()
    if " | " in cleaned:
        cleaned = cleaned.split(" | ", 1)[0].strip()
    if "/" in cleaned:
        base, suffix = cleaned.split("/", 1)
        normalized_base = _canonicalize_provider_base(base)
        normalized_suffix = re.sub(r"[^a-z0-9._-]+", "-", suffix.strip().lower()).strip("-")
        if normalized_base and normalized_suffix:
            return f"{normalized_base}/{normalized_suffix}"
        return normalized_base or normalized_suffix
    return _canonicalize_provider_base(cleaned)


def _extract_variant(endpoint: dict) -> str | None:
    unknown_values = {"unknown", "none", "n/a", "na", "null"}
    for key in ("variant", "quantization", "precision", "dtype"):
        value = endpoint.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = value.strip().lower()
            if cleaned in unknown_values:
                return None
            return cleaned
    for key in ("name", "model", "id"):
        value = endpoint.get(key)
        if not isinstance(value, str):
            continue
        match = re.search(r"(fp\d+|int\d+|bf16|f16|f32|fp16)", value.lower())
        if match:
            return match.group(1)
    return None


def _format_endpoint_provider(endpoint: dict) -> str | None:
    for key in ("name", "id"):
        value = endpoint.get(key)
        if isinstance(value, str) and " | " in value:
            provider = _normalize_provider_slug(value)
            variant = _extract_variant(endpoint)
            if variant and f"/{variant}" not in provider:
                return f"{provider}/{variant}"
            return provider

    provider = endpoint.get("provider") or endpoint.get("provider_name")
    name = _normalize_provider_entry(provider) or _normalize_provider_entry(endpoint)
    if not name:
        return None

    slug = _normalize_provider_slug(name)
    variant = _extract_variant(endpoint)
    if variant and f"/{variant}" not in slug:
        return f"{slug}/{variant}"
    return slug


def _extract_providers(model_entry: dict) -> list[str]:
    providers_raw = model_entry.get("providers") or model_entry.get("provider") or []
    if isinstance(providers_raw, dict):
        providers_raw = [providers_raw]
    if not isinstance(providers_raw, list):
        providers_raw = []

    providers = []
    for entry in providers_raw:
        name = _normalize_provider_entry(entry)
        if name:
            providers.append(_normalize_provider_slug(name))

    top_provider = model_entry.get("top_provider")
    if isinstance(top_provider, dict):
        top_name = _normalize_provider_entry(top_provider)
        if top_name:
            providers.append(_normalize_provider_slug(top_name))

    endpoints = model_entry.get("endpoints") or []
    if isinstance(endpoints, dict):
        endpoints = [endpoints]
    if isinstance(endpoints, list):
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            name = _format_endpoint_provider(endpoint)
            if name:
                providers.append(name)

    return sorted(set(providers))


def list_openrouter_providers(model: str) -> list[str]:
    """List OpenRouter providers (with quantization) serving a model.

    Args:
        model: Hugging Face or OpenRouter model name.

    Returns:
        Sorted list of provider strings (e.g., ["deepinfra/fp8", "groq"]).
    """
    openrouter_model = get_openrouter_name(model)
    models = _fetch_registry()

    match = None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == openrouter_model:
            match = entry
            break

    if match is None:
        raise ValueError(f"Model not found in OpenRouter registry: {openrouter_model}")

    providers = _extract_providers(match)
    if providers:
        return providers

    endpoints = _fetch_endpoints(openrouter_model)
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        name = _format_endpoint_provider(endpoint)
        if name:
            providers.append(name)
    return sorted(set(providers))


async def generate_openrouter_responses(
    client: openai.AsyncOpenAI,
    conversations: list[list[dict[str, str]]],
    model: str,
    provider: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    concurrency: int = 8,
    seed: int | None = None,
) -> list[ChatCompletion]:
    """Generate responses for multiple conversations via OpenRouter.

    Args:
        client: OpenRouter AsyncOpenAI client.
        conversations: List of conversations (each is a list of message dicts).
        model: OpenRouter model name.
        provider: Backend provider (e.g., "fireworks", "cerebras").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        concurrency: Number of concurrent requests.
        seed: Optional random seed.

    Returns:
        List of raw ChatCompletion objects in the same order as input conversations.
    """
    semaphore = asyncio.Semaphore(concurrency)
    extra_body: dict = {"provider": {"only": [provider]}}

    async def _request(idx: int, messages: list[dict[str, str]]) -> tuple[int, ChatCompletion | None]:
        async with semaphore:
            for attempt in range(4):
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                        extra_body=extra_body,
                    )
                    return idx, completion
                except openai.RateLimitError:
                    if attempt < 3:
                        await asyncio.sleep(10 * (attempt + 1))
                    else:
                        raise
            return idx, None  # unreachable

    tasks = [asyncio.create_task(_request(i, conv)) for i, conv in enumerate(conversations)]
    results: list[ChatCompletion | None] = [None] * len(conversations)

    try:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
            idx, completion = await fut
            results[idx] = completion
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return results  # type: ignore[return-value]


def tokenize_openrouter_responses(
    conversations: list[list[dict[str, str]]],
    responses: list[ChatCompletion],
    tokenizer,
    max_tokens: int | None = None,
) -> list[TokenSequence]:
    """Convert OpenRouter responses to TokenSequence objects.

    Args:
        conversations: List of input conversations (for prompt tokenization).
        responses: List of raw ChatCompletion objects from generate_openrouter_responses.
        tokenizer: HuggingFace tokenizer for the model.
        max_tokens: Optional maximum tokens for response truncation.

    Returns:
        List of TokenSequence objects.
    """
    outputs = []

    for conv, completion in zip(conversations, responses, strict=True):
        # Extract content and reasoning from response
        message = completion.choices[0].message
        content = message.content or ""
        reasoning = _extract_reasoning_text(message)
        # Avoid duplicating thinking segments when content already carries them inline.
        if reasoning and "<think>" in content and "</think>" in content:
            reasoning = ""

        # Tokenize prompt
        rendered = render_conversation_for_tokenization(
            tokenizer,
            conv,
            add_generation_prompt=True,
            model_name=getattr(tokenizer, "name_or_path", None),
        )
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        # Tokenize response
        response_token_ids = encode_thinking_response(
            content,
            reasoning,
            tokenizer,
            max_tokens,
            model_name=getattr(tokenizer, "name_or_path", None),
        )

        outputs.append(TokenSequence(prompt_token_ids=prompt_token_ids, output_token_ids=response_token_ids))

    return outputs


def save_results(
    conversations: list[list[dict[str, str]]],
    responses: list[ChatCompletion],
    save_path: Path,
    config: dict[str, object],
    model_name: str,
    max_tokens: int,
) -> None:
    """Save responses as JSON in VLLM-style format with tokenized prompts and responses.

    Args:
        conversations: List of input conversations.
        responses: List of raw ChatCompletion objects.
        save_path: Path to save the JSON file.
        config: Configuration dictionary to include in output.
        model_name: HuggingFace model name for tokenizer.
        max_tokens: Maximum tokens for response truncation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    outputs = tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens)
    del tokenizer

    vllm_samples = [
        {"prompt_token_ids": seq.prompt_token_ids, "outputs": [{"token_ids": seq.output_token_ids}]} for seq in outputs
    ]

    payload = {"config": config, "samples": vllm_samples}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Saved {len(outputs)} samples to {save_path}")


async def main():
    model_name = "meta-llama/llama-3.1-8b-instruct"
    max_tokens = 500
    temperature = 0.0
    concurrency = 50

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    for provider in ["cerebras", "hyperbolic", "groq", "siliconflow/fp8", "deepinfra"]:
        save_dir = Path("openrouter_responses")
        n_samples = 2000
        max_ctx_len = 512

        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        conversations = construct_prompts(n_prompts=n_samples, max_ctx_len=max_ctx_len, model_name=model_name)
        print(f"Loaded {len(conversations)} prompts from dataset.")

        responses = await generate_openrouter_responses(
            client,
            conversations,
            model_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        model_tag = _sanitize(f"{provider}_{model_name}")
        save_filename = f"openrouter_{model_tag}_token_difr_prompts_test.json"
        config = {
            "model": model_name,
            "provider": provider,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_samples": n_samples,
            "max_ctx_len": max_ctx_len,
        }
        save_results(
            conversations,
            responses,
            save_dir / save_filename,
            config=config,
            model_name=model_name,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())
