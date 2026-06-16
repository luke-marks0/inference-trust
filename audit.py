# ruff: noqa: E402

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

TOKEN_DIFR_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOKEN_DIFR_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_dotenv()

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from token_difr import (
    FireworksVerificationError,
    audit_provider,
    collect_provider_sequences,
    construct_prompts,
    list_openrouter_providers,
    verify_provider_sequences,
)
from token_difr.api import verify_outputs_fireworks, verify_outputs_openai_compatible
from token_difr.common import TokenSequence, compute_metrics_summary
from token_difr.model_registry import get_fireworks_name, guess_fireworks_name, resolve_hf_name

# Audit parameters
N_PROMPTS = 100
MAX_TOKENS = 200
SEED = 42
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.0
FIREWORKS_MGMT_BASE_URL = "https://api.fireworks.ai"
FIREWORKS_API_USER_AGENT = "token-difr-audit/1.0"
STATE_FILE = Path(TOKEN_DIFR_ROOT) / "state" / "servers.json"
CONFIG_DIR = Path(TOKEN_DIFR_ROOT) / "configs"
SENSITIVE_PARAMETER_FIELDS = {
    "fireworks_on_demand_deployment",
    "fireworks_serverless_model",
    "fireworks_base_model_for_deployment",
}
SENSITIVE_PROVIDER_FIELDS = {
    "fireworks_verification_target",
}
ORG_IDENTIFIER_PATTERN = re.compile(r"\borg_[A-Za-z0-9_-]+\b")
COMPLETION_IDENTIFIER_PATTERN = re.compile(r"\bcmpl-[A-Za-z0-9_-]+\b")
DEPLOYMENT_PATH_PATTERN = re.compile(
    r"accounts/[A-Za-z0-9._-]+/deployments/[A-Za-z0-9._-]+"
)


def _normalize_openai_base_url(raw_base_url: str, *, ensure_v1_path: bool) -> str:
    parsed = urllib.parse.urlparse(raw_base_url.strip())
    path = parsed.path.rstrip("/")
    if ensure_v1_path and not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    if not path:
        path = "/"
    return urllib.parse.urlunparse(parsed._replace(path=path))


def _split_openai_base_url_and_query(base_url: str) -> tuple[str, dict[str, str]]:
    parsed = urllib.parse.urlparse(base_url.strip())
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query: dict[str, str] = {key: value for key, value in query_pairs}
    clean_base_url = urllib.parse.urlunparse(parsed._replace(query=""))
    return clean_base_url, query


def _create_async_openai_client(*, api_key: str, base_url: str) -> AsyncOpenAI:
    clean_base_url, default_query = _split_openai_base_url_and_query(base_url)
    if default_query:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=clean_base_url,
            default_query=default_query,
        )
    return AsyncOpenAI(api_key=api_key, base_url=clean_base_url)


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Warning: invalid {name}={raw!r}; using default {default}.")
        return default


DEFAULT_VAST_VERIFICATION_CONCURRENCY = _get_env_int(
    "TOKEN_DIFR_VAST_VERIFICATION_CONCURRENCY",
    10,
)


def _sanitize_name(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "server"


def _to_int(value: Any, default: int, *, min_value: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _to_float(value: Any, default: float, *, min_value: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _extract_account_id(account_ref: str) -> str:
    ref = account_ref.strip()
    if ref.startswith("accounts/"):
        return ref.split("/", 1)[1]
    return ref


def _extract_deployment_parts(deployment_ref: str) -> tuple[str | None, str]:
    ref = deployment_ref.strip()
    if "/deployments/" in ref:
        left, deployment_id = ref.rsplit("/deployments/", 1)
        account_id = None
        if left.startswith("accounts/"):
            account_id = left.split("/", 1)[1]
        return account_id, deployment_id
    return None, ref


def _fireworks_request(
    method: str,
    path: str,
    api_key: str,
    *,
    payload: dict | None = None,
) -> dict:
    url = f"{FIREWORKS_MGMT_BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": FIREWORKS_API_USER_AGENT,
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "/v1/accounts" in path:
            raise RuntimeError(
                "Fireworks API access denied for account discovery "
                f"({method} {path}, HTTP 403). This is usually account-scope or WAF policy, not a bad key. "
                "Set FIREWORKS_ACCOUNT_ID to bypass /v1/accounts listing, or use "
                "--fireworks-create-deployment-cmd / --fireworks-verification-model. "
                f"Response body: {body}"
            ) from exc
        raise RuntimeError(f"Fireworks API {method} {path} failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Fireworks API {method} {path} failed: {exc}") from exc

    if not body:
        return {}
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}
    except json.JSONDecodeError:
        return {}


def _resolve_fireworks_account_id(api_key: str) -> str:
    env_account = os.environ.get("FIREWORKS_ACCOUNT_ID") or os.environ.get("FIREWORKS_ACCOUNT")
    if env_account:
        return _extract_account_id(env_account)

    payload = _fireworks_request("GET", "/v1/accounts", api_key)
    raw_accounts = payload.get("accounts")
    if not isinstance(raw_accounts, list) or not raw_accounts:
        raise RuntimeError(
            "Unable to resolve Fireworks account automatically. Set FIREWORKS_ACCOUNT_ID."
        )

    account_ids: list[str] = []
    for account in raw_accounts:
        if not isinstance(account, dict):
            continue
        candidate = account.get("name") or account.get("id") or account.get("accountId")
        if isinstance(candidate, str) and candidate.strip():
            account_ids.append(_extract_account_id(candidate))

    if not account_ids:
        raise RuntimeError(
            "Unable to resolve Fireworks account automatically. Set FIREWORKS_ACCOUNT_ID."
        )

    unique_ids = sorted(set(account_ids))
    if len(unique_ids) > 1:
        print(
            f"Multiple Fireworks accounts available ({', '.join(unique_ids)}); using {unique_ids[0]}. "
            "Set FIREWORKS_ACCOUNT_ID to choose a different one."
        )
    return unique_ids[0]


def _get_fireworks_deployment_shape_name(api_key: str, base_model: str) -> str | None:
    filter_str = f'snapshot.base_model="{base_model}" AND latest_validated=true'
    params = urllib.parse.urlencode({"filter": filter_str, "order_by": "create_time desc"})
    path = f"/v1/accounts/-/deploymentShapes/-/versions?{params}"
    try:
        payload = _fireworks_request("GET", path, api_key)
    except Exception as exc:
        print(f"Warning: unable to fetch deployment shapes ({exc}); continuing without shape.")
        return None
    versions = payload.get("deploymentShapeVersions")
    if not isinstance(versions, list) or not versions:
        return None
    version_name = versions[0].get("name", "")
    if not version_name or "/versions/" not in version_name:
        return None
    shape_name = "/".join(version_name.split("/")[:-2])
    print(f"Found deployment shape: {shape_name}")
    return shape_name


def _create_temp_fireworks_deployment_via_api(
    *,
    api_key: str,
    account_id: str,
    base_model: str,
    hf_model: str,
) -> str:
    print("Creating temporary Fireworks deployment via API...")

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in hf_model).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:32] or "model"
    display_name = f"audit-{slug}-{timestamp}"[:63].rstrip("-")

    encoded_account = urllib.parse.quote(account_id, safe="")
    path = f"/v1/accounts/{encoded_account}/deployments"
    base_payload = {
        "baseModel": base_model,
        "minReplicaCount": 0,
        "maxReplicaCount": 1,
        "displayName": display_name,
    }

    payload_candidates: list[dict] = []
    seen_payloads: set[str] = set()

    def add_payload(payload: dict) -> bool:
        key = json.dumps(payload, sort_keys=True)
        if key in seen_payloads:
            return False
        seen_payloads.add(key)
        payload_candidates.append(payload)
        return True

    shape_name = _get_fireworks_deployment_shape_name(api_key, base_model)
    if shape_name:
        add_payload({**base_payload, "deploymentShape": shape_name})

    known_accelerators = [
        "NVIDIA_B200_180GB",
        "NVIDIA_H200_141GB",
        "NVIDIA_H100_80GB",
        "AMD_MI300X_192GB",
    ]
    for precision in ("FP8_MM", "FP8_MM_V2", "FP8", "FP8_V2", "PRECISION_UNSPECIFIED"):
        for accelerator in known_accelerators:
            add_payload(
                {
                    **base_payload,
                    "precision": precision,
                    "acceleratorType": accelerator,
                    "acceleratorCount": 1,
                }
            )
    add_payload(
        {
            **base_payload,
            "precision": "FP4_BLOCKSCALED_MM",
            "acceleratorType": "NVIDIA_B200_180GB",
            "acceleratorCount": 1,
        }
    )

    last_error: Exception | None = None
    idx = 0
    while idx < len(payload_candidates):
        payload = payload_candidates[idx]
        idx += 1
        try:
            response = _fireworks_request("POST", path, api_key, payload=payload)
            deployment_name = response.get("name")
            if not isinstance(deployment_name, str) or "/deployments/" not in deployment_name:
                deployment_name = f"accounts/{account_id}/deployments/{display_name}"
            print(f"Created temporary deployment: {deployment_name}")
            return deployment_name
        except Exception as exc:
            last_error = exc
            payload_desc = {k: payload[k] for k in ("precision", "acceleratorType", "acceleratorCount", "deploymentShape") if k in payload}
            print(f"Create deployment attempt failed with payload {payload_desc or '{default}'}: {exc}")

            message = str(exc)
            min_count_match = re.search(r"minimum accelerators required for model is (\d+)", message)
            if min_count_match and "acceleratorType" in payload:
                required_count = int(min_count_match.group(1))
                current_count = payload.get("acceleratorCount", 1)
                if isinstance(current_count, int) and current_count < required_count:
                    adjusted_payload = {**payload, "acceleratorCount": required_count}
                    key = json.dumps(adjusted_payload, sort_keys=True)
                    if key not in seen_payloads:
                        seen_payloads.add(key)
                        print(
                            "Retrying deployment create with higher accelerator count: "
                            f"{payload.get('acceleratorType')} x {required_count} "
                            f"(precision={payload.get('precision')})"
                        )
                        payload_candidates.insert(idx, adjusted_payload)

            precision_accel_match = re.search(
                r"precision ([A-Z0-9_]+) can only be used with (.+?) accelerators",
                message,
            )
            if precision_accel_match:
                precision = precision_accel_match.group(1)
                accelerators_raw = precision_accel_match.group(2)
                for accelerator in re.findall(r"[A-Z]+_[A-Z0-9]+_[0-9]+GB", accelerators_raw):
                    add_payload(
                        {
                            **base_payload,
                            "precision": precision,
                            "acceleratorType": accelerator,
                            "acceleratorCount": 1,
                        }
                    )

    raise RuntimeError(f"Failed to create temporary deployment after retries: {last_error}")


def _delete_temp_fireworks_deployment_via_api(*, api_key: str, deployment: str, fallback_account_id: str) -> None:
    parsed_account_id, deployment_id = _extract_deployment_parts(deployment)
    account_id = parsed_account_id or fallback_account_id

    print(f"Deleting temporary deployment via API: {deployment}")
    encoded_account = urllib.parse.quote(account_id, safe="")
    encoded_deployment = urllib.parse.quote(deployment_id, safe="")
    query = urllib.parse.urlencode({"ignoreChecks": "true"})
    path = f"/v1/accounts/{encoded_account}/deployments/{encoded_deployment}?{query}"
    _fireworks_request("DELETE", path, api_key)
    print("Deleted temporary deployment")


def _wait_for_temp_fireworks_deployment_ready_via_api(
    *,
    api_key: str,
    deployment: str,
    fallback_account_id: str,
    timeout_seconds: int = 1200,
    poll_interval_seconds: int = 10,
) -> None:
    parsed_account_id, deployment_id = _extract_deployment_parts(deployment)
    account_id = parsed_account_id or fallback_account_id
    encoded_account = urllib.parse.quote(account_id, safe="")
    encoded_deployment = urllib.parse.quote(deployment_id, safe="")
    path = f"/v1/accounts/{encoded_account}/deployments/{encoded_deployment}"

    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        payload = _fireworks_request("GET", path, api_key)

        state = str(payload.get("state") or "").upper()
        replica_count = payload.get("replicaCount")
        desired_replica_count = payload.get("desiredReplicaCount")
        status = payload.get("status")

        status_code = ""
        status_message = ""
        if isinstance(status, dict):
            status_code = str(status.get("code") or "")
            status_message = str(status.get("message") or "")

        if isinstance(replica_count, int) and replica_count > 0:
            print(
                "Temporary deployment is ready: "
                f"state={state or 'UNKNOWN'}, replicas={replica_count}"
            )
            return

        if state in {"READY", "RUNNING", "ACTIVE", "DEPLOYED"}:
            print(f"Temporary deployment is ready: state={state}")
            return

        if state in {"FAILED", "ERROR", "DELETED"}:
            detail = status_message or status_code or "unknown error"
            raise RuntimeError(
                f"Temporary deployment entered terminal state {state}: {detail}"
            )

        elapsed = time.time() - start
        if elapsed >= timeout_seconds:
            detail = status_message or status_code or "still creating"
            raise RuntimeError(
                "Timed out waiting for temporary deployment readiness after "
                f"{int(elapsed)}s (state={state or 'UNKNOWN'}, detail={detail})"
            )

        if attempt == 1 or attempt % 3 == 0:
            replicas_now = replica_count if isinstance(replica_count, int) else "?"
            replicas_target = desired_replica_count if isinstance(desired_replica_count, int) else "?"
            detail = status_message or status_code or "creating"
            print(
                "Waiting for temporary deployment readiness: "
                f"state={state or 'UNKNOWN'}, replicas={replicas_now}/{replicas_target}, detail={detail}"
            )

        time.sleep(poll_interval_seconds)


def save_results(results: dict, output_file: str) -> None:
    sanitized_results = _sanitize_results_for_public_output(results)
    with open(output_file, "w") as f:
        json.dump(sanitized_results, f, indent=2)


def _is_error_field_name(field_name: str) -> bool:
    return field_name == "error" or field_name.endswith("_error")


def _redact_error_text(text: str) -> str:
    redacted = ORG_IDENTIFIER_PATTERN.sub("redacted", text)
    redacted = COMPLETION_IDENTIFIER_PATTERN.sub("redacted", redacted)
    redacted = DEPLOYMENT_PATH_PATTERN.sub("accounts/redacted/deployments/redacted", redacted)
    return redacted


def _sanitize_results_for_public_output(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if parent_key == "parameters" and key in SENSITIVE_PARAMETER_FIELDS:
                continue
            if key in SENSITIVE_PROVIDER_FIELDS:
                continue
            if isinstance(item, str) and _is_error_field_name(key):
                sanitized[key] = _redact_error_text(item)
                continue
            sanitized[key] = _sanitize_results_for_public_output(item, parent_key=key)
        return sanitized

    if isinstance(value, list):
        return [_sanitize_results_for_public_output(item, parent_key=parent_key) for item in value]

    return value


# ---------------------------------------------------------------------------
# Collected-token persistence helpers
# ---------------------------------------------------------------------------


def _update_collected_tokens_file(
    path: str,
    hf_model: str,
    collected: dict[str, tuple[list[TokenSequence], int]],
    conversations: list[list[dict[str, str]]],
) -> None:
    """Upsert collected provider sequences for a model into a JSON file."""
    payload: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            payload = {}

    payload.setdefault("version", 1)
    payload.setdefault("models", {})

    now = datetime.utcnow().isoformat()
    providers_data: dict = {}
    for provider, (sequences, vocab_size) in collected.items():
        providers_data[provider] = {
            "sequences": [s.to_dict() for s in sequences],
            "vocab_size": vocab_size,
            "collected_at": now,
        }

    payload["models"][hf_model] = {
        "conversations": conversations,
        "providers": providers_data,
        "collected_at": now,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    total_seqs = sum(len(d["sequences"]) for d in providers_data.values())
    print(
        f"Saved collected tokens for {hf_model} "
        f"({len(providers_data)} providers, {total_seqs} sequences) to {path}"
    )


def _load_model_collected_tokens(
    path: str,
    hf_model: str,
) -> dict[str, tuple[list[TokenSequence], int]] | None:
    """Load collected provider sequences for a model from a JSON file.

    Returns dict[provider -> (sequences, vocab_size)], or None if the model
    is not present in the file.
    """
    with open(path) as f:
        payload = json.load(f)

    model_data = payload.get("models", {}).get(hf_model)
    if model_data is None:
        return None

    result: dict[str, tuple[list[TokenSequence], int]] = {}
    for provider, pdata in model_data.get("providers", {}).items():
        sequences = [TokenSequence.from_dict(s) for s in pdata["sequences"]]
        vocab_size = int(pdata["vocab_size"])
        result[provider] = (sequences, vocab_size)
    return result


# ---------------------------------------------------------------------------
# vast.ai server lifecycle helpers (called from _main_vast)
# ---------------------------------------------------------------------------


def _read_vast_state() -> dict[str, dict]:
    if not STATE_FILE.exists():
        return {}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    vast_servers = payload.get("vast_servers")
    if not isinstance(vast_servers, dict):
        return {}
    return {k: v for k, v in vast_servers.items() if isinstance(k, str) and isinstance(v, dict)}


def _start_vast_verification_server_for_model(
    *,
    hf_model: str,
    vast_gpu: str | None,
    vast_num_gpus: int | None,
    vast_disk_gb: float | None,
    vast_max_price: float | None,
    vast_on_demand: bool = False,
) -> tuple[str, str]:
    """Start a vast.ai vLLM server via serve.py and return (server_name, base_url)."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    server_name = f"audit-{_sanitize_name(hf_model)}-{timestamp}"
    serve_script = os.path.join(TOKEN_DIFR_ROOT, "serve.py")
    command = [sys.executable, serve_script, "vast", "start", "--name", server_name, "--model", hf_model]
    if vast_gpu:
        command.extend(["--gpu", vast_gpu])
    if vast_num_gpus is not None:
        command.extend(["--num-gpus", str(vast_num_gpus)])
    if vast_disk_gb is not None:
        command.extend(["--disk-gb", str(vast_disk_gb)])
    if vast_max_price is not None:
        command.extend(["--max-price", str(vast_max_price)])
    if vast_on_demand:
        command.append("--on-demand")

    print(f"Starting vast.ai verification server {server_name}...")
    stderr_lines: list[str] = []
    proc = subprocess.Popen(command, text=True, stdout=None, stderr=subprocess.PIPE)
    assert proc.stderr is not None
    for line in proc.stderr:
        line = line.rstrip()
        stderr_lines.append(line)
        print(line, flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to start vast.ai server {server_name!r} (exit {proc.returncode}). "
            f"stderr: {' '.join(stderr_lines[-10:])}"
        )

    vast_servers = _read_vast_state()
    entry = vast_servers.get(server_name)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Vast server {server_name!r} started but was not found in state file {STATE_FILE}."
        )

    base_url = entry.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimeError(f"Vast server {server_name!r} has no base URL recorded in state.")

    normalized_base_url = _normalize_openai_base_url(base_url, ensure_v1_path=True)
    return server_name, normalized_base_url


def _stop_vast_verification_server_by_name(server_name: str) -> None:
    if not server_name.strip():
        return
    serve_script = os.path.join(TOKEN_DIFR_ROOT, "serve.py")
    command = [sys.executable, serve_script, "vast", "stop", "--name", server_name]
    print(f"Destroying vast verification server {server_name}...")
    stderr_lines: list[str] = []
    proc = subprocess.Popen(command, text=True, stdout=None, stderr=subprocess.PIPE)
    assert proc.stderr is not None
    for line in proc.stderr:
        stderr_lines.append(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to stop vast server {server_name!r} (exit {proc.returncode}). "
            f"stderr: {' '.join(stderr_lines[-10:])}"
        )
    print(f"Vast verification server {server_name} destroyed.")


# ---------------------------------------------------------------------------
# Reference metrics computation
# ---------------------------------------------------------------------------


def _compute_reference_metrics(
    model_name: str,
    sequences: list[TokenSequence],
    verification_backend: str = "fireworks",
    fireworks_on_demand_deployment: str | None = None,
    local_verification_base_url: str | None = None,
    local_verification_model: str | None = None,
) -> dict:
    model_name = resolve_hf_name(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vocab_size = len(tokenizer)
    except Exception as tokenizer_error:
        print(f"Warning: tokenizer load failed ({tokenizer_error}); reading vocab_size from tokenizer.json")
        from huggingface_hub import hf_hub_download
        _tj_path = hf_hub_download(model_name, "tokenizer.json")
        with open(_tj_path) as _f:
            _tj = json.load(_f)
        _vocab = _tj.get("model", {}).get("vocab", {})
        vocab_size = max(_vocab.values()) + 1 if isinstance(_vocab, dict) and _vocab else 0
        for _tok in _tj.get("added_tokens", []):
            if isinstance(_tok, dict):
                _tid = _tok.get("id", -1)
                if isinstance(_tid, int) and _tid >= vocab_size:
                    vocab_size = _tid + 1
        if vocab_size == 0:
            raise RuntimeError(
                f"Unable to determine vocab_size for {model_name}: tokenizer load failed "
                f"and tokenizer.json fallback returned 0"
            ) from tokenizer_error
        print(f"  Vocab size from tokenizer.json: {vocab_size}")

    async def _verify_fireworks(target_model: str):
        fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
        if not fireworks_api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable not set")
        fireworks_client = AsyncOpenAI(
            api_key=fireworks_api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )
        return await verify_outputs_fireworks(
            sequences,
            vocab_size=vocab_size,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            seed=SEED,
            client=fireworks_client,
            model=target_model,
            topk_logprobs=5,
        )

    async def _verify_vast(target_model: str, base_url: str):
        vast_api_key = os.environ.get("VAST_VERIFICATION_API_KEY") or "vast-verification"
        vast_client = _create_async_openai_client(
            api_key=vast_api_key,
            base_url=base_url,
        )
        return await verify_outputs_openai_compatible(
            sequences,
            vocab_size=vocab_size,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            seed=SEED,
            client=vast_client,
            model=target_model,
            topk_logprobs=5,
            backend_label="vast vLLM",
            request_extra_body={"return_tokens_as_token_ids": True},
            concurrency=DEFAULT_VAST_VERIFICATION_CONCURRENCY,
        )

    backend = verification_backend.strip().lower()

    if backend == "vast":
        raw_base_url = local_verification_base_url or os.environ.get("VAST_VERIFICATION_BASE_URL")
        if not raw_base_url:
            raise ValueError(
                "Vast reference verification requires --vast-verification-base-url "
                "or VAST_VERIFICATION_BASE_URL."
            )
        normalized_base_url = _normalize_openai_base_url(raw_base_url, ensure_v1_path=True)
        target_model = local_verification_model or model_name
        results_tokens = asyncio.run(_verify_vast(target_model, normalized_base_url))
        summary = compute_metrics_summary(results_tokens)
        summary["n_sequences"] = len(sequences)
        summary["verification_backend"] = "vast"
        summary["verification_target"] = target_model
        summary["vast_verification_base_url"] = normalized_base_url
        return summary

    if backend != "fireworks":
        raise ValueError(f"Unsupported verification backend: {verification_backend}")

    try:
        serverless_model = get_fireworks_name(model_name)
    except Exception as mapping_error:
        if not fireworks_on_demand_deployment:
            raise
        print(f"No serverless mapping for {model_name}: {mapping_error}")
        print(f"Using on-demand deployment for reference verification: {fireworks_on_demand_deployment}")
        results_tokens = asyncio.run(_verify_fireworks(fireworks_on_demand_deployment))
        summary = compute_metrics_summary(results_tokens)
        summary["n_sequences"] = len(sequences)
        summary["fireworks_verification_target"] = fireworks_on_demand_deployment
        summary["fireworks_verification_mode"] = "on-demand"
        summary["serverless_error"] = str(mapping_error)
        return summary

    try:
        results_tokens = asyncio.run(_verify_fireworks(serverless_model))
        verification_target = serverless_model
        verification_mode = "serverless"
    except Exception as serverless_error:
        if not fireworks_on_demand_deployment:
            raise
        print(f"Serverless reference verification failed ({serverless_model}): {serverless_error}")
        print(f"Retrying reference verification with on-demand deployment: {fireworks_on_demand_deployment}")
        results_tokens = asyncio.run(_verify_fireworks(fireworks_on_demand_deployment))
        verification_target = fireworks_on_demand_deployment
        verification_mode = "on-demand"

    summary = compute_metrics_summary(results_tokens)
    summary["n_sequences"] = len(sequences)
    summary["fireworks_verification_target"] = verification_target
    summary["fireworks_verification_mode"] = verification_mode
    return summary


# ---------------------------------------------------------------------------
# vast.ai audit path
# ---------------------------------------------------------------------------


def _main_vast(
    models: list[str],
    use_reference_tokens: bool,
    vast_verification_base_url: str | None,
    vast_verification_model: str | None,
    vast_stop_after_verification: bool,
    vast_gpu: str | None,
    vast_num_gpus: int | None,
    vast_disk_gb: float | None,
    vast_max_price: float | None,
    vast_on_demand: bool = False,
    collect_only: bool = False,
    save_collected_tokens: str | None = None,
    from_collected_tokens: str | None = None,
    profile_timing: bool = False,
) -> None:
    raw_base_url = vast_verification_base_url or os.environ.get("VAST_VERIFICATION_BASE_URL")
    fixed_base_url = (
        _normalize_openai_base_url(raw_base_url, ensure_v1_path=True)
        if raw_base_url and str(raw_base_url).strip()
        else None
    )
    auto_manage = fixed_base_url is None
    if auto_manage and not collect_only:
        print(
            "No vast.ai verification base URL provided. "
            "Auto-managing per-model vast.ai verification servers."
        )

    for requested_model in models:
        hf_model = resolve_hf_name(requested_model)
        if hf_model != requested_model:
            print(f"Resolved model alias: {requested_model} -> {hf_model}")

        try:
            providers = list_openrouter_providers(hf_model)
        except Exception as exc:
            print(f"Failed to list providers for {hf_model}: {exc}")
            continue
        if not providers:
            print(f"No providers listed for {hf_model}")
            continue

        verification_target = vast_verification_model or hf_model

        # Load prompts / reference tokens — no GPU needed.
        reference_sequences: list[TokenSequence] = []
        if use_reference_tokens:
            prompts, reference_sequences = _load_reference_bundle(hf_model)
            print(f"Loaded {len(reference_sequences)} reference sequences")
        else:
            prompts = construct_prompts(
                n_prompts=N_PROMPTS,
                model_name=hf_model,
                system_prompt="You are a helpful assistant.",
            )
            print(f"Constructed {len(prompts)} prompts")

        # Set up the results file early so progress is visible from the first save,
        # mirroring the fireworks path.
        results: dict = {
            "model": hf_model,
            "parameters": {
                "n_prompts": N_PROMPTS,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "temperature": TEMPERATURE,
                "verification_backend": "vast",
                "vast_verification_target": verification_target,
                "vast_verification_base_url": fixed_base_url,
                "vast_verification_strategy": (
                    "auto-managed-per-model" if auto_manage else "fixed-base-url"
                ),
                "vast_server_name": None,
            },
            "providers": {},
        }

        safe_model_name = hf_model.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "audit_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{safe_model_name}_audit_results_{timestamp}.json"

        save_results(results, output_file)
        print(f"Results will be saved to {output_file}")

        # ---------------------------------------------------------------
        # Phase 1: Collect provider token sequences (no vast instance).
        # ---------------------------------------------------------------
        collected_provider_sequences: dict[str, tuple[list[TokenSequence], int]] = {}

        if from_collected_tokens:
            print(f"\nLoading pre-collected tokens for {hf_model} from {from_collected_tokens} ...")
            loaded = _load_model_collected_tokens(from_collected_tokens, hf_model)
            if loaded is None:
                print(f"  Model {hf_model} not found in {from_collected_tokens}; skipping.")
                continue
            for provider in providers:
                if provider in loaded:
                    seqs, vocab_size = loaded[provider]
                    collected_provider_sequences[provider] = (seqs, vocab_size)
                    token_count = sum(len(s.output_token_ids) for s in seqs)
                    results["providers"][provider] = {
                        "collection_complete": True,
                        "collected_sequences": len(seqs),
                        "collected_tokens": token_count,
                    }
                else:
                    print(f"  Provider {provider} not in collected tokens file, skipping.")
            save_results(results, output_file)
        else:
            for provider in providers:
                print(f"\nCollecting tokens for provider: {provider}")
                try:
                    sequences, vocab_size = collect_provider_sequences(
                        prompts,
                        model=hf_model,
                        provider=provider,
                        max_tokens=MAX_TOKENS,
                        seed=SEED,
                        temperature=TEMPERATURE,
                    )
                    collected_provider_sequences[provider] = (sequences, vocab_size)
                    token_count = sum(len(s.output_token_ids) for s in sequences)
                    results["providers"][provider] = {
                        "collection_complete": True,
                        "collected_sequences": len(sequences),
                        "collected_tokens": token_count,
                    }
                    print(f"  Collected {token_count} tokens across {len(sequences)} sequences")
                except Exception as collection_error:
                    print(f"  ERROR during token collection: {collection_error}")
                    results["providers"][provider] = {
                        "error": str(collection_error),
                        "collection_complete": False,
                    }
                save_results(results, output_file)

        if save_collected_tokens and not from_collected_tokens:
            _update_collected_tokens_file(
                save_collected_tokens, hf_model, collected_provider_sequences, prompts
            )

        if collect_only:
            print(f"Collection complete for {hf_model}. Skipping verification (--collect-only).")
            continue

        if not collected_provider_sequences:
            print(f"No provider sequences collected for {hf_model}; skipping verification.")
            continue

        # ---------------------------------------------------------------
        # Phase 2: Start vast instance (billing begins here).
        # ---------------------------------------------------------------
        verification_base_url = fixed_base_url
        vast_server_name: str | None = None

        # Optional timing profile (enabled via --profile-timing):
        # (download weights + start vLLM server) vs auditing.
        _profile_server_seconds = 0.0
        _profile_audit_seconds = 0.0

        try:
            if auto_manage:
                _profile_server_start = time.time()
                vast_server_name, verification_base_url = (
                    _start_vast_verification_server_for_model(
                        hf_model=hf_model,
                        vast_gpu=vast_gpu,
                        vast_num_gpus=vast_num_gpus,
                        vast_disk_gb=vast_disk_gb,
                        vast_max_price=vast_max_price,
                        vast_on_demand=vast_on_demand,
                    )
                )
                _profile_server_seconds = time.time() - _profile_server_start
                results["parameters"]["vast_server_name"] = vast_server_name
                results["parameters"]["vast_verification_base_url"] = verification_base_url
                print(
                    f"Using auto-managed vast verification endpoint for {hf_model}: {verification_base_url}"
                )
                save_results(results, output_file)

            if not verification_base_url:
                raise ValueError(
                    "Vast verification base URL is not available. "
                    "Pass --vast-verification-base-url or configure auto-managed mode."
                )

            if use_reference_tokens:
                reference_metrics = _compute_reference_metrics(
                    hf_model,
                    reference_sequences,
                    verification_backend="vast",
                    local_verification_base_url=verification_base_url,
                    local_verification_model=verification_target,
                )
                results["reference"] = reference_metrics
                save_results(results, output_file)

            # ---------------------------------------------------------------
            # Phase 3: Verify all pre-collected sequences (vast is now running).
            # ---------------------------------------------------------------
            _profile_audit_start = time.time()
            for provider in providers:
                if provider not in collected_provider_sequences:
                    continue
                sequences, vocab_size = collected_provider_sequences[provider]
                print(f"\nVerifying provider: {provider}")
                try:
                    result = verify_provider_sequences(
                        sequences,
                        vocab_size=vocab_size,
                        model=hf_model,
                        seed=SEED,
                        top_k=TOP_K,
                        top_p=TOP_P,
                        temperature=TEMPERATURE,
                        verification_backend="vast",
                        verification_base_url=verification_base_url,
                        verification_model=verification_target,
                    )
                    provider_results = asdict(result)
                    provider_results["verification_backend"] = "vast"
                    provider_results["verification_target"] = verification_target
                    results["providers"][provider] = provider_results

                    print(f"  Total tokens: {result.total_tokens}")
                    print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                    print(f"  Avg probability: {result.avg_prob:.4f}")
                except Exception as provider_error:
                    print(f"  ERROR: {provider_error}")
                    results["providers"][provider] = {
                        "error": str(provider_error),
                        "verification_backend": "vast",
                        "verification_target": verification_target,
                    }

                save_results(results, output_file)

            _profile_audit_seconds = time.time() - _profile_audit_start

            print(f"\nAll results saved to {output_file}")

            if profile_timing:
                def _fmt_secs(s: float) -> str:
                    return f"{s:.1f}s ({s / 60:.1f} min)"

                print("\n=== TIMING PROFILE for", hf_model, "===")
                print(f"  Download weights + start vLLM server: {_fmt_secs(_profile_server_seconds)}")
                print(f"  Auditing (verification):              {_fmt_secs(_profile_audit_seconds)}")
                print(
                    f"  Total:                                "
                    f"{_fmt_secs(_profile_server_seconds + _profile_audit_seconds)}"
                )
                print("=" * (len("=== TIMING PROFILE for ") + len(hf_model) + 4))
        finally:
            if auto_manage and vast_server_name and vast_stop_after_verification:
                try:
                    _stop_vast_verification_server_by_name(vast_server_name)
                except Exception as teardown_error:
                    print(
                        f"Warning: failed to destroy vast server {vast_server_name}: {teardown_error}"
                    )


# ---------------------------------------------------------------------------
# fireworks audit path (unchanged)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit providers for one or more Hugging Face model names.",
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more Hugging Face model names (e.g. Qwen/Qwen3-235B-A22B-Instruct-2507).",
    )
    parser.add_argument(
        "--reference-tokens",
        action="store_true",
        help="Use saved reference token sequences per model from reference_tokens/.",
    )
    parser.add_argument(
        "--fireworks-verification-model",
        "--fireworks-deployment",
        dest="fireworks_on_demand_deployment",
        default=None,
        help=(
            "Optional Fireworks on-demand deployment to use as fallback. If omitted, "
            "audit.py will auto-create a temporary deployment when serverless "
            "verification is unavailable."
        ),
    )
    parser.add_argument(
        "--fireworks-create-deployment-cmd",
        default=None,
        help=(
            "Optional shell command override to create a temporary Fireworks deployment "
            "for each audited model. Command output must include the deployment path "
            "(accounts/<account>/deployments/<deployment-id>) on stdout. "
            "Supported placeholders: {model}, {fireworks_model}."
        ),
    )
    parser.add_argument(
        "--fireworks-delete-deployment-cmd",
        default=None,
        help=(
            "Optional shell command override to delete a temporary Fireworks deployment "
            "after each model audit. Supported placeholders: {deployment}, "
            "{model}, {fireworks_model}."
        ),
    )
    parser.add_argument(
        "--verification-backend",
        choices=("fireworks", "vast"),
        default="fireworks",
        help="Verification backend to use for provider audits and reference checks.",
    )
    # vast.ai verification flags
    parser.add_argument(
        "--vast-verification-base-url",
        default=None,
        help=(
            "OpenAI-compatible vast.ai vLLM base URL. If omitted, audit.py auto-manages "
            "a per-model vast.ai instance and destroys it after each model "
            "(unless --no-vast-stop-after-verification is set)."
        ),
    )
    parser.add_argument(
        "--vast-verification-model",
        default=None,
        help=(
            "Optional model identifier sent to the vast.ai verification backend. "
            "Defaults to the audited HuggingFace model name."
        ),
    )
    parser.add_argument(
        "--vast-gpu",
        default=None,
        help="GPU type to search for on vast.ai (e.g. H100_SXM4_80GB). Overrides model config.",
    )
    parser.add_argument(
        "--vast-num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to rent. Defaults to tensor_parallel_size from model config.",
    )
    parser.add_argument(
        "--vast-disk-gb",
        type=float,
        default=None,
        help="Minimum disk space in GB for the vast.ai instance.",
    )
    parser.add_argument(
        "--vast-max-price",
        type=float,
        default=None,
        help="Maximum price per hour (USD) for the vast.ai instance.",
    )
    parser.add_argument(
        "--vast-stop-after-verification",
        dest="vast_stop_after_verification",
        action="store_true",
        default=True,
        help="Destroy the vast.ai instance after each model audit (default: enabled).",
    )
    parser.add_argument(
        "--no-vast-stop-after-verification",
        dest="vast_stop_after_verification",
        action="store_false",
        help="Leave the vast.ai instance running after the audit.",
    )
    parser.add_argument(
        "--vast-on-demand",
        dest="vast_on_demand",
        action="store_true",
        default=False,
        help=(
            "Search on-demand vast.ai offers instead of interruptible (spot) instances. "
            "Interruptible instances are cheaper but can be reclaimed; on-demand are stable."
        ),
    )
    # Token collection / verification decoupling
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Collect provider token sequences and save them (requires --save-collected-tokens), "
            "then exit without running verification. No GPU instance is started."
        ),
    )
    parser.add_argument(
        "--save-collected-tokens",
        default=None,
        metavar="PATH",
        help=(
            "After collecting provider tokens, write them to this JSON file. "
            "Multiple runs append/update the file. Use with --collect-only to "
            "pre-collect tokens for all models before any GPU instance starts."
        ),
    )
    parser.add_argument(
        "--from-collected-tokens",
        default=None,
        metavar="PATH",
        help=(
            "Skip provider token collection; load previously saved sequences from "
            "this JSON file (written by --save-collected-tokens). Only the "
            "verification phase runs, minimising GPU instance uptime."
        ),
    )
    parser.add_argument(
        "--profile-timing",
        dest="profile_timing",
        action="store_true",
        default=False,
        help=(
            "Print a timing breakdown (weight download + vLLM startup vs. "
            "verification) for each model after the audit completes."
        ),
    )
    return parser.parse_args()


def _run_shell_command(
    command_template: str,
    *,
    context: str,
    model: str,
    fireworks_model: str,
    deployment: str | None = None,
) -> str:
    format_values = {
        "model": model,
        "fireworks_model": fireworks_model,
        "deployment": deployment or "",
    }
    try:
        command = command_template.format(**format_values)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Unknown placeholder {{{missing}}} in {context} command template") from exc

    completed = subprocess.run(command, shell=True, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{context} command failed (exit {completed.returncode}): {command}\n"
            f"stdout: {completed.stdout.strip()}\n"
            f"stderr: {completed.stderr.strip()}"
        )

    return completed.stdout


def _create_temp_fireworks_deployment(
    create_command: str,
    *,
    model: str,
    fireworks_model: str,
) -> str:
    print("Creating temporary Fireworks deployment for this audit...")
    stdout = _run_shell_command(
        create_command,
        context="Deployment create",
        model=model,
        fireworks_model=fireworks_model,
    )
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            "Deployment create command succeeded but returned no output. "
            "Expected deployment path on stdout."
        )
    deployment = lines[-1]
    if "/deployments/" not in deployment:
        raise RuntimeError(
            "Deployment create command output does not look like a deployment path: "
            f"{deployment}"
        )
    print(f"Created temporary deployment: {deployment}")
    return deployment


def _delete_temp_fireworks_deployment(
    delete_command: str,
    *,
    deployment: str,
    model: str,
    fireworks_model: str,
) -> None:
    print(f"Deleting temporary deployment: {deployment}")
    _run_shell_command(
        delete_command,
        context="Deployment delete",
        model=model,
        fireworks_model=fireworks_model,
        deployment=deployment,
    )
    print("Deleted temporary deployment")


def _load_reference_bundle(model_name: str) -> tuple[list[list[dict[str, str]]], list[TokenSequence]]:
    model_name = resolve_hf_name(model_name)
    safe_model_name = model_name.replace("/", "_")
    reference_path = os.path.join(TOKEN_DIFR_ROOT, "reference_tokens", f"{safe_model_name}.json")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference tokens not found for {model_name}: {reference_path}")
    with open(reference_path, "r") as f:
        payload = json.load(f)
    conversations = payload.get("conversations")
    sequences_raw = payload.get("sequences", payload)
    if not isinstance(conversations, list) or not all(isinstance(c, list) for c in conversations):
        raise ValueError(f"Reference file missing conversations for {model_name}: {reference_path}")
    if not isinstance(sequences_raw, list):
        raise ValueError(f"Invalid reference token format in {reference_path}")
    sequences = [TokenSequence.from_dict(s) for s in sequences_raw]
    return conversations, sequences


def main(
    models: list[str],
    use_reference_tokens: bool,
    fireworks_on_demand_deployment: str | None = None,
    fireworks_create_deployment_cmd: str | None = None,
    fireworks_delete_deployment_cmd: str | None = None,
    verification_backend: str = "fireworks",
    vast_verification_base_url: str | None = None,
    vast_verification_model: str | None = None,
    vast_stop_after_verification: bool = True,
    vast_gpu: str | None = None,
    vast_num_gpus: int | None = None,
    vast_disk_gb: float | None = None,
    vast_max_price: float | None = None,
    vast_on_demand: bool = False,
    collect_only: bool = False,
    save_collected_tokens: str | None = None,
    from_collected_tokens: str | None = None,
    profile_timing: bool = False,
) -> None:
    backend = verification_backend.strip().lower()
    if backend == "vast":
        _main_vast(
            models=models,
            use_reference_tokens=use_reference_tokens,
            vast_verification_base_url=vast_verification_base_url,
            vast_verification_model=vast_verification_model,
            vast_stop_after_verification=vast_stop_after_verification,
            vast_gpu=vast_gpu,
            vast_num_gpus=vast_num_gpus,
            vast_disk_gb=vast_disk_gb,
            vast_max_price=vast_max_price,
            vast_on_demand=vast_on_demand,
            collect_only=collect_only,
            save_collected_tokens=save_collected_tokens,
            from_collected_tokens=from_collected_tokens,
            profile_timing=profile_timing,
        )
        return
    if backend != "fireworks":
        raise ValueError(f"Unsupported verification backend: {verification_backend}")

    auto_create_when_needed = fireworks_on_demand_deployment is None and fireworks_create_deployment_cmd is None

    for requested_model in models:
        HF_MODEL = resolve_hf_name(requested_model)
        if HF_MODEL != requested_model:
            print(f"Resolved model alias: {requested_model} -> {HF_MODEL}")

        fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")

        serverless_fireworks_model: str | None = None
        serverless_mapping_error = None
        try:
            serverless_fireworks_model = get_fireworks_name(HF_MODEL)
        except Exception as mapping_error:
            serverless_mapping_error = str(mapping_error)
            print(f"No Fireworks serverless mapping for {HF_MODEL}: {mapping_error}")

        base_fireworks_model = serverless_fireworks_model or guess_fireworks_name(HF_MODEL)
        if not serverless_fireworks_model:
            print(f"Guessed Fireworks base model for deployment creation: {base_fireworks_model}")

        model_fallback_deployment = fireworks_on_demand_deployment
        created_deployment_for_model = False
        created_deployment_via_api = False
        deployment_account_id: str | None = None
        verification_mode = "serverless"

        def ensure_fallback_deployment(reason: str) -> str:
            nonlocal model_fallback_deployment
            nonlocal created_deployment_for_model
            nonlocal created_deployment_via_api
            nonlocal deployment_account_id

            if model_fallback_deployment:
                return model_fallback_deployment

            if fireworks_create_deployment_cmd:
                print(f"{reason} Creating fallback deployment with --fireworks-create-deployment-cmd.")
                model_fallback_deployment = _create_temp_fireworks_deployment(
                    fireworks_create_deployment_cmd,
                    model=HF_MODEL,
                    fireworks_model=base_fireworks_model,
                )
                parsed_account, _ = _extract_deployment_parts(model_fallback_deployment)
                deployment_account_id = parsed_account
                created_deployment_for_model = True
                if fireworks_api_key:
                    if not deployment_account_id:
                        deployment_account_id = _resolve_fireworks_account_id(fireworks_api_key)
                    _wait_for_temp_fireworks_deployment_ready_via_api(
                        api_key=fireworks_api_key,
                        deployment=model_fallback_deployment,
                        fallback_account_id=deployment_account_id,
                        timeout_seconds=_get_env_int(
                            "FIREWORKS_DEPLOYMENT_READY_TIMEOUT_SECONDS",
                            1200,
                        ),
                        poll_interval_seconds=_get_env_int(
                            "FIREWORKS_DEPLOYMENT_READY_POLL_SECONDS",
                            10,
                        ),
                    )
                return model_fallback_deployment

            if not auto_create_when_needed:
                raise RuntimeError("No fallback deployment is available")

            if not fireworks_api_key:
                raise ValueError("FIREWORKS_API_KEY environment variable not set")

            if not deployment_account_id:
                deployment_account_id = _resolve_fireworks_account_id(fireworks_api_key)

            print(f"{reason} Creating fallback deployment via Fireworks API.")
            model_fallback_deployment = _create_temp_fireworks_deployment_via_api(
                api_key=fireworks_api_key,
                account_id=deployment_account_id,
                base_model=base_fireworks_model,
                hf_model=HF_MODEL,
            )
            created_deployment_for_model = True
            created_deployment_via_api = True
            _wait_for_temp_fireworks_deployment_ready_via_api(
                api_key=fireworks_api_key,
                deployment=model_fallback_deployment,
                fallback_account_id=deployment_account_id,
                timeout_seconds=_get_env_int(
                    "FIREWORKS_DEPLOYMENT_READY_TIMEOUT_SECONDS",
                    1200,
                ),
                poll_interval_seconds=_get_env_int(
                    "FIREWORKS_DEPLOYMENT_READY_POLL_SECONDS",
                    10,
                ),
            )
            return model_fallback_deployment

        def recycle_fallback_deployment_for_retry(reason: str) -> str:
            nonlocal model_fallback_deployment
            nonlocal created_deployment_for_model
            nonlocal created_deployment_via_api
            nonlocal deployment_account_id

            if model_fallback_deployment and not created_deployment_for_model:
                print(
                    f"{reason} Fallback deployment is externally managed; "
                    "retrying once with the same deployment."
                )
                return model_fallback_deployment

            if model_fallback_deployment and created_deployment_for_model:
                failing_deployment = model_fallback_deployment
                print(f"{reason} Deleting failing on-demand deployment: {failing_deployment}")
                try:
                    if fireworks_delete_deployment_cmd:
                        _delete_temp_fireworks_deployment(
                            fireworks_delete_deployment_cmd,
                            deployment=failing_deployment,
                            model=HF_MODEL,
                            fireworks_model=base_fireworks_model,
                        )
                    else:
                        if not fireworks_api_key:
                            raise ValueError("FIREWORKS_API_KEY environment variable not set")
                        if deployment_account_id is None:
                            deployment_account_id = _resolve_fireworks_account_id(fireworks_api_key)
                        _delete_temp_fireworks_deployment_via_api(
                            api_key=fireworks_api_key,
                            deployment=failing_deployment,
                            fallback_account_id=deployment_account_id,
                        )
                except Exception as delete_error:
                    print(
                        "Warning: failed to delete failing on-demand deployment "
                        f"{failing_deployment}: {delete_error}"
                    )
                finally:
                    if model_fallback_deployment == failing_deployment:
                        model_fallback_deployment = None
                    created_deployment_for_model = False
                    created_deployment_via_api = False

            return ensure_fallback_deployment(
                f"{reason} Creating fallback deployment for one final retry."
            )

        if model_fallback_deployment:
            print(f"Using Fireworks on-demand fallback deployment: {model_fallback_deployment}")

        try:
            try:
                providers = list_openrouter_providers(HF_MODEL)
            except Exception as exc:
                print(f"Failed to list providers for {HF_MODEL}: {exc}")
                continue
            if not providers:
                print(f"No providers listed for {HF_MODEL}")
                continue

            results = {
                "model": HF_MODEL,
                "parameters": {
                    "n_prompts": N_PROMPTS,
                    "max_tokens": MAX_TOKENS,
                    "seed": SEED,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "temperature": TEMPERATURE,
                    "verification_backend": "fireworks",
                    "fireworks_verification_mode": verification_mode,
                    "fireworks_verification_strategy": "fixed-per-audit",
                    "fireworks_on_demand_deployment": model_fallback_deployment,
                    "fireworks_deployment_created_for_audit": created_deployment_for_model,
                    "fireworks_serverless_model": serverless_fireworks_model,
                    "fireworks_base_model_for_deployment": base_fireworks_model,
                    "fireworks_serverless_mapping_error": serverless_mapping_error,
                },
                "providers": {},
            }

            prompts = None
            reference_metrics = None
            if use_reference_tokens:
                prompts, reference_sequences = _load_reference_bundle(HF_MODEL)
                print(f"Loaded {len(reference_sequences)} reference sequences")
                try:
                    reference_metrics = _compute_reference_metrics(
                        HF_MODEL,
                        reference_sequences,
                        fireworks_on_demand_deployment=model_fallback_deployment,
                    )
                except Exception as reference_error:
                    if model_fallback_deployment:
                        raise
                    try:
                        fallback = ensure_fallback_deployment(
                            "Reference verification could not use serverless."
                        )
                    except Exception:
                        raise reference_error
                    print(f"Retrying reference verification with on-demand deployment: {fallback}")
                    reference_metrics = _compute_reference_metrics(
                        HF_MODEL,
                        reference_sequences,
                        fireworks_on_demand_deployment=fallback,
                    )
                    reference_metrics["serverless_error"] = str(reference_error)
                results["reference"] = reference_metrics
                ref_mode = reference_metrics.get("fireworks_verification_mode")
                if ref_mode == "on-demand":
                    verification_mode = "on-demand"
                    ref_target = reference_metrics.get("fireworks_verification_target")
                    if isinstance(ref_target, str) and ref_target:
                        model_fallback_deployment = ref_target
                    print("Locking verification mode to on-demand for this audit.")
                results["parameters"]["fireworks_on_demand_deployment"] = model_fallback_deployment
                results["parameters"]["fireworks_deployment_created_for_audit"] = created_deployment_for_model
                results["parameters"]["fireworks_verification_mode"] = verification_mode
            else:
                prompts = construct_prompts(
                    n_prompts=N_PROMPTS,
                    model_name=HF_MODEL,
                    system_prompt="You are a helpful assistant.",
                )
                print(f"Constructed {len(prompts)} prompts")

            safe_model_name = HF_MODEL.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "audit_results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{safe_model_name}_audit_results_{timestamp}.json"

            save_results(results, output_file)
            print(f"Results will be saved to {output_file}")

            collected_provider_sequences: dict[str, tuple[list[TokenSequence], int]] = {}
            skip_remaining_providers = False

            if from_collected_tokens:
                print(f"\nLoading pre-collected tokens for {HF_MODEL} from {from_collected_tokens} ...")
                loaded = _load_model_collected_tokens(from_collected_tokens, HF_MODEL)
                if loaded is None:
                    print(f"  Model {HF_MODEL} not found in {from_collected_tokens}; skipping.")
                    continue
                for provider in providers:
                    if provider in loaded:
                        seqs, vocab_size = loaded[provider]
                        collected_provider_sequences[provider] = (seqs, vocab_size)
                        token_count = sum(len(s.output_token_ids) for s in seqs)
                        results["providers"][provider] = {
                            "collection_complete": True,
                            "collected_sequences": len(seqs),
                            "collected_tokens": token_count,
                        }
                    else:
                        print(f"  Provider {provider} not in collected tokens file, skipping verification.")
                save_results(results, output_file)
            else:
                for provider in providers:
                    print(f"\nCollecting tokens for provider: {provider}")
                    try:
                        sequences, vocab_size = collect_provider_sequences(
                            prompts,
                            model=HF_MODEL,
                            provider=provider,
                            max_tokens=MAX_TOKENS,
                            seed=SEED,
                            temperature=TEMPERATURE,
                        )
                        collected_provider_sequences[provider] = (sequences, vocab_size)
                        token_count = sum(len(sequence.output_token_ids) for sequence in sequences)
                        results["providers"][provider] = {
                            "collection_complete": True,
                            "collected_sequences": len(sequences),
                            "collected_tokens": token_count,
                        }
                        print(f"  Collected {token_count} tokens across {len(sequences)} sequences")
                    except Exception as provider_error:
                        print(f"  ERROR during token collection: {provider_error}")
                        results["providers"][provider] = {"error": str(provider_error), "collection_complete": False}

                    save_results(results, output_file)

            if save_collected_tokens and not from_collected_tokens:
                _update_collected_tokens_file(
                    save_collected_tokens, HF_MODEL, collected_provider_sequences, prompts
                )

            if collect_only:
                print(f"Collection complete for {HF_MODEL}. Skipping verification (--collect-only).")
                continue

            for provider in providers:
                if provider not in collected_provider_sequences:
                    continue

                sequences, vocab_size = collected_provider_sequences[provider]
                print(f"\nVerifying provider: {provider}")
                try:
                    if verification_mode == "on-demand":
                        if not model_fallback_deployment:
                            raise RuntimeError("on-demand verification mode selected but no deployment is available")
                        result = verify_provider_sequences(
                            sequences,
                            vocab_size=vocab_size,
                            model=HF_MODEL,
                            seed=SEED,
                            top_k=TOP_K,
                            top_p=TOP_P,
                            temperature=TEMPERATURE,
                            fireworks_verification_model=model_fallback_deployment,
                        )
                        provider_results = asdict(result)
                        provider_results["fireworks_verification_mode"] = "on-demand"
                        provider_results["fireworks_verification_target"] = model_fallback_deployment
                        results["providers"][provider] = provider_results
                    else:
                        result = verify_provider_sequences(
                            sequences,
                            vocab_size=vocab_size,
                            model=HF_MODEL,
                            seed=SEED,
                            top_k=TOP_K,
                            top_p=TOP_P,
                            temperature=TEMPERATURE,
                        )
                        provider_results = asdict(result)
                        provider_results["fireworks_verification_mode"] = "serverless"
                        provider_results["fireworks_verification_target"] = serverless_fireworks_model
                        results["providers"][provider] = provider_results

                    print(f"  Total tokens: {result.total_tokens}")
                    print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                    print(f"  Avg probability: {result.avg_prob:.4f}")

                except FireworksVerificationError as serverless_error:
                    if verification_mode == "on-demand":
                        print(f"  On-demand verification failed: {serverless_error}")
                        try:
                            fallback = recycle_fallback_deployment_for_retry(
                                "On-demand verification failed."
                            )
                            results["parameters"]["fireworks_on_demand_deployment"] = fallback
                            results["parameters"]["fireworks_deployment_created_for_audit"] = (
                                created_deployment_for_model
                            )
                            print(f"  Retrying once with on-demand deployment: {fallback}")
                            result = verify_provider_sequences(
                                sequences,
                                vocab_size=vocab_size,
                                model=HF_MODEL,
                                seed=SEED,
                                top_k=TOP_K,
                                top_p=TOP_P,
                                temperature=TEMPERATURE,
                                fireworks_verification_model=fallback,
                            )
                            provider_results = asdict(result)
                            provider_results["fireworks_verification_mode"] = "on-demand"
                            provider_results["fireworks_verification_target"] = fallback
                            provider_results["on_demand_first_error"] = str(serverless_error)
                            results["providers"][provider] = provider_results
                            print(f"  Total tokens: {result.total_tokens}")
                            print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                            print(f"  Avg probability: {result.avg_prob:.4f}")
                        except Exception as final_on_demand_error:
                            print(
                                "  ERROR: on-demand retry after deployment recycle failed: "
                                f"{final_on_demand_error}"
                            )
                            results["providers"][provider] = {
                                "error": str(final_on_demand_error),
                                "on_demand_first_error": str(serverless_error),
                                "fireworks_verification_mode": "on-demand-fallback-failed",
                                "fireworks_verification_target": model_fallback_deployment,
                            }
                            results["parameters"]["model_skipped_after_provider"] = provider
                            results["parameters"]["model_skipped_reason"] = (
                                "on-demand verification failed after one recycle/retry"
                            )
                            skip_remaining_providers = True

                    else:
                        if not model_fallback_deployment:
                            try:
                                ensure_fallback_deployment("Serverless verification failed.")
                                results["parameters"]["fireworks_on_demand_deployment"] = model_fallback_deployment
                                results["parameters"]["fireworks_deployment_created_for_audit"] = (
                                    created_deployment_for_model
                                )
                            except Exception as create_error:
                                print(f"  ERROR: {serverless_error}")
                                print(f"  ERROR: unable to create fallback deployment: {create_error}")
                                results["providers"][provider] = {
                                    "error": str(serverless_error),
                                    "fallback_error": str(create_error),
                                }
                                save_results(results, output_file)
                                continue

                        verification_mode = "on-demand"
                        results["parameters"]["fireworks_verification_mode"] = verification_mode
                        print(f"  Serverless verification failed: {serverless_error}")
                        print("  Switching verification mode to on-demand for remaining providers.")
                        print(f"  Retrying with on-demand deployment: {model_fallback_deployment}")
                        try:
                            result = verify_provider_sequences(
                                sequences,
                                vocab_size=vocab_size,
                                model=HF_MODEL,
                                seed=SEED,
                                top_k=TOP_K,
                                top_p=TOP_P,
                                temperature=TEMPERATURE,
                                fireworks_verification_model=model_fallback_deployment,
                            )
                            provider_results = asdict(result)
                            provider_results["fireworks_verification_mode"] = "on-demand"
                            provider_results["fireworks_verification_target"] = model_fallback_deployment
                            provider_results["serverless_error"] = str(serverless_error)
                            results["providers"][provider] = provider_results

                            print(f"  Total tokens: {result.total_tokens}")
                            print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                            print(f"  Avg probability: {result.avg_prob:.4f}")
                            print("  Verification target: on-demand deployment fallback")
                        except Exception as on_demand_error:
                            print(f"  On-demand fallback failed: {on_demand_error}")
                            try:
                                fallback = recycle_fallback_deployment_for_retry(
                                    "On-demand fallback failed."
                                )
                                results["parameters"]["fireworks_on_demand_deployment"] = fallback
                                results["parameters"]["fireworks_deployment_created_for_audit"] = (
                                    created_deployment_for_model
                                )
                                print(f"  Retrying once with on-demand deployment: {fallback}")
                                result = verify_provider_sequences(
                                    sequences,
                                    vocab_size=vocab_size,
                                    model=HF_MODEL,
                                    seed=SEED,
                                    top_k=TOP_K,
                                    top_p=TOP_P,
                                    temperature=TEMPERATURE,
                                    fireworks_verification_model=fallback,
                                )
                                provider_results = asdict(result)
                                provider_results["fireworks_verification_mode"] = "on-demand"
                                provider_results["fireworks_verification_target"] = fallback
                                provider_results["serverless_error"] = str(serverless_error)
                                provider_results["on_demand_first_error"] = str(on_demand_error)
                                results["providers"][provider] = provider_results
                                print(f"  Total tokens: {result.total_tokens}")
                                print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                                print(f"  Avg probability: {result.avg_prob:.4f}")
                                print("  Verification target: recycled on-demand deployment")
                            except Exception as final_on_demand_error:
                                print(
                                    "  ERROR: on-demand retry after deployment recycle failed: "
                                    f"{final_on_demand_error}"
                                )
                                results["providers"][provider] = {
                                    "error": str(final_on_demand_error),
                                    "serverless_error": str(serverless_error),
                                    "on_demand_first_error": str(on_demand_error),
                                    "fireworks_verification_mode": "on-demand-fallback-failed",
                                    "fireworks_verification_target": model_fallback_deployment,
                                }
                                results["parameters"]["model_skipped_after_provider"] = provider
                                results["parameters"]["model_skipped_reason"] = (
                                    "on-demand verification failed after one recycle/retry"
                                )
                                skip_remaining_providers = True
                except Exception as provider_error:
                    print(f"  ERROR: {provider_error}")
                    results["providers"][provider] = {"error": str(provider_error)}

                save_results(results, output_file)
                if skip_remaining_providers:
                    print("  Skipping remaining providers for this model to avoid wasted credits.")
                    break

            print(f"\nAll results saved to {output_file}")
        finally:
            if created_deployment_for_model:
                if fireworks_delete_deployment_cmd:
                    try:
                        _delete_temp_fireworks_deployment(
                            fireworks_delete_deployment_cmd,
                            deployment=model_fallback_deployment,
                            model=HF_MODEL,
                            fireworks_model=base_fireworks_model,
                        )
                    except Exception as delete_error:
                        print(f"Failed to delete temporary deployment {model_fallback_deployment}: {delete_error}")
                else:
                    try:
                        if not fireworks_api_key:
                            raise ValueError("FIREWORKS_API_KEY environment variable not set")
                        if deployment_account_id is None:
                            deployment_account_id = _resolve_fireworks_account_id(fireworks_api_key)
                        _delete_temp_fireworks_deployment_via_api(
                            api_key=fireworks_api_key,
                            deployment=model_fallback_deployment,
                            fallback_account_id=deployment_account_id,
                        )
                    except Exception as delete_error:
                        if created_deployment_via_api:
                            print(f"Failed to delete temporary deployment {model_fallback_deployment}: {delete_error}")
                        else:
                            print(
                                "Temporary deployment was created for this audit but automatic API deletion failed: "
                                f"{delete_error}"
                            )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.models,
        args.reference_tokens,
        args.fireworks_on_demand_deployment,
        args.fireworks_create_deployment_cmd,
        args.fireworks_delete_deployment_cmd,
        args.verification_backend,
        args.vast_verification_base_url,
        args.vast_verification_model,
        args.vast_stop_after_verification,
        args.vast_gpu,
        args.vast_num_gpus,
        args.vast_disk_gb,
        args.vast_max_price,
        vast_on_demand=args.vast_on_demand,
        collect_only=args.collect_only,
        save_collected_tokens=args.save_collected_tokens,
        from_collected_tokens=args.from_collected_tokens,
        profile_timing=args.profile_timing,
    )
