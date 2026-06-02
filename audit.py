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
DEFAULT_MODAL_APP_NAME = os.environ.get("TOKEN_DIFR_MODAL_APP_NAME", "token-difr-vllm")
DEFAULT_MODAL_CLASS_NAME = os.environ.get("TOKEN_DIFR_MODAL_CLASS_NAME", "VllmServer")
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


DEFAULT_MODAL_AUDIT_MAX_MODEL_LEN_CAP = _get_env_int(
    "TOKEN_DIFR_MODAL_AUDIT_MAX_MODEL_LEN_CAP",
    8192,
)
DEFAULT_MODAL_VERIFICATION_CONCURRENCY = _get_env_int(
    "TOKEN_DIFR_MODAL_VERIFICATION_CONCURRENCY",
    1,
)


def _canonicalize_url_for_match(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    normalized_path = parsed.path.rstrip("/")
    query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_items.sort()
    normalized_query = urllib.parse.urlencode(query_items)
    return urllib.parse.urlunparse(parsed._replace(path=normalized_path, query=normalized_query))


def _url_without_query(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    normalized_path = parsed.path.rstrip("/")
    return urllib.parse.urlunparse(parsed._replace(path=normalized_path, query=""))


def _read_modal_state() -> dict[str, dict]:
    if not STATE_FILE.exists():
        return {}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    modal_servers = payload.get("modal_servers")
    if not isinstance(modal_servers, dict):
        return {}
    result: dict[str, dict] = {}
    for name, entry in modal_servers.items():
        if isinstance(name, str) and isinstance(entry, dict):
            result[name] = entry
    return result


def _find_modal_server_name_by_base_url(base_url: str) -> str | None:
    modal_servers = _read_modal_state()
    if not modal_servers:
        return None

    target_full = _canonicalize_url_for_match(base_url)
    target_no_query = _url_without_query(target_full)

    full_matches: list[str] = []
    path_matches: list[str] = []

    for name, entry in modal_servers.items():
        entry_base_url = entry.get("base_url")
        if not isinstance(entry_base_url, str) or not entry_base_url.strip():
            continue
        entry_full = _canonicalize_url_for_match(entry_base_url)
        if entry_full == target_full:
            full_matches.append(name)
            continue
        if _url_without_query(entry_full) == target_no_query:
            path_matches.append(name)

    if len(full_matches) == 1:
        return full_matches[0]
    if len(full_matches) > 1:
        print(
            "Skipping modal teardown: multiple tracked servers matched verification URL exactly "
            f"({', '.join(sorted(full_matches))})."
        )
        return None
    if len(path_matches) == 1:
        return path_matches[0]
    if len(path_matches) > 1:
        print(
            "Skipping modal teardown: multiple tracked servers matched verification host/path "
            f"({', '.join(sorted(path_matches))})."
        )
        return None
    return None


def _stop_modal_verification_server(raw_base_url: str | None) -> None:
    if not raw_base_url or not str(raw_base_url).strip():
        print("Skipping modal teardown: no modal verification base URL provided.")
        return

    normalized_base_url = _normalize_openai_base_url(raw_base_url, ensure_v1_path=True)
    server_name = _find_modal_server_name_by_base_url(normalized_base_url)
    if not server_name:
        print(
            "Skipping modal teardown: could not find a matching tracked modal server for "
            f"{normalized_base_url}. If this endpoint is managed outside serve.py, stop it manually."
        )
        return

    serve_script = os.path.join(TOKEN_DIFR_ROOT, "serve.py")
    command = [sys.executable, serve_script, "modal", "stop", "--name", server_name]
    print(f"Scaling down modal verification server {server_name} to stop billing...")
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Failed to stop modal verification server "
            f"{server_name!r} (exit {completed.returncode}). "
            f"stdout: {stdout} stderr: {stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    print(f"Modal verification server {server_name} scaled down.")


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


def _config_file_candidates(hf_model: str) -> list[Path]:
    candidates: list[Path] = []
    dedupe: set[str] = set()

    model_tail = hf_model.split("/", 1)[-1]
    for raw_name in (model_tail, hf_model):
        slug = _sanitize_name(raw_name)
        if slug in dedupe:
            continue
        dedupe.add(slug)
        candidates.append(CONFIG_DIR / f"{slug}.json")

    return candidates


def _load_modal_profile(hf_model: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "model": hf_model,
        "served_model_name": "",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 0,
        "max_num_seqs": 0,
        "enforce_eager": False,
        "trust_remote_code": True,
        "modal_gpu": "H100",
        "modal_min_containers": 0,
        "modal_max_containers": 1,
        "modal_scaledown_window_seconds": 60,
    }

    config_payload: dict[str, Any] | None = None
    config_path: Path | None = None

    for candidate in _config_file_candidates(hf_model):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        config_payload = payload
        config_path = candidate
        break

    if config_payload is None and CONFIG_DIR.is_dir():
        for candidate in sorted(CONFIG_DIR.glob("*.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            model_name = payload.get("model")
            if not isinstance(model_name, str):
                continue
            try:
                resolved = resolve_hf_name(model_name)
            except Exception:
                resolved = model_name
            if resolved != hf_model:
                continue
            config_payload = payload
            config_path = candidate
            break

    if isinstance(config_payload, dict):
        served_model_name = config_payload.get("served_model_name")
        if isinstance(served_model_name, str):
            profile["served_model_name"] = served_model_name

        dtype = config_payload.get("dtype")
        if isinstance(dtype, str) and dtype.strip():
            profile["dtype"] = dtype

        gpu = config_payload.get("modal_gpu")
        if isinstance(gpu, str) and gpu.strip():
            profile["modal_gpu"] = gpu

        trust_remote_code = config_payload.get("trust_remote_code")
        if isinstance(trust_remote_code, bool):
            profile["trust_remote_code"] = trust_remote_code
        enforce_eager = config_payload.get("enforce_eager")
        if isinstance(enforce_eager, bool):
            profile["enforce_eager"] = enforce_eager

        profile["tensor_parallel_size"] = _to_int(
            config_payload.get("tensor_parallel_size"),
            1,
            min_value=1,
        )
        profile["gpu_memory_utilization"] = _to_float(
            config_payload.get("gpu_memory_utilization"),
            0.9,
            min_value=0.1,
        )
        profile["max_model_len"] = _to_int(config_payload.get("max_model_len"), 0, min_value=0)
        profile["max_num_seqs"] = _to_int(config_payload.get("max_num_seqs"), 0, min_value=0)
        profile["modal_min_containers"] = _to_int(config_payload.get("modal_min_containers"), 0, min_value=0)
        profile["modal_max_containers"] = _to_int(config_payload.get("modal_max_containers"), 1, min_value=1)
        profile["modal_scaledown_window_seconds"] = _to_int(
            config_payload.get("modal_scaledown_window_seconds"),
            60,
            min_value=0,
        )

    # Audit runs use short prompts/generations; cap context length to avoid
    # expensive or unstable cold starts from oversized KV-cache reservations.
    cap = DEFAULT_MODAL_AUDIT_MAX_MODEL_LEN_CAP
    max_model_len = _to_int(profile.get("max_model_len"), 0, min_value=0)
    if max_model_len <= 0 or max_model_len > cap:
        profile["max_model_len"] = cap

    if config_path is not None:
        profile["config_path"] = str(config_path)
    return profile


def _build_modal_start_command(
    *,
    server_name: str,
    app_name: str,
    class_name: str,
    profile: dict[str, Any],
    deploy: bool,
) -> list[str]:
    serve_script = os.path.join(TOKEN_DIFR_ROOT, "serve.py")
    command = [
        sys.executable,
        serve_script,
        "modal",
        "start",
        "--name",
        server_name,
        "--model",
        str(profile["model"]),
        "--app-name",
        app_name,
        "--class-name",
        class_name,
        "--gpu",
        str(profile["modal_gpu"]),
        "--min-containers",
        "0",
        "--max-containers",
        str(_to_int(profile.get("modal_max_containers"), 1, min_value=1)),
        "--scaledown-window-seconds",
        str(profile["modal_scaledown_window_seconds"]),
        "--tensor-parallel-size",
        str(profile["tensor_parallel_size"]),
        "--dtype",
        str(profile["dtype"]),
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"]),
        "--max-model-len",
        str(profile["max_model_len"]),
        "--max-num-seqs",
        str(_to_int(profile.get("max_num_seqs"), 0, min_value=0)),
    ]

    served_model_name = str(profile.get("served_model_name") or "")
    if served_model_name:
        command.extend(["--served-model-name", served_model_name])

    if bool(profile.get("trust_remote_code", True)):
        command.append("--trust-remote-code")
    else:
        command.append("--no-trust-remote-code")
    if bool(profile.get("enforce_eager", False)):
        command.append("--enforce-eager")
    else:
        command.append("--no-enforce-eager")

    if not deploy:
        command.append("--no-deploy")

    return command


def _start_modal_verification_server_for_model(
    *,
    hf_model: str,
    app_name: str,
    class_name: str,
    deploy: bool,
) -> tuple[str, str, dict[str, Any]]:
    profile = _load_modal_profile(hf_model)
    profile["model"] = hf_model

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    server_name = f"audit-{_sanitize_name(hf_model)}-{timestamp}"
    command = _build_modal_start_command(
        server_name=server_name,
        app_name=app_name,
        class_name=class_name,
        profile=profile,
        deploy=deploy,
    )
    print(
        "Starting modal verification server "
        f"{server_name} (gpu={profile['modal_gpu']}, tp={profile['tensor_parallel_size']}, "
        f"min_containers=0, max_containers={_to_int(profile.get('modal_max_containers'), 1, min_value=1)})"
    )
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Failed to start modal verification server "
            f"{server_name!r} (exit {completed.returncode}). stdout: {stdout} stderr: {stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())

    modal_servers = _read_modal_state()
    entry = modal_servers.get(server_name)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Modal server {server_name!r} started but was not found in state file {STATE_FILE}."
        )

    base_url = entry.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimeError(f"Modal server {server_name!r} has no base URL recorded in state.")
    normalized_base_url = _normalize_openai_base_url(base_url, ensure_v1_path=True)
    return server_name, normalized_base_url, profile


def _stop_modal_verification_server_by_name(server_name: str) -> None:
    if not server_name.strip():
        return
    serve_script = os.path.join(TOKEN_DIFR_ROOT, "serve.py")
    command = [sys.executable, serve_script, "modal", "stop", "--name", server_name]
    print(f"Scaling down modal verification server {server_name} to zero containers...")
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Failed to stop modal verification server "
            f"{server_name!r} (exit {completed.returncode}). stdout: {stdout} stderr: {stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())


def _extract_account_id(account_ref: str) -> str:
    """Extract account ID from either '<id>' or 'accounts/<id>'."""
    ref = account_ref.strip()
    if ref.startswith("accounts/"):
        return ref.split("/", 1)[1]
    return ref


def _extract_deployment_parts(deployment_ref: str) -> tuple[str | None, str]:
    """Extract (account_id, deployment_id) from deployment ref."""
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
    """Send a Fireworks management API request."""
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
    """Resolve Fireworks account ID from env override or /v1/accounts."""
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
    """Find a validated deployment shape name for the given model.

    Uses the global wildcard versions endpoint with a model filter — the same
    approach that reliably works for Fireworks-hosted models regardless of
    which account is making the request.
    """
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
    # Shape name is the version name with the trailing /versions/<id> stripped
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
    """Create a temporary Fireworks deployment via management API."""
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

    # Primary: use a validated deployment shape name, same approach as the standalone script.
    # The global wildcard versions endpoint finds Fireworks-hosted shapes regardless of account.
    shape_name = _get_fireworks_deployment_shape_name(api_key, base_model)
    if shape_name:
        add_payload({**base_payload, "deploymentShape": shape_name})

    # Fallback matrix for cases where shape lookup fails or the shape is unavailable.
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
                        # Try this immediately next; avoid burning through the whole queue.
                        payload_candidates.insert(idx, adjusted_payload)

            # If API tells us which accelerators are allowed for a precision, enqueue those variants.
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
    """Delete a temporary Fireworks deployment via management API."""
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
    """Wait until a temporary Fireworks deployment is ready for inference."""
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
        choices=("fireworks", "modal"),
        default="fireworks",
        help="Verification backend to use for provider audits and reference checks.",
    )
    parser.add_argument(
        "--modal-verification-base-url",
        default=None,
        help=(
            "OpenAI-compatible Modal verification base URL. If omitted, audit.py auto-manages "
            "a per-model Modal verification server and scales it down to zero after each model "
            "(unless --no-modal-stop-after-verification is set)."
        ),
    )
    parser.add_argument(
        "--modal-verification-model",
        default=None,
        help=(
            "Optional model/deployment identifier sent to the Modal verification backend. "
            "Defaults to the audited HuggingFace model name."
        ),
    )
    parser.add_argument(
        "--modal-app-name",
        default=DEFAULT_MODAL_APP_NAME,
        help="Modal app name used when auto-managing verification servers.",
    )
    parser.add_argument(
        "--modal-class-name",
        default=DEFAULT_MODAL_CLASS_NAME,
        help="Modal class name used when auto-managing verification servers.",
    )
    parser.add_argument(
        "--modal-deploy-before-start",
        dest="modal_deploy_before_start",
        action="store_true",
        default=True,
        help=(
            "When auto-managing Modal verification, deploy modal_vllm_app.py before the first "
            "model starts (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-modal-deploy-before-start",
        dest="modal_deploy_before_start",
        action="store_false",
        help="Skip modal deploy and reuse an already deployed Modal app in auto-managed mode.",
    )
    parser.add_argument(
        "--modal-stop-after-verification",
        dest="modal_stop_after_verification",
        action="store_true",
        default=True,
        help=(
            "Scale down tracked Modal verification server(s) to zero to stop billing "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-modal-stop-after-verification",
        dest="modal_stop_after_verification",
        action="store_false",
        help="Leave Modal verification server(s) running after the audit.",
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
    """Run a shell command template and return stdout."""
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
    """Create a temporary Fireworks deployment and return deployment path."""
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
    """Delete a temporary Fireworks deployment."""
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


def _compute_reference_metrics(
    model_name: str,
    sequences: list[TokenSequence],
    verification_backend: str = "fireworks",
    fireworks_on_demand_deployment: str | None = None,
    modal_verification_base_url: str | None = None,
    modal_verification_model: str | None = None,
) -> dict:
    model_name = resolve_hf_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = len(tokenizer)

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

    async def _verify_modal(target_model: str, base_url: str):
        modal_api_key = os.environ.get("MODAL_VERIFICATION_API_KEY") or "modal-verification"
        modal_client = _create_async_openai_client(
            api_key=modal_api_key,
            base_url=base_url,
        )
        return await verify_outputs_openai_compatible(
            sequences,
            vocab_size=vocab_size,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            seed=SEED,
            client=modal_client,
            model=target_model,
            topk_logprobs=5,
            backend_label="modal API",
            request_extra_body={"return_tokens_as_token_ids": True},
            concurrency=DEFAULT_MODAL_VERIFICATION_CONCURRENCY,
        )

    backend = verification_backend.strip().lower()
    if backend == "modal":
        raw_base_url = modal_verification_base_url or os.environ.get("MODAL_VERIFICATION_BASE_URL")
        if not raw_base_url:
            raise ValueError(
                "Modal reference verification requires --modal-verification-base-url "
                "or MODAL_VERIFICATION_BASE_URL."
            )
        normalized_base_url = _normalize_openai_base_url(
            raw_base_url,
            ensure_v1_path=True,
        )
        target_model = modal_verification_model or model_name
        results_tokens = asyncio.run(_verify_modal(target_model, normalized_base_url))
        summary = compute_metrics_summary(results_tokens)
        summary["n_sequences"] = len(sequences)
        summary["verification_backend"] = "modal"
        summary["verification_target"] = target_model
        summary["modal_verification_base_url"] = normalized_base_url
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


def _main_modal(
    models: list[str],
    use_reference_tokens: bool,
    modal_verification_base_url: str | None,
    modal_verification_model: str | None,
    modal_stop_after_verification: bool,
    modal_app_name: str,
    modal_class_name: str,
    modal_deploy_before_start: bool,
) -> None:
    raw_base_url = modal_verification_base_url or os.environ.get("MODAL_VERIFICATION_BASE_URL")
    fixed_base_url = (
        _normalize_openai_base_url(raw_base_url, ensure_v1_path=True)
        if raw_base_url and str(raw_base_url).strip()
        else None
    )
    auto_manage = fixed_base_url is None
    if auto_manage:
        print(
            "No modal verification base URL provided. "
            "Auto-managing per-model modal verification servers."
        )
    deploy_for_each_model = bool(modal_deploy_before_start and auto_manage)

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

        verification_base_url = fixed_base_url
        modal_server_name: str | None = None
        modal_profile: dict[str, Any] | None = None

        try:
            if auto_manage:
                modal_server_name, verification_base_url, modal_profile = (
                    _start_modal_verification_server_for_model(
                        hf_model=hf_model,
                        app_name=modal_app_name,
                        class_name=modal_class_name,
                        deploy=deploy_for_each_model,
                    )
                )
                print(
                    f"Using auto-managed modal verification endpoint for {hf_model}: {verification_base_url}"
                )
            if not verification_base_url:
                raise ValueError(
                    "Modal verification base URL is not available. "
                    "Pass --modal-verification-base-url or configure auto-managed mode."
                )

            verification_target = modal_verification_model or hf_model
            results = {
                "model": hf_model,
                "parameters": {
                    "n_prompts": N_PROMPTS,
                    "max_tokens": MAX_TOKENS,
                    "seed": SEED,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "temperature": TEMPERATURE,
                    "verification_backend": "modal",
                    "modal_verification_target": verification_target,
                    "modal_verification_base_url": verification_base_url,
                    "modal_verification_strategy": (
                        "auto-managed-per-model" if auto_manage else "fixed-base-url"
                    ),
                    "modal_server_name": modal_server_name,
                },
                "providers": {},
            }
            if isinstance(modal_profile, dict):
                modal_profile_summary = {
                    "gpu": modal_profile.get("modal_gpu"),
                    "tensor_parallel_size": modal_profile.get("tensor_parallel_size"),
                    "dtype": modal_profile.get("dtype"),
                    "gpu_memory_utilization": modal_profile.get("gpu_memory_utilization"),
                    "max_model_len": modal_profile.get("max_model_len"),
                    "max_num_seqs": modal_profile.get("max_num_seqs"),
                    "enforce_eager": modal_profile.get("enforce_eager"),
                    "max_containers": modal_profile.get("modal_max_containers"),
                    "trust_remote_code": modal_profile.get("trust_remote_code"),
                    "config_path": modal_profile.get("config_path"),
                }
                results["parameters"]["modal_profile"] = modal_profile_summary

            if use_reference_tokens:
                prompts, reference_sequences = _load_reference_bundle(hf_model)
                print(f"Loaded {len(reference_sequences)} reference sequences")
                reference_metrics = _compute_reference_metrics(
                    hf_model,
                    reference_sequences,
                    verification_backend="modal",
                    modal_verification_base_url=verification_base_url,
                    modal_verification_model=verification_target,
                )
                results["reference"] = reference_metrics
            else:
                prompts = construct_prompts(
                    n_prompts=N_PROMPTS,
                    model_name=hf_model,
                    system_prompt="You are a helpful assistant.",
                )
                print(f"Constructed {len(prompts)} prompts")

            safe_model_name = hf_model.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "audit_results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{safe_model_name}_audit_results_{timestamp}.json"
            save_results(results, output_file)
            print(f"Results will be saved to {output_file}")

            for provider in providers:
                print(f"\nAuditing provider: {provider}")
                try:
                    result = audit_provider(
                        prompts,
                        model=hf_model,
                        provider=provider,
                        max_tokens=MAX_TOKENS,
                        seed=SEED,
                        top_k=TOP_K,
                        top_p=TOP_P,
                        temperature=TEMPERATURE,
                        verification_backend="modal",
                        verification_model=verification_target,
                        verification_base_url=verification_base_url,
                    )
                    provider_results = asdict(result)
                    provider_results["verification_backend"] = "modal"
                    provider_results["verification_target"] = verification_target
                    results["providers"][provider] = provider_results

                    print(f"  Total tokens: {result.total_tokens}")
                    print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                    print(f"  Avg probability: {result.avg_prob:.4f}")
                except Exception as provider_error:
                    print(f"  ERROR: {provider_error}")
                    results["providers"][provider] = {
                        "error": str(provider_error),
                        "verification_backend": "modal",
                        "verification_target": verification_target,
                    }

                save_results(results, output_file)

            print(f"\nAll results saved to {output_file}")
        finally:
            if auto_manage and modal_server_name and modal_stop_after_verification:
                try:
                    _stop_modal_verification_server_by_name(modal_server_name)
                except Exception as teardown_error:
                    print(
                        "Warning: failed to scale down auto-managed modal verification server "
                        f"{modal_server_name}: {teardown_error}"
                    )

    if fixed_base_url and modal_stop_after_verification:
        try:
            _stop_modal_verification_server(fixed_base_url)
        except Exception as teardown_error:
            print(f"Warning: Modal teardown failed: {teardown_error}")


def main(
    models: list[str],
    use_reference_tokens: bool,
    fireworks_on_demand_deployment: str | None = None,
    fireworks_create_deployment_cmd: str | None = None,
    fireworks_delete_deployment_cmd: str | None = None,
    verification_backend: str = "fireworks",
    modal_verification_base_url: str | None = None,
    modal_verification_model: str | None = None,
    modal_stop_after_verification: bool = True,
    modal_app_name: str = DEFAULT_MODAL_APP_NAME,
    modal_class_name: str = DEFAULT_MODAL_CLASS_NAME,
    modal_deploy_before_start: bool = True,
) -> None:
    backend = verification_backend.strip().lower()
    if backend == "modal":
        _main_modal(
            models=models,
            use_reference_tokens=use_reference_tokens,
            modal_verification_base_url=modal_verification_base_url,
            modal_verification_model=modal_verification_model,
            modal_stop_after_verification=modal_stop_after_verification,
            modal_app_name=modal_app_name,
            modal_class_name=modal_class_name,
            modal_deploy_before_start=modal_deploy_before_start,
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

            # Initialize results structure with metadata
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

            # Write initial file so we can watch progress
            save_results(results, output_file)
            print(f"Results will be saved to {output_file}")

            collected_provider_sequences: dict[str, tuple[list[TokenSequence], int]] = {}
            skip_remaining_providers = False
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

                # Save after each provider completes
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
        args.modal_verification_base_url,
        args.modal_verification_model,
        args.modal_stop_after_verification,
        args.modal_app_name,
        args.modal_class_name,
        args.modal_deploy_before_start,
    )
