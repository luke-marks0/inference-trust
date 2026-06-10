# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

TOKEN_DIFR_ROOT = Path(__file__).resolve().parent
SRC_DIR = TOKEN_DIFR_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv()

from token_difr.model_registry import resolve_hf_name

STATE_DIR = TOKEN_DIFR_ROOT / "state"
STATE_FILE = STATE_DIR / "servers.json"
LOG_DIR = STATE_DIR / "logs"
DEFAULT_LOCAL_HOST = "0.0.0.0"
DEFAULT_LOCAL_PORT = 8000
O200K_BASE_TIKTOKEN_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
O200K_BASE_TIKTOKEN_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
CONFIG_DIR = TOKEN_DIFR_ROOT / "configs"

VASTAI_API_BASE = "https://console.vast.ai/api/v0"
DEFAULT_VAST_GPU = os.environ.get("VAST_DEFAULT_GPU", "H100_SXM4_80GB")
DEFAULT_VAST_DISK_GB = float(os.environ.get("VAST_DEFAULT_DISK_GB", "200"))
DEFAULT_VAST_MAX_PRICE = float(os.environ.get("VAST_DEFAULT_MAX_PRICE", "10.0"))
DEFAULT_VAST_VLLM_IMAGE = os.environ.get("VAST_VLLM_IMAGE", "vllm/vllm-openai:latest")
DEFAULT_VAST_INSTANCE_TIMEOUT = int(os.environ.get("VAST_INSTANCE_TIMEOUT_SECONDS", "600"))
DEFAULT_VAST_READY_TIMEOUT = int(os.environ.get("VAST_READY_TIMEOUT_SECONDS", "3600"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _load_local_profile(hf_model: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "served_model_name": "",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 0,
        "enforce_eager": False,
        "trust_remote_code": True,
        "extra_args": "",
    }

    for candidate in _config_file_candidates(hf_model):
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("served_model_name"), str):
            profile["served_model_name"] = payload["served_model_name"]
        profile["tensor_parallel_size"] = _to_int(payload.get("tensor_parallel_size"), profile["tensor_parallel_size"], min_value=1)
        if isinstance(payload.get("dtype"), str) and payload["dtype"].strip():
            profile["dtype"] = payload["dtype"].strip()
        if isinstance(payload.get("kv_cache_dtype"), str) and payload["kv_cache_dtype"].strip():
            profile["kv_cache_dtype"] = payload["kv_cache_dtype"].strip()
        profile["gpu_memory_utilization"] = _to_float(
            payload.get("gpu_memory_utilization"),
            profile["gpu_memory_utilization"],
            min_value=0.0,
        )
        profile["max_model_len"] = _to_int(payload.get("max_model_len"), profile["max_model_len"], min_value=0)
        if isinstance(payload.get("enforce_eager"), bool):
            profile["enforce_eager"] = payload["enforce_eager"]
        if isinstance(payload.get("trust_remote_code"), bool):
            profile["trust_remote_code"] = payload["trust_remote_code"]
        if isinstance(payload.get("extra_args"), str) and payload["extra_args"].strip():
            profile["extra_args"] = payload["extra_args"].strip()
        break

    return profile


def _load_vast_profile(hf_model: str) -> dict[str, Any]:
    """Load model config with vast.ai-specific fields merged in."""
    profile = _load_local_profile(hf_model)
    profile["vast_gpu"] = DEFAULT_VAST_GPU
    profile["vast_num_gpus"] = profile["tensor_parallel_size"]
    profile["vast_disk_gb"] = DEFAULT_VAST_DISK_GB
    profile["vast_max_price"] = DEFAULT_VAST_MAX_PRICE
    profile["vast_vllm_image"] = DEFAULT_VAST_VLLM_IMAGE

    for candidate in _config_file_candidates(hf_model):
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("vast_gpu"), str) and payload["vast_gpu"].strip():
            profile["vast_gpu"] = payload["vast_gpu"].strip()
        profile["vast_num_gpus"] = _to_int(
            payload.get("vast_num_gpus"),
            profile["vast_num_gpus"],
            min_value=1,
        )
        profile["vast_disk_gb"] = _to_float(
            payload.get("vast_disk_gb"),
            profile["vast_disk_gb"],
            min_value=20.0,
        )
        profile["vast_max_price"] = _to_float(
            payload.get("vast_max_price"),
            profile["vast_max_price"],
            min_value=0.01,
        )
        if isinstance(payload.get("vast_vllm_image"), str) and payload["vast_vllm_image"].strip():
            profile["vast_vllm_image"] = payload["vast_vllm_image"].strip()
        break

    return profile


def _resolve_local_start_settings(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    profile = _load_local_profile(model_name)
    return {
        "served_model_name": args.served_model_name if args.served_model_name is not None else profile["served_model_name"],
        "tensor_parallel_size": int(args.tensor_parallel_size) if args.tensor_parallel_size is not None else int(profile["tensor_parallel_size"]),
        "dtype": str(args.dtype) if args.dtype is not None else str(profile["dtype"]),
        "kv_cache_dtype": str(args.kv_cache_dtype) if args.kv_cache_dtype is not None else str(profile["kv_cache_dtype"]),
        "gpu_memory_utilization": float(args.gpu_memory_utilization) if args.gpu_memory_utilization is not None else float(profile["gpu_memory_utilization"]),
        "max_model_len": int(args.max_model_len) if args.max_model_len is not None else int(profile["max_model_len"]),
        "enforce_eager": bool(args.enforce_eager) if args.enforce_eager is not None else bool(profile["enforce_eager"]),
        "trust_remote_code": bool(args.trust_remote_code) if args.trust_remote_code is not None else bool(profile["trust_remote_code"]),
        "extra_args": str(profile["extra_args"]),
    }


def _read_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {"local_servers": {}, "vast_servers": {}}
    payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"local_servers": {}, "vast_servers": {}}
    local_servers = payload.get("local_servers")
    vast_servers = payload.get("vast_servers")
    if not isinstance(local_servers, dict):
        local_servers = {}
    if not isinstance(vast_servers, dict):
        vast_servers = {}
    return {"local_servers": local_servers, "vast_servers": vast_servers}


def _write_state(payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_openai_server(base_url: str, timeout_seconds: int, poll_seconds: float = 1.0) -> bool:
    deadline = time.time() + timeout_seconds
    # Strip /v1 suffix — vLLM health is at /health, not /v1/health
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    root = urllib.parse.urlunparse(parsed._replace(path=re.sub(r"/v1$", "", parsed.path)))
    endpoint = root.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(poll_seconds)
    return False


def _append_query(url: str, params: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    encoded = urllib.parse.urlencode(query)
    return urllib.parse.urlunparse(parsed._replace(query=encoded))


def _normalize_base_url(raw_url: str) -> str:
    parsed = urllib.parse.urlparse(raw_url.strip())
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urllib.parse.urlunparse(parsed._replace(path=path))


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    existing = env.get(key, "").strip()
    if not existing:
        env[key] = value
        return
    parts = [part for part in existing.split(":") if part]
    if value not in parts:
        env[key] = ":".join([value, *parts])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_o200k_base_vocab() -> Path:
    encodings_dir = STATE_DIR / "encodings"
    target_path = encodings_dir / "o200k_base.tiktoken"
    if target_path.is_file():
        if _sha256_file(target_path) == O200K_BASE_TIKTOKEN_SHA256:
            return encodings_dir
        target_path.unlink()

    encodings_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        O200K_BASE_TIKTOKEN_URL,
        headers={"User-Agent": "token-difr-serve/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != O200K_BASE_TIKTOKEN_SHA256:
        raise RuntimeError(
            "Downloaded o200k_base.tiktoken digest mismatch: "
            f"expected {O200K_BASE_TIKTOKEN_SHA256}, got {digest}."
        )

    target_path.write_bytes(payload)
    return encodings_dir


def _build_local_runtime_env(model_name: str) -> dict[str, str]:
    env = os.environ.copy()

    try:
        import vllm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Local vLLM startup requires `vllm` in the active environment. "
            "Reinstall `token-difr` so the default dependencies are present, for example:\n"
            "uv pip install -e ."
        ) from exc

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Local vLLM startup requires PyTorch to be installed in the active environment."
        ) from exc

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            "Local vLLM startup requires a CUDA-enabled PyTorch build. "
            "Reinstall `token-difr` with uv so it resolves the CUDA PyTorch wheels, for example:\n"
            "uv pip install -e ."
        )

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    libtorch_cuda = torch_lib_dir / "libtorch_cuda.so"
    if not libtorch_cuda.is_file():
        raise RuntimeError(
            f"Expected CUDA runtime library not found: {libtorch_cuda}. "
            "Reinstall a CUDA-enabled PyTorch build before starting local vLLM."
        )

    _prepend_env_path(env, "LD_LIBRARY_PATH", str(torch_lib_dir))

    if resolve_hf_name(model_name).startswith("openai/gpt-oss") and not env.get("TIKTOKEN_ENCODINGS_BASE"):
        encodings_dir = _ensure_o200k_base_vocab()
        env["TIKTOKEN_ENCODINGS_BASE"] = f"{encodings_dir}/"

    return env


def _build_local_command(args: argparse.Namespace, settings: dict[str, Any] | None = None) -> list[str]:
    model_name = resolve_hf_name(args.model)
    if settings is None:
        settings = {
            "served_model_name": getattr(args, "served_model_name", ""),
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
            "dtype": getattr(args, "dtype", "auto"),
            "kv_cache_dtype": getattr(args, "kv_cache_dtype", "auto"),
            "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", 0.9),
            "max_model_len": getattr(args, "max_model_len", 0),
            "enforce_eager": getattr(args, "enforce_eager", False),
            "trust_remote_code": getattr(args, "trust_remote_code", True),
            "extra_args": getattr(args, "extra_args", ""),
        }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(settings["tensor_parallel_size"]),
        "--dtype",
        str(settings["dtype"]),
        "--kv-cache-dtype",
        str(settings["kv_cache_dtype"]),
        "--gpu-memory-utilization",
        str(settings["gpu_memory_utilization"]),
    ]
    if int(settings["max_model_len"]) > 0:
        command.extend(["--max-model-len", str(settings["max_model_len"])])
    if str(settings["served_model_name"]).strip():
        command.extend(["--served-model-name", str(settings["served_model_name"])])
    if bool(settings["enforce_eager"]):
        command.append("--enforce-eager")
    if bool(settings["trust_remote_code"]):
        command.append("--trust-remote-code")
    combined_extra = " ".join(filter(None, [str(settings.get("extra_args", "")), args.extra_args]))
    if combined_extra.strip():
        command.extend(shlex.split(combined_extra))
    return command


def _local_start(args: argparse.Namespace) -> None:
    state = _read_state()
    local_servers = dict(state["local_servers"])
    model_name = resolve_hf_name(args.model)
    name = args.name or f"{_sanitize_name(model_name)}-{args.port}"

    existing = local_servers.get(name)
    if isinstance(existing, dict):
        pid = existing.get("pid")
        if isinstance(pid, int) and _pid_is_running(pid):
            raise RuntimeError(f"Local server {name!r} is already running (pid={pid}).")
        local_servers.pop(name, None)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"local-{name}.log"
    settings = _resolve_local_start_settings(args, model_name)
    command = _build_local_command(args, settings)
    runtime_env = _build_local_runtime_env(model_name)

    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=runtime_env,
            start_new_session=True,
        )

    probe_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    base_url = f"http://{probe_host}:{args.port}"
    if not _wait_for_openai_server(base_url, timeout_seconds=args.start_timeout):
        if _pid_is_running(process.pid):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise RuntimeError(
            f"Local vLLM server did not become ready on {base_url} within {args.start_timeout}s. "
            f"See log: {log_path}"
        )

    local_servers[name] = {
        "name": name,
        "model": model_name,
        "host": args.host,
        "port": int(args.port),
        "pid": int(process.pid),
        "base_url": _normalize_base_url(base_url),
        "log_file": str(log_path),
        "started_at_utc": _utc_now_iso(),
        "command": command,
        **settings,
    }
    _write_state({"local_servers": local_servers, "vast_servers": state["vast_servers"]})
    print(f"Started local vLLM server {name}: pid={process.pid} base_url={_normalize_base_url(base_url)}")


def _stop_process(pid: int, timeout_seconds: int = 20) -> bool:
    if not _pid_is_running(pid):
        return True

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.5)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
    return not _pid_is_running(pid)


def _local_stop(args: argparse.Namespace) -> None:
    state = _read_state()
    local_servers = dict(state["local_servers"])

    if args.all:
        target_names = list(local_servers.keys())
    elif args.name:
        target_names = [args.name]
    else:
        raise ValueError("Specify --name or --all for local stop.")

    for name in target_names:
        entry = local_servers.get(name)
        if not isinstance(entry, dict):
            print(f"Local server {name!r} not found in state.")
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int):
            print(f"Local server {name!r} has invalid pid in state; removing entry.")
            local_servers.pop(name, None)
            continue
        stopped = _stop_process(pid)
        if stopped:
            print(f"Stopped local server {name} (pid={pid}).")
        else:
            print(f"Failed to stop local server {name} (pid={pid}).")
        local_servers.pop(name, None)

    _write_state({"local_servers": local_servers, "vast_servers": state["vast_servers"]})


def _local_list() -> None:
    state = _read_state()
    local_servers = state["local_servers"]
    if not local_servers:
        print("No local servers recorded.")
        return
    for name in sorted(local_servers.keys()):
        entry = local_servers[name]
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        status = "running" if isinstance(pid, int) and _pid_is_running(pid) else "stopped"
        print(
            f"{name}: status={status} model={entry.get('model')} "
            f"pid={pid} base_url={entry.get('base_url')}"
        )


# ---------------------------------------------------------------------------
# vast.ai server management
# ---------------------------------------------------------------------------


def _vastai_request(
    method: str,
    path: str,
    api_key: str,
    *,
    payload: dict | None = None,
    params: dict[str, str] | None = None,
) -> dict:
    url = f"{VASTAI_API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
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
        raise RuntimeError(
            f"vast.ai API {method} {path} failed (HTTP {exc.code}): {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vast.ai API {method} {path} failed: {exc}") from exc
    if not body:
        return {}
    parsed = json.loads(body)
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


def _search_vast_offers(
    api_key: str,
    *,
    gpu_name: str,
    num_gpus: int,
    disk_gb: float,
    max_price: float,
) -> list[dict]:
    """Return available offers sorted cheapest-first."""
    query = {
        "gpu_name": {"in": [gpu_name]},
        "num_gpus": {"gte": num_gpus, "lte": num_gpus},
        "disk_space": {"gte": disk_gb},
        "dph_total": {"lte": max_price},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "direct_port_count": {"gte": 1},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
    }
    result = _vastai_request("GET", "/bundles/", api_key, params={"q": json.dumps(query)})
    offers = result.get("offers", [])
    return offers if isinstance(offers, list) else []


def _create_vast_instance(
    api_key: str,
    offer_id: int,
    *,
    image: str,
    args: str,
    disk_gb: float,
    label: str,
    env_vars: dict[str, str],
) -> int:
    """Rent a vast.ai instance and return its contract/instance ID."""
    # vast.ai expects env as Docker CLI flags: {"-e KEY": "value", "-p PORT:PORT": "1"}
    env_payload: dict[str, str] = {f"-e {k}": v for k, v in env_vars.items()}
    env_payload["-p 8000:8000"] = "1"
    body = {
        "client_id": "me",
        "image": image,
        "disk": float(disk_gb),
        "label": label,
        "onstart": args,
        "runtype": "ssh_direct",
        "env": env_payload,
    }
    response = _vastai_request("PUT", f"/asks/{offer_id}/", api_key, payload=body)
    if not response.get("success"):
        raise RuntimeError(f"vast.ai instance creation failed: {response}")
    instance_id = response.get("new_contract")
    if not isinstance(instance_id, int):
        raise RuntimeError(f"vast.ai did not return a contract ID: {response}")
    return instance_id


def _get_vast_instance(api_key: str, instance_id: int) -> dict | None:
    """Return the instance dict, or None if not found."""
    result = _vastai_request("GET", "/instances/", api_key, params={"owner": "me"})
    instances = result.get("instances", [])
    if not isinstance(instances, list):
        return None
    for inst in instances:
        if isinstance(inst, dict) and inst.get("id") == instance_id:
            return inst
    return None


def _destroy_vast_instance(api_key: str, instance_id: int) -> None:
    _vastai_request("DELETE", f"/instances/{instance_id}/", api_key)


def _extract_vast_port(instance: dict, container_port: int = 8000) -> int | None:
    """Return the mapped public port for container_port, or None if not yet assigned."""
    # Docker-style ports dict (populated by some runtypes)
    ports = instance.get("ports")
    if isinstance(ports, dict):
        for key in (f"{container_port}/tcp", str(container_port)):
            mappings = ports.get(key)
            if isinstance(mappings, list) and mappings:
                host_port = mappings[0].get("HostPort")
                if host_port:
                    return int(host_port)

    # runtype=args with -p 8000:8000: vast.ai maps to direct_port_start
    for field in ("direct_port_start", "direct_port_end"):
        val = instance.get(field)
        if val is not None and int(val) > 0:
            return int(val)

    return None


def _build_vllm_onstart(
    *,
    model: str,
    tensor_parallel_size: int,
    dtype: str,
    kv_cache_dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    served_model_name: str,
    enforce_eager: bool,
    trust_remote_code: bool,
    extra_args: str,
    port: int = 8000,
) -> str:
    """Build the bash onstart command that launches vLLM in the background."""
    parts: list[str] = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
        "--kv-cache-dtype", kv_cache_dtype,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if max_model_len > 0:
        parts.extend(["--max-model-len", str(max_model_len)])
    if served_model_name:
        parts.extend(["--served-model-name", served_model_name])
    if enforce_eager:
        parts.append("--enforce-eager")
    if trust_remote_code:
        parts.append("--trust-remote-code")
    if extra_args.strip():
        parts.extend(shlex.split(extra_args))

    cmd = " ".join(shlex.quote(str(p)) for p in parts)
    return f"nohup {cmd} > /var/log/vllm.log 2>&1 &"


def _vast_start(args: argparse.Namespace) -> None:
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        raise ValueError("VASTAI_API_KEY environment variable not set.")

    hf_model = resolve_hf_name(args.model)
    profile = _load_vast_profile(hf_model)
    name = args.name or _sanitize_name(hf_model)

    gpu = args.gpu or profile["vast_gpu"]
    num_gpus = args.num_gpus if args.num_gpus is not None else int(profile["vast_num_gpus"])
    disk_gb = args.disk_gb if args.disk_gb is not None else float(profile["vast_disk_gb"])
    max_price = args.max_price if args.max_price is not None else float(profile["vast_max_price"])
    vllm_image = args.vllm_image or profile["vast_vllm_image"]
    instance_timeout = args.instance_timeout
    ready_timeout = args.ready_timeout

    state = _read_state()
    vast_servers = dict(state["vast_servers"])
    if name in vast_servers:
        raise RuntimeError(f"Vast server {name!r} already tracked. Run `vast stop --name {name}` first.")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN not set — gated models will fail to download.")

    vllm_args = _build_vllm_onstart(
        model=hf_model,
        tensor_parallel_size=int(profile["tensor_parallel_size"]),
        dtype=str(profile["dtype"]),
        kv_cache_dtype=str(profile["kv_cache_dtype"]),
        gpu_memory_utilization=float(profile["gpu_memory_utilization"]),
        max_model_len=int(profile["max_model_len"]),
        served_model_name=str(profile["served_model_name"]),
        enforce_eager=bool(profile["enforce_eager"]),
        trust_remote_code=bool(profile["trust_remote_code"]),
        extra_args=str(profile["extra_args"]),
    )

    print(f"Searching vast.ai: gpu={gpu} x{num_gpus}, disk>={disk_gb:.0f}GB, max ${max_price:.2f}/hr...")
    offers = _search_vast_offers(api_key, gpu_name=gpu, num_gpus=num_gpus, disk_gb=disk_gb, max_price=max_price)
    if not offers:
        raise RuntimeError(
            f"No vast.ai offers found for gpu={gpu} x{num_gpus}, "
            f"disk>={disk_gb:.0f}GB, max ${max_price:.2f}/hr. "
            "Check https://vast.ai/console for availability, or relax constraints with --vast-gpu / --max-price."
        )

    best = offers[0]
    offer_id = int(best["id"])
    actual_price = float(best.get("dph_total", 0))
    print(f"Best offer #{offer_id}: {gpu} x{num_gpus} @ ${actual_price:.3f}/hr")

    env_vars: dict[str, str] = {"HF_HOME": "/root/.cache/huggingface"}
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
        env_vars["HUGGING_FACE_HUB_TOKEN"] = hf_token

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    label = f"audit-{_sanitize_name(hf_model)}-{timestamp}"[:63]

    print(f"Creating vast.ai instance (image={vllm_image})...")
    instance_id = _create_vast_instance(
        api_key, offer_id,
        image=vllm_image,
        args=vllm_args,
        disk_gb=disk_gb,
        label=label,
        env_vars=env_vars,
    )
    print(f"Created instance {instance_id}. Waiting for it to reach running state...")

    # Write state immediately so we can clean up if something fails later.
    vast_servers[name] = {
        "name": name,
        "model": hf_model,
        "instance_id": instance_id,
        "base_url": None,
        "started_at_utc": _utc_now_iso(),
        "vast_gpu": gpu,
        "num_gpus": num_gpus,
        "disk_gb": disk_gb,
        "dph_total": actual_price,
        "label": label,
        "vllm_image": vllm_image,
        "tensor_parallel_size": int(profile["tensor_parallel_size"]),
        "dtype": str(profile["dtype"]),
        "kv_cache_dtype": str(profile["kv_cache_dtype"]),
        "gpu_memory_utilization": float(profile["gpu_memory_utilization"]),
        "max_model_len": int(profile["max_model_len"]),
        "enforce_eager": bool(profile["enforce_eager"]),
        "trust_remote_code": bool(profile["trust_remote_code"]),
    }
    _write_state({"local_servers": state["local_servers"], "vast_servers": vast_servers})

    # Poll until running and port is mapped.
    start_time = time.time()
    base_url: str | None = None
    last_status = ""
    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > instance_timeout:
            raise RuntimeError(
                f"Vast instance {instance_id} did not reach running state within {instance_timeout}s. "
                "Check the vast.ai console for errors."
            )

        inst = _get_vast_instance(api_key, instance_id)
        if inst is None:
            raise RuntimeError(f"Instance {instance_id} disappeared from vast.ai after creation.")

        actual_status = str(inst.get("actual_status") or "").lower()
        status_msg = str(inst.get("status_msg") or "")

        if actual_status in ("exited", "stopped", "failed", "deleted"):
            raise RuntimeError(
                f"Vast instance {instance_id} entered terminal state {actual_status!r}: {status_msg}"
            )

        if actual_status == "running":
            public_ip = inst.get("public_ipaddr") or inst.get("ssh_host")
            mapped_port = _extract_vast_port(inst, 8000)
            if public_ip and mapped_port:
                base_url = _normalize_base_url(f"http://{public_ip}:{mapped_port}")
                vast_servers[name]["base_url"] = base_url
                vast_servers[name]["public_ipaddr"] = public_ip
                vast_servers[name]["vllm_port"] = mapped_port
                _write_state({"local_servers": state["local_servers"], "vast_servers": vast_servers})
                print(f"Instance running: {public_ip}:{mapped_port}")
                break
            else:
                ports_raw = inst.get("ports")
                if actual_status != last_status or elapsed % 15 == 0:
                    print(
                        f"  [{elapsed}s] running but port not yet mapped "
                        f"(public_ip={public_ip!r}, ports={ports_raw!r})"
                    )

        if actual_status != last_status or elapsed % 15 == 0:
            print(f"  [{elapsed}s] status={actual_status or 'loading'} msg={status_msg or 'none'}")
            last_status = actual_status

        time.sleep(10)

    # Poll vLLM health endpoint — model download + load happens here.
    # vLLM's health endpoint is at /health (root), not /v1/health.
    root_url = f"http://{public_ip}:{mapped_port}"
    print(f"Waiting for vLLM to become ready at {root_url} (timeout={ready_timeout}s)...")
    print("  Note: large model downloads can take 10-30 minutes on first start.")
    health_url = root_url + "/health"
    health_start = time.time()
    last_health_print = health_start
    ready = False
    while True:
        elapsed_health = time.time() - health_start
        if elapsed_health > ready_timeout:
            break

        # Check instance is still alive every 30s
        if int(elapsed_health) % 30 == 0 and int(elapsed_health) > 0:
            inst = _get_vast_instance(api_key, instance_id)
            if inst is not None:
                actual_status = str(inst.get("actual_status") or "").lower()
                if actual_status in ("exited", "stopped", "failed", "deleted"):
                    status_msg = str(inst.get("status_msg") or "")
                    raise RuntimeError(
                        f"Vast instance {instance_id} exited while waiting for vLLM "
                        f"(status={actual_status!r}, msg={status_msg!r}). "
                        f"Check logs: vastai logs {instance_id}"
                    )

        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass

        now = time.time()
        if now - last_health_print >= 30:
            print(f"  [{int(elapsed_health)}s] vLLM not ready yet — model still downloading/loading...")
            last_health_print = now

        time.sleep(5)

    if not ready:
        raise RuntimeError(
            f"vLLM on instance {instance_id} did not become healthy within {ready_timeout}s. "
            "The model may still be downloading or failed to start. "
            f"Check logs: vastai logs {instance_id}"
        )

    print(f"Started vast server {name}: instance_id={instance_id} base_url={base_url}")


def _vast_stop(args: argparse.Namespace) -> None:
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        raise ValueError("VASTAI_API_KEY environment variable not set.")

    state = _read_state()
    vast_servers = dict(state["vast_servers"])

    if args.all:
        target_names = list(vast_servers.keys())
    elif args.name:
        target_names = [args.name]
    else:
        raise ValueError("Specify --name or --all for vast stop.")

    for name in target_names:
        entry = vast_servers.get(name)
        if not isinstance(entry, dict):
            print(f"Vast server {name!r} not found in state.")
            continue
        instance_id = entry.get("instance_id")
        if not isinstance(instance_id, int):
            print(f"Vast server {name!r} has no instance_id; removing stale state entry.")
            vast_servers.pop(name, None)
            continue
        try:
            _destroy_vast_instance(api_key, instance_id)
            print(f"Destroyed vast instance {instance_id} ({name}).")
        except Exception as exc:
            print(f"Warning: failed to destroy instance {instance_id} ({name}): {exc}. Removing state entry.")
        finally:
            vast_servers.pop(name, None)

    _write_state({"local_servers": state["local_servers"], "vast_servers": vast_servers})


def _vast_list() -> None:
    state = _read_state()
    vast_servers = state["vast_servers"]
    if not vast_servers:
        print("No vast servers recorded.")
        return
    for name in sorted(vast_servers.keys()):
        entry = vast_servers[name]
        if not isinstance(entry, dict):
            continue
        print(
            f"{name}: instance_id={entry.get('instance_id')} model={entry.get('model')} "
            f"gpu={entry.get('vast_gpu')} x{entry.get('num_gpus')} "
            f"${entry.get('dph_total', 0):.3f}/hr base_url={entry.get('base_url')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight local/vast.ai vLLM server management for token-difr."
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)

    # ------------------------------------------------------------------
    # local subcommand
    # ------------------------------------------------------------------
    local_parser = subparsers.add_parser("local", help="Manage local vLLM servers.")
    local_subparsers = local_parser.add_subparsers(dest="action", required=True)

    local_start = local_subparsers.add_parser("start", help="Start a local vLLM server.")
    local_start.add_argument("--name", default="", help="Server name used in local state.")
    local_start.add_argument("--model", required=True, help="HuggingFace model to serve.")
    local_start.add_argument("--host", default=DEFAULT_LOCAL_HOST, help="Host to bind.")
    local_start.add_argument("--port", type=int, default=DEFAULT_LOCAL_PORT, help="Port to bind.")
    local_start.add_argument("--served-model-name", default=None, help="Optional served model alias.")
    local_start.add_argument("--tensor-parallel-size", type=int, default=None)
    local_start.add_argument("--dtype", default=None)
    local_start.add_argument("--kv-cache-dtype", default=None)
    local_start.add_argument("--gpu-memory-utilization", type=float, default=None)
    local_start.add_argument("--max-model-len", type=int, default=None)
    local_start.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    local_start.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    local_start.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    local_start.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    local_start.set_defaults(enforce_eager=None, trust_remote_code=None)
    local_start.add_argument("--extra-args", default="", help="Extra args appended to the vLLM command.")
    local_start.add_argument("--start-timeout", type=int, default=180)

    local_stop = local_subparsers.add_parser("stop", help="Stop local vLLM server(s).")
    local_stop.add_argument("--name", default="", help="Server name to stop.")
    local_stop.add_argument("--all", action="store_true", help="Stop all tracked local servers.")

    local_subparsers.add_parser("list", help="List tracked local servers.")

    # ------------------------------------------------------------------
    # vast subcommand
    # ------------------------------------------------------------------
    vast_parser = subparsers.add_parser("vast", help="Manage remote vast.ai vLLM servers.")
    vast_subparsers = vast_parser.add_subparsers(dest="action", required=True)

    vast_start = vast_subparsers.add_parser("start", help="Rent a vast.ai instance and start vLLM.")
    vast_start.add_argument("--name", default="", help="Server name used in local state.")
    vast_start.add_argument("--model", required=True, help="HuggingFace model to serve.")
    vast_start.add_argument(
        "--gpu", default=None,
        help=f"GPU type on vast.ai (e.g. H100_SXM4_80GB, RTX_4090). Overrides config. Default: {DEFAULT_VAST_GPU}",
    )
    vast_start.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs. Defaults to tensor_parallel_size from model config.",
    )
    vast_start.add_argument(
        "--disk-gb", type=float, default=None,
        help=f"Minimum disk space in GB. Default: {DEFAULT_VAST_DISK_GB:.0f}",
    )
    vast_start.add_argument(
        "--max-price", type=float, default=None,
        help=f"Max price per hour in USD. Default: {DEFAULT_VAST_MAX_PRICE:.2f}",
    )
    vast_start.add_argument(
        "--vllm-image", default=None,
        help=f"Docker image for vLLM. Default: {DEFAULT_VAST_VLLM_IMAGE}",
    )
    vast_start.add_argument(
        "--instance-timeout", type=int, default=DEFAULT_VAST_INSTANCE_TIMEOUT,
        help="Seconds to wait for the instance to reach running state.",
    )
    vast_start.add_argument(
        "--ready-timeout", type=int, default=DEFAULT_VAST_READY_TIMEOUT,
        help="Seconds to wait for vLLM /health to respond (includes model download).",
    )

    vast_stop = vast_subparsers.add_parser("stop", help="Destroy vast.ai vLLM server(s).")
    vast_stop.add_argument("--name", default="", help="Server name to destroy.")
    vast_stop.add_argument("--all", action="store_true", help="Destroy all tracked vast servers.")

    vast_subparsers.add_parser("list", help="List tracked vast servers.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "local":
        if args.action == "start":
            _local_start(args)
            return
        if args.action == "stop":
            _local_stop(args)
            return
        if args.action == "list":
            _local_list()
            return

    if args.backend == "vast":
        if args.action == "start":
            _vast_start(args)
            return
        if args.action == "stop":
            _vast_stop(args)
            return
        if args.action == "list":
            _vast_list()
            return

    raise ValueError(f"Unsupported command: backend={args.backend} action={getattr(args, 'action', None)}")


if __name__ == "__main__":
    main()
