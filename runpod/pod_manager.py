#!/usr/bin/env python3
"""RunPod pod lifecycle manager for Memory Maze training.

CLI tool wrapping the runpod Python SDK. Handles pod creation with
cost-efficient defaults, GPU selection, spot instances, and tracking.

Usage:
    python runpod/pod_manager.py gpus
    python runpod/pod_manager.py recommend --workload smoke_test
    python runpod/pod_manager.py create --workload smoke_test --agent impala --backend mujoco
    python runpod/pod_manager.py list
    python runpod/pod_manager.py status [pod_id]
    python runpod/pod_manager.py stop <pod_id>
    python runpod/pod_manager.py resume <pod_id>
    python runpod/pod_manager.py terminate <pod_id>
    python runpod/pod_manager.py cost
    python runpod/pod_manager.py cleanup

Requires: pip install runpod
Requires: RUNPOD_API_KEY environment variable
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import runpod
except ImportError:
    print("ERROR: runpod package not installed. Run: pip install runpod", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path(__file__).parent / "pod_registry.json"
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "serapikalov/memorymaze-train:latest")
VOLUME_MOUNT = "/workspace"

# GPU preference order: Workstation/professional GPUs first (better driver stack),
# consumer GPUs (RTX 3090/4090) deprioritized — some hosts have incomplete NVIDIA
# Vulkan/rendering library mounts (libGLX_nvidia.so.0, libEGL_nvidia.so.0 missing).
GPU_PREFERENCE = [
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
    "NVIDIA RTX A5000",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA RTX A4500",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX A4000",
    "NVIDIA A30",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",  # last resort: some hosts missing Vulkan libs
]

WORKLOAD_PROFILES = {
    "smoke_test": {
        "vcpu": 8,
        "ram_gb": 16,
        "volume_gb": 10,
        "container_disk_gb": 20,
        "spot": True,
        "min_vram_gb": 0,  # any GPU
        "auto_approve_limit": 5.0,
    },
    "short_train": {
        "vcpu": 16,
        "ram_gb": 32,
        "volume_gb": 20,
        "container_disk_gb": 20,
        "spot": True,
        "min_vram_gb": 0,
        "auto_approve_limit": 50.0,
    },
    "long_train": {
        "vcpu": 32,
        "ram_gb": 62,
        "volume_gb": 50,
        "container_disk_gb": 20,
        "spot": True,
        "min_vram_gb": 0,
        "auto_approve_limit": 50.0,
    },
    "dev": {
        "vcpu": 4,
        "ram_gb": 8,
        "volume_gb": 20,
        "container_disk_gb": 30,
        "spot": False,
        "min_vram_gb": 20,  # batched mode needs ≥20 GB
        "auto_approve_limit": 0.0,  # always ask
    },
}

PROJECT_BUDGET = 300.0
BUDGET_WARN_THRESHOLDS = [0.50, 0.80, 1.00]

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def load_registry() -> dict:
    """Load pod registry from disk."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"version": 1, "budget": PROJECT_BUDGET, "network_volume_id": None, "pods": {}}


def save_registry(reg: dict):
    """Save pod registry to disk."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def init_api():
    """Initialize RunPod API and verify key is set."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    runpod.api_key = api_key


def get_gpus() -> list[dict]:
    """Fetch available GPU types with pricing via GraphQL.

    The simple runpod.get_gpus() only returns id/displayName/memoryInGb.
    We need communityCloud, communityPrice, communitySpotPrice for GPU
    selection and cost estimation.
    """
    query = """
    query GpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
            communityCloud
            communityPrice
            communitySpotPrice
            secureCloud
            securePrice
            secureSpotPrice
        }
    }
    """
    try:
        from runpod.api.graphql import run_graphql_query
        result = run_graphql_query(query)
        if result and "data" in result and result["data"].get("gpuTypes"):
            return result["data"]["gpuTypes"]
    except Exception:
        pass
    # Fallback to basic SDK (no pricing data)
    return runpod.get_gpus()


def find_best_gpu(gpus: list[dict], min_vram_gb: int = 0, spot: bool = True) -> dict | None:
    """Select cheapest available GPU matching requirements.

    Strategy:
    1. Filter by VRAM requirement
    2. Pick cheapest from preference list if available
    3. Fallback to absolute cheapest available
    """
    return rank_gpus(gpus, min_vram_gb, spot)[0] if rank_gpus(gpus, min_vram_gb, spot) else None


def rank_gpus(gpus: list[dict], min_vram_gb: int = 0, spot: bool = True) -> list[dict]:
    """Return GPUs matching requirements, sorted by preference then price.

    Returns a list so callers can try successive GPUs if the first is
    unavailable for provisioning (e.g., on-demand capacity exhausted).
    """
    price_key = "communitySpotPrice" if spot else "communityPrice"

    # Filter GPUs that are available and meet VRAM requirement
    available = []
    for gpu in gpus:
        vram = gpu.get("memoryInGb", 0)
        price = gpu.get(price_key)
        stock = gpu.get("communityCloud")
        if vram >= min_vram_gb and price and price > 0 and stock:
            available.append(gpu)

    if not available:
        # Try on-demand if spot has nothing
        if spot:
            return rank_gpus(gpus, min_vram_gb, spot=False)
        return []

    # Build ordered list: preference list first, then cheapest remainders
    result = []
    seen = set()
    for pref_name in GPU_PREFERENCE:
        for gpu in available:
            gid = gpu.get("id") or gpu.get("displayName")
            if (gpu.get("id") == pref_name or gpu.get("displayName") == pref_name) and gid not in seen:
                result.append(gpu)
                seen.add(gid)

    # Append remaining GPUs sorted by price
    remaining = sorted(
        [g for g in available if (g.get("id") or g.get("displayName")) not in seen],
        key=lambda g: g.get(price_key, float("inf")),
    )
    result.extend(remaining)
    return result


def format_price(price) -> str:
    """Format price as dollar string."""
    if price is None or price == 0:
        return "N/A"
    return f"${price:.3f}/hr"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_gpus(args):
    """List available GPUs with spot/on-demand pricing."""
    gpus = get_gpus()

    # Sort by spot price
    gpus.sort(key=lambda g: g.get("communitySpotPrice") or float("inf"))

    print(f"{'GPU Type':<40} {'VRAM':>6} {'Spot':>10} {'On-Demand':>10} {'Stock'}")
    print("-" * 85)
    for gpu in gpus:
        name = gpu.get("id", gpu.get("displayName", "Unknown"))
        vram = gpu.get("memoryInGb", "?")
        spot_price = format_price(gpu.get("communitySpotPrice"))
        demand_price = format_price(gpu.get("communityPrice"))
        stock = "Yes" if gpu.get("communityCloud") else "No"
        print(f"{name:<40} {vram:>4} GB {spot_price:>10} {demand_price:>10} {stock:>5}")


def cmd_recommend(args):
    """Show recommended config for a workload without creating anything."""
    workload = args.workload
    if workload not in WORKLOAD_PROFILES:
        print(f"ERROR: Unknown workload '{workload}'. Options: {list(WORKLOAD_PROFILES.keys())}", file=sys.stderr)
        sys.exit(1)

    profile = WORKLOAD_PROFILES[workload]
    gpus = get_gpus()
    gpu = find_best_gpu(gpus, min_vram_gb=profile["min_vram_gb"], spot=profile["spot"])

    print(f"Recommended config for workload: {workload}")
    print(f"  vCPU:           {profile['vcpu']}")
    print(f"  RAM:            {profile['ram_gb']} GB")
    print(f"  Volume:         {profile['volume_gb']} GB")
    print(f"  Container disk: {profile['container_disk_gb']} GB")
    print(f"  Spot instance:  {profile['spot']}")
    print(f"  Min VRAM:       {profile['min_vram_gb']} GB")
    if gpu:
        price_key = "communitySpotPrice" if profile["spot"] else "communityPrice"
        gpu_name = gpu.get("id", gpu.get("displayName", "Unknown"))
        price = gpu.get(price_key, 0)
        print(f"  Best GPU:       {gpu_name} ({gpu.get('memoryInGb', '?')} GB VRAM)")
        print(f"  Price:          ${price:.3f}/hr")
        print(f"  Auto-approve:   {'Yes' if price * 24 < profile['auto_approve_limit'] else 'No (exceeds threshold)'}")
    else:
        print("  Best GPU:       No suitable GPU available!")


def cmd_create(args):
    """Create a pod with optimal GPU selection."""
    workload = args.workload
    if workload not in WORKLOAD_PROFILES:
        print(f"ERROR: Unknown workload '{workload}'. Options: {list(WORKLOAD_PROFILES.keys())}", file=sys.stderr)
        sys.exit(1)

    profile = WORKLOAD_PROFILES[workload]
    agent = args.agent or "impala"
    backend = args.backend or "mujoco"

    # Check budget before creating
    reg = load_registry()
    total_spent = sum(
        sum(e.get("cost", 0) for e in pod.get("cost_log", []))
        for pod in reg["pods"].values()
    )
    budget = reg.get("budget", PROJECT_BUDGET)
    _check_budget_warnings(total_spent, budget)

    # Find best GPU
    # --on-demand overrides the workload profile's spot setting
    # --spot overrides workloads that default to on-demand (e.g. dev)
    if getattr(args, "on_demand", False):
        is_spot = False
        print("  Note: --on-demand flag set; overriding spot preference for SSH stability.")
    elif getattr(args, "spot", False):
        is_spot = True
        print("  Note: --spot flag set; using spot instance despite workload default.")
    else:
        is_spot = profile["spot"]
    gpus = get_gpus()
    min_vram = profile["min_vram_gb"]
    if backend == "genesis":
        min_vram = max(min_vram, 16)  # Genesis batched needs ≥16 GB VRAM
    gpu_candidates = rank_gpus(gpus, min_vram_gb=min_vram, spot=is_spot)

    if not gpu_candidates:
        print("ERROR: No suitable GPU available matching requirements.", file=sys.stderr)
        sys.exit(1)

    # Build environment variables
    env_vars = {
        "AGENT": agent,
        "BACKEND": backend,
        "MAZE_SIZE": args.maze_size or "9x9",
        "TOTAL_STEPS": str(args.total_steps) if args.total_steps else "100000000",

        "JUPYTER_PASSWORD": os.environ.get("JUPYTER_PASSWORD", "memorymaze"),
    }
    # Pass through optional env vars
    for var in ["WANDB_API_KEY", "XPID", "EXTRA_FLAGS"]:
        val = os.environ.get(var) or getattr(args, var.lower(), None)
        if val:
            env_vars[var] = val

    print(f"Creating {workload} pod:")
    print(f"  Spot:     {is_spot}")
    print(f"  Agent:    {agent}")
    print(f"  Backend:  {backend}")

    # Load SSH public key (prefer id_rsa, fall back to id_ed25519)
    ssh_pub_key = None
    for key_path in [Path.home() / ".ssh" / "id_rsa.pub", Path.home() / ".ssh" / "id_ed25519.pub"]:
        if key_path.exists():
            ssh_pub_key = key_path.read_text().strip()
            print(f"  SSH key: {key_path} (will inject via SSH_PUBLIC_KEY env var)")
            break

    if ssh_pub_key:
        # RunPod's GraphQL mutations do not accept a publicKey field. The supported
        # method is to pass the key via the SSH_PUBLIC_KEY environment variable,
        # which RunPod's container startup injects into authorized_keys.
        env_vars["SSH_PUBLIC_KEY"] = ssh_pub_key

    reg_volume = reg.get("network_volume_id")

    # Try each GPU candidate in preference order until one provisions successfully.
    # First try COMMUNITY cloud, then fall back to SECURE if all community GPUs
    # are exhausted (SECURE uses cloud providers like AWS/GCP — higher availability).
    pod = None
    last_error = None
    price_key = "communitySpotPrice" if is_spot else "communityPrice"

    for cloud_type in ["COMMUNITY", "SECURE"]:
        if pod and pod.get("id"):
            break
        if cloud_type == "SECURE":
            # SECURE cloud doesn't support spot — switch to on-demand pricing
            if is_spot:
                print("  Note: SECURE cloud does not support spot; using on-demand.")
            secure_price_key = "securePrice"
            print(f"  Retrying with SECURE cloud...")

        for gpu in gpu_candidates:
            gpu_name = gpu.get("id", gpu.get("displayName"))
            if cloud_type == "SECURE":
                hourly_rate = gpu.get(secure_price_key, 0) or gpu.get(price_key, 0)
                if not gpu.get("secureCloud"):
                    continue  # GPU not available on secure cloud
            else:
                hourly_rate = gpu.get(price_key, 0)

            pod_config = {
                "name": f"mmaze-{workload}-{agent}-{backend}",
                "image_name": DOCKER_IMAGE,
                "gpu_type_id": gpu_name,
                "cloud_type": cloud_type,
                "volume_in_gb": profile["volume_gb"],
                "container_disk_in_gb": profile["container_disk_gb"],
                "min_vcpu_count": profile["vcpu"],
                "min_memory_in_gb": profile["ram_gb"],
                "ports": "8888/http,22/tcp",
                "volume_mount_path": VOLUME_MOUNT,
                "env": env_vars,
            }
            if reg_volume:
                pod_config["network_volume_id"] = reg_volume

            print(f"  Trying GPU: {gpu_name} ({gpu.get('memoryInGb', '?')} GB, ${hourly_rate:.3f}/hr, {cloud_type}) ...", end="", flush=True)
            try:
                if is_spot and cloud_type == "COMMUNITY":
                    pod = _create_spot_pod(pod_config)
                else:
                    pod = _create_on_demand_pod(pod_config)
                if pod and pod.get("id"):
                    print(" OK")
                    break
                print(" no pod ID returned, trying next GPU")
            except Exception as e:
                last_error = e
                print(f" failed ({e}), trying next GPU")
                pod = None

    if not pod or not pod.get("id"):
        print(f"ERROR: All GPU candidates exhausted (COMMUNITY + SECURE). Last error: {last_error}", file=sys.stderr)
        sys.exit(1)

    pod_id = pod.get("id", "unknown")
    print(f"\nPod created: {pod_id}")
    print(f"  Status:   {pod.get('desiredStatus', 'UNKNOWN')}")

    # Register pod
    reg["pods"][pod_id] = {
        "purpose": workload,
        "requestor": args.requestor or "user",
        "gpu_type": gpu_name,
        "hourly_rate": hourly_rate,
        "spot": is_spot,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": pod_config,
        "env": env_vars,
        "cost_log": [],
    }
    save_registry(reg)

    # Print SSH info once ready
    _wait_and_print_ssh(pod_id)


def _create_on_demand_pod(config: dict) -> dict:
    """Create an on-demand pod using the RunPod SDK.

    Uses ``runpod.create_pod()`` which handles the GraphQL mutation internally
    and supports env vars, cloud_type, and volume mounts.
    """
    sdk_kwargs = {
        "name": config["name"],
        "image_name": config["image_name"],
        "gpu_type_id": config["gpu_type_id"],
        "cloud_type": config.get("cloud_type", "ALL"),
        "volume_in_gb": config.get("volume_in_gb", 20),
        "container_disk_in_gb": config.get("container_disk_in_gb", 20),
        "ports": config.get("ports", "8888/http,22/tcp"),
    }
    if config.get("volume_mount_path"):
        sdk_kwargs["volume_mount_path"] = config["volume_mount_path"]
    if config.get("min_vcpu_count"):
        sdk_kwargs["min_vcpu_count"] = config["min_vcpu_count"]
    if config.get("min_memory_in_gb"):
        sdk_kwargs["min_memory_in_gb"] = config["min_memory_in_gb"]
    if config.get("env"):
        sdk_kwargs["env"] = config["env"]
    if config.get("network_volume_id"):
        sdk_kwargs["network_volume_id"] = config["network_volume_id"]
    return runpod.create_pod(**sdk_kwargs)


def _create_spot_pod(config: dict) -> dict:
    """Create a spot (interruptable) pod via GraphQL, with on-demand fallback.

    The SDK's ``create_pod`` does not support spot instances directly, so we
    use the ``podRentInterruptable`` GraphQL mutation. If that fails, falls
    back to on-demand via the SDK.
    """
    try:
        from runpod.api.graphql import run_graphql_query

        env_entries = ""
        if config.get("env"):
            pairs = ", ".join(
                f'{{ key: "{k}", value: "{v}" }}' for k, v in config["env"].items()
            )
            env_entries = f"env: [{pairs}]"

        volume_part = ""
        if config.get("network_volume_id"):
            volume_part = f'networkVolumeId: "{config["network_volume_id"]}"'

        cloud_type = config.get("cloud_type", "COMMUNITY")
        mutation = f"""
        mutation {{
            podRentInterruptable(input: {{
                name: "{config['name']}"
                imageName: "{config['image_name']}"
                gpuTypeId: "{config['gpu_type_id']}"
                cloudType: {cloud_type}
                volumeInGb: {config['volume_in_gb']}
                containerDiskInGb: {config['container_disk_in_gb']}
                minVcpuCount: {config.get('min_vcpu_count', 4)}
                minMemoryInGb: {config.get('min_memory_in_gb', 8)}
                ports: "{config['ports']}"
                volumeMountPath: "{config.get('volume_mount_path', '/workspace')}"
                {env_entries}
                {volume_part}
            }}) {{
                id
                desiredStatus
                imageName
                machineId
                machine {{
                    podHostId
                }}
            }}
        }}
        """
        response = run_graphql_query(mutation)
        if response and "data" in response and response["data"].get("podRentInterruptable"):
            return response["data"]["podRentInterruptable"]
    except Exception:
        pass  # Fall through to on-demand

    # Fallback: on-demand via SDK
    print("  Note: Spot creation unavailable, falling back to on-demand.")
    return _create_on_demand_pod(config)


def _wait_and_print_ssh(pod_id: str, timeout: int = 120):
    """Poll pod status until RUNNING, then print SSH connection info."""
    print("\nWaiting for pod to start...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            pod = runpod.get_pod(pod_id)
        except Exception:
            time.sleep(5)
            continue

        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime")

        if status == "RUNNING" and runtime:
            ports = runtime.get("ports", [])
            ssh_port = None
            for port_info in ports:
                if port_info.get("privatePort") == 22:
                    ssh_port = port_info.get("publicPort")
                    ssh_host = port_info.get("ip")
                    break

            print(f"\n\nPod {pod_id} is RUNNING!")
            if ssh_port and ssh_host:
                print(f"  SSH: ssh -o StrictHostKeyChecking=no root@{ssh_host} -p {ssh_port} -i ~/.ssh/id_rsa")
                print(f"  SSH_HOST={ssh_host}")
                print(f"  SSH_PORT={ssh_port}")
            else:
                print("  SSH details not yet available. Check: python runpod/pod_manager.py status " + pod_id)
            return

        if status in ("EXITED", "TERMINATED", "ERROR"):
            print(f"\nPod entered unexpected state: {status}")
            return

        print(".", end="", flush=True)
        time.sleep(5)

    print(f"\nTimeout ({timeout}s) waiting for pod to start. Check status manually.")


def cmd_list(args):
    """List all tracked pods with current status."""
    reg = load_registry()
    pods = reg.get("pods", {})

    if not pods:
        print("No tracked pods.")
        return

    print(f"{'Pod ID':<25} {'Workload':<12} {'GPU':<30} {'Spot':>5} {'Rate':>10} {'Status'}")
    print("-" * 100)

    for pod_id, info in pods.items():
        # Fetch live status
        try:
            live = runpod.get_pod(pod_id)
            status = live.get("desiredStatus", "UNKNOWN")
        except Exception:
            status = "UNREACHABLE"

        gpu = info.get("gpu_type", "?")[:28]
        rate = f"${info.get('hourly_rate', 0):.3f}/hr"
        spot = "Yes" if info.get("spot") else "No"
        workload = info.get("purpose", "?")

        print(f"{pod_id:<25} {workload:<12} {gpu:<30} {spot:>5} {rate:>10} {status}")


def cmd_status(args):
    """Show detailed status for a specific pod or all pods."""
    reg = load_registry()
    pod_id = args.pod_id

    if pod_id:
        _show_pod_status(pod_id, reg)
    else:
        for pid in reg.get("pods", {}):
            _show_pod_status(pid, reg)
            print()


def _show_pod_status(pod_id: str, reg: dict):
    """Print detailed status for a single pod."""
    info = reg.get("pods", {}).get(pod_id)

    try:
        live = runpod.get_pod(pod_id)
    except Exception as e:
        print(f"Pod {pod_id}: UNREACHABLE ({e})")
        return

    status = live.get("desiredStatus", "UNKNOWN")
    runtime = live.get("runtime") or {}
    gpu_util = runtime.get("gpus", [{}])

    print(f"Pod: {pod_id}")
    print(f"  Status:     {status}")
    print(f"  GPU:        {info.get('gpu_type', '?') if info else live.get('gpuTypeId', '?')}")

    if info:
        print(f"  Workload:   {info.get('purpose', '?')}")
        print(f"  Requestor:  {info.get('requestor', '?')}")
        print(f"  Spot:       {info.get('spot', '?')}")
        print(f"  Rate:       ${info.get('hourly_rate', 0):.3f}/hr")

        # Calculate running cost
        created = info.get("created_at")
        if created and status == "RUNNING":
            created_dt = datetime.fromisoformat(created)
            hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
            session_cost = hours * info.get("hourly_rate", 0)
            print(f"  Uptime:     {hours:.1f} hours")
            print(f"  Session $:  ${session_cost:.2f}")

        logged_cost = sum(e.get("cost", 0) for e in info.get("cost_log", []))
        if logged_cost > 0:
            print(f"  Logged $:   ${logged_cost:.2f}")

    # GPU utilization
    if gpu_util:
        for i, g in enumerate(gpu_util):
            util = g.get("gpuUtilPercent", "?")
            mem = g.get("memoryUtilPercent", "?")
            print(f"  GPU[{i}]:     {util}% util, {mem}% mem")

    # SSH info
    ports = runtime.get("ports", [])
    for port_info in ports:
        if port_info.get("privatePort") == 22:
            print(f"  SSH:        ssh root@{port_info.get('ip')} -p {port_info.get('publicPort')}")
            break


def cmd_stop(args):
    """Stop a pod (preserves volume, stops billing for compute)."""
    pod_id = args.pod_id
    reg = load_registry()

    # Log cost before stopping
    _log_session_cost(pod_id, reg, reason="manual_stop")

    try:
        runpod.stop_pod(pod_id)
        print(f"Pod {pod_id} stopped.")
    except Exception as e:
        print(f"ERROR stopping pod {pod_id}: {e}", file=sys.stderr)
        sys.exit(1)

    save_registry(reg)


def cmd_resume(args):
    """Resume a stopped pod."""
    pod_id = args.pod_id

    try:
        runpod.resume_pod(pod_id, gpu_count=1)
        print(f"Pod {pod_id} resuming...")
        _wait_and_print_ssh(pod_id)
    except Exception as e:
        print(f"ERROR resuming pod {pod_id}: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_terminate(args):
    """Permanently delete a pod."""
    pod_id = args.pod_id
    reg = load_registry()

    # Log final cost
    _log_session_cost(pod_id, reg, reason="terminate")

    try:
        runpod.terminate_pod(pod_id)
        print(f"Pod {pod_id} terminated.")
    except Exception as e:
        print(f"ERROR terminating pod {pod_id}: {e}", file=sys.stderr)
        sys.exit(1)

    # Remove from registry
    if pod_id in reg.get("pods", {}):
        del reg["pods"][pod_id]
    save_registry(reg)


def cmd_cost(args):
    """Show cost summary per pod and cumulative total."""
    reg = load_registry()
    pods = reg.get("pods", {})
    budget = reg.get("budget", PROJECT_BUDGET)
    total = 0.0

    print(f"{'Pod ID':<25} {'Workload':<12} {'Logged':>10} {'Current':>10} {'Total':>10}")
    print("-" * 75)

    for pod_id, info in pods.items():
        logged = sum(e.get("cost", 0) for e in info.get("cost_log", []))

        # Estimate current session cost
        current = 0.0
        try:
            live = runpod.get_pod(pod_id)
            if live.get("desiredStatus") == "RUNNING":
                created = info.get("created_at")
                if created:
                    created_dt = datetime.fromisoformat(created)
                    hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
                    current = hours * info.get("hourly_rate", 0)
        except Exception:
            pass

        pod_total = logged + current
        total += pod_total
        workload = info.get("purpose", "?")
        print(f"{pod_id:<25} {workload:<12} ${logged:>9.2f} ${current:>9.2f} ${pod_total:>9.2f}")

    print("-" * 75)
    print(f"{'TOTAL':<37} {'':>10} {'':>10} ${total:>9.2f}")
    print(f"Budget: ${budget:.2f}  |  Remaining: ${budget - total:.2f}  |  Used: {total / budget * 100:.1f}%")

    _check_budget_warnings(total, budget)


def cmd_cleanup(args):
    """Flag idle pods (30+ min with 0% GPU utilization)."""
    reg = load_registry()
    pods = reg.get("pods", {})
    idle_pods = []

    for pod_id, info in pods.items():
        try:
            live = runpod.get_pod(pod_id)
            if live.get("desiredStatus") != "RUNNING":
                continue

            runtime = live.get("runtime") or {}
            gpu_utils = runtime.get("gpus", [])

            # Check if all GPUs are at 0%
            all_idle = all(
                g.get("gpuUtilPercent", 100) == 0
                for g in gpu_utils
            ) if gpu_utils else False

            if not all_idle:
                continue

            # Check uptime > 30 min
            created = info.get("created_at")
            if created:
                created_dt = datetime.fromisoformat(created)
                minutes = (datetime.now(timezone.utc) - created_dt).total_seconds() / 60
                if minutes >= 30:
                    hourly = info.get("hourly_rate", 0)
                    idle_pods.append((pod_id, info.get("purpose"), minutes, hourly))

        except Exception:
            continue

    if not idle_pods:
        print("No idle pods detected.")
        return

    print("Idle pods (30+ min, 0% GPU utilization):")
    print(f"{'Pod ID':<25} {'Workload':<12} {'Idle (min)':>10} {'Rate':>10}")
    print("-" * 65)
    for pod_id, workload, mins, rate in idle_pods:
        print(f"{pod_id:<25} {workload:<12} {mins:>10.0f} ${rate:.3f}/hr")
    print(f"\nTo stop:      python runpod/pod_manager.py stop <pod_id>")
    print(f"To terminate: python runpod/pod_manager.py terminate <pod_id>")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_session_cost(pod_id: str, reg: dict, reason: str = ""):
    """Log the current session cost to the registry."""
    info = reg.get("pods", {}).get(pod_id)
    if not info:
        return

    created = info.get("created_at")
    if not created:
        return

    try:
        live = runpod.get_pod(pod_id)
        if live.get("desiredStatus") != "RUNNING":
            return
    except Exception:
        return

    created_dt = datetime.fromisoformat(created)
    hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
    cost = hours * info.get("hourly_rate", 0)

    if cost > 0:
        info.setdefault("cost_log", []).append({
            "session": len(info.get("cost_log", [])) + 1,
            "cost": round(cost, 2),
            "hours": round(hours, 2),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def _check_budget_warnings(spent: float, budget: float):
    """Print budget warnings at threshold levels."""
    if budget <= 0:
        return
    ratio = spent / budget
    for threshold in BUDGET_WARN_THRESHOLDS:
        if ratio >= threshold:
            pct = int(threshold * 100)
            if threshold >= 1.0:
                print(f"\n*** BUDGET EXCEEDED: ${spent:.2f} / ${budget:.2f} ({ratio * 100:.0f}%) ***")
            else:
                print(f"\nWARNING: Budget {pct}% used — ${spent:.2f} / ${budget:.2f}")
            break  # Only show highest applicable warning


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RunPod pod lifecycle manager for Memory Maze training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # gpus
    sub.add_parser("gpus", help="List available GPUs with pricing")

    # recommend
    p_rec = sub.add_parser("recommend", help="Show recommended config for workload")
    p_rec.add_argument("--workload", required=True, choices=WORKLOAD_PROFILES.keys())

    # create
    p_create = sub.add_parser("create", help="Create a pod")
    p_create.add_argument("--workload", required=True, choices=WORKLOAD_PROFILES.keys())
    p_create.add_argument("--agent", choices=["impala"], default="impala")
    p_create.add_argument("--backend", choices=["mujoco", "genesis"], default="mujoco")
    p_create.add_argument("--maze-size", default="9x9")
    p_create.add_argument("--total-steps", type=int)
    p_create.add_argument("--on-demand", action="store_true",
                          help="Force on-demand (non-spot) instance for SSH stability")
    p_create.add_argument("--spot", action="store_true",
                          help="Force spot instance even for workloads that default to on-demand")
    p_create.add_argument("--requestor", default="user", help="Who requested this pod")
    p_create.add_argument("--wandb-api-key")
    p_create.add_argument("--xpid", help="Experiment ID (for spot resume)")
    p_create.add_argument("--extra-flags", help="Extra flags for training script")

    # list
    sub.add_parser("list", help="List tracked pods")

    # status
    p_status = sub.add_parser("status", help="Detailed pod status")
    p_status.add_argument("pod_id", nargs="?", help="Pod ID (omit for all)")

    # stop
    p_stop = sub.add_parser("stop", help="Stop a pod")
    p_stop.add_argument("pod_id")

    # resume
    p_resume = sub.add_parser("resume", help="Resume a stopped pod")
    p_resume.add_argument("pod_id")

    # terminate
    p_term = sub.add_parser("terminate", help="Terminate (delete) a pod")
    p_term.add_argument("pod_id")

    # cost
    sub.add_parser("cost", help="Cost summary")

    # cleanup
    sub.add_parser("cleanup", help="Flag idle pods")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    init_api()

    commands = {
        "gpus": cmd_gpus,
        "recommend": cmd_recommend,
        "create": cmd_create,
        "list": cmd_list,
        "status": cmd_status,
        "stop": cmd_stop,
        "resume": cmd_resume,
        "terminate": cmd_terminate,
        "cost": cmd_cost,
        "cleanup": cmd_cleanup,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
