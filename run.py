import os
import re
import signal
import subprocess
import time

from Adam.ADAM import ADAM

DEFAULT_VIEWER_PORT = 3007
DEFAULT_GAME_SERVER_PORT = 3000
MINEFLAYER_PATTERN = "/root/ADAM-sparse/env/mineflayer/index.js"


def load_llm_config(config_path="API_key.txt"):
    with open(config_path, "r", encoding="utf-8") as key_file:
        raw = key_file.read().strip()

    if not raw:
        raise RuntimeError(f"{config_path} is empty")

    if "\n" not in raw and ":" not in raw:
        return {
            "api_key": raw,
            "base_url": "https://xiaoai.plus/v1",
            "model": "gpt-4-turbo",
        }

    config = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip().lower()] = value.strip()

    api_key = config.get("key") or config.get("api_key")
    if not api_key:
        raise RuntimeError(
            f"{config_path} must contain either a raw API key or key:/api_key: entries"
        )

    base_url = config.get("relay website") or config.get("relay") or config.get("base_url")
    if base_url:
        base_url = base_url.rstrip("/")
    else:
        base_url = "https://xiaoai.plus/v1"

    model = config.get("model") or "gpt-4-turbo"

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def detect_minecraft_lan_port():
    override = os.environ.get("ADAM_MC_PORT", "").strip()
    if override:
        return int(override)

    try:
        output = subprocess.check_output(
            ["ss", "-ltnp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        output = ""

    for line in output.splitlines():
        if "java" not in line:
            continue
        match = re.search(r":(\d+)\s+", line)
        if match:
            port = int(match.group(1))
            if port > 1024:
                return port

    log_path = "/root/.minecraft/logs/latest.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            matches = re.findall(r"Local game hosted on port (\d+)", log_file.read())
        if matches:
            return int(matches[-1])

    raise RuntimeError(
        "Could not find a running Minecraft LAN port. Open your world to LAN first, "
        "or set ADAM_MC_PORT explicitly."
    )


def stop_stale_run_and_mineflayer_processes(server_ports):
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return

    current_pid = os.getpid()
    stale_pids = []
    for line in output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        args = fields[1]
        if pid == current_pid:
            continue
        should_stop = False
        if "python3 run.py" in args or "python run.py" in args:
            should_stop = True
        elif MINEFLAYER_PATTERN in args:
            for port in server_ports:
                if f" {port} " in f" {args} ":
                    should_stop = True
                    break
        if should_stop:
            print(f"Stopping stale process PID {pid}: {args}")
            try:
                os.kill(pid, signal.SIGTERM)
                stale_pids.append(pid)
            except OSError:
                continue
    time.sleep(1)
    for pid in stale_pids:
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        print(f"Force killing stale process PID {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    time.sleep(1)
    wait_for_ports_to_close(server_ports)


def wait_for_ports_to_close(server_ports, timeout_seconds=5):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            output = subprocess.check_output(
                ["ss", "-ltnp"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return
        occupied = []
        for port in server_ports:
            if f":{port} " in output:
                occupied.append(port)
        if not occupied:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"Mineflayer ports still occupied after cleanup: {server_ports}. "
        f"Stop old run.py / mineflayer processes and retry."
    )


def detect_server_display():
    if os.environ.get("DISPLAY"):
        return os.environ["DISPLAY"]
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    match = re.search(r"Xtigervnc\s+(:\d+)", output)
    if match:
        return match.group(1)
    return None


def open_viewer_in_browser(viewer_url):
    display = detect_server_display()
    env = os.environ.copy()
    if display:
        env["DISPLAY"] = display
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

    try:
        process_listing = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        process_listing = ""

    if process_listing:
        for line in process_listing.splitlines():
            if "firefox" not in line:
                continue
            if "adam-gpu-viewer-profile" in line:
                continue
            print(f"Detected existing non-GPU Firefox instance; ignoring it for viewer launch: {line.strip()}")
            break

    stale_gpu_firefox = []
    if process_listing:
        for line in process_listing.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                continue
            pid = int(fields[0])
            args = fields[1]
            if "adam-gpu-viewer-profile" not in args:
                continue
            stale_gpu_firefox.append((pid, args))

    if stale_gpu_firefox:
        for pid, args in stale_gpu_firefox:
            print(f"Stopping stale GPU Firefox viewer PID {pid}: {args}")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue
        time.sleep(2)
        for pid, args in stale_gpu_firefox:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            print(f"Force killing stale GPU Firefox viewer PID {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue

    try:
        subprocess.Popen(
            ["/root/start-firefox-gpu.sh", viewer_url],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def print_gpu_process_status():
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("GPU status: unavailable")
        return

    interesting = []
    for line in output.splitlines():
        if any(token in line for token in ("firefox", "minecraft-launcher", "java")):
            interesting.append(line.strip())

    if interesting:
        print("GPU-attached graphics processes:")
        for line in interesting:
            print(line)
    else:
        print("GPU-attached graphics processes: none detected yet")

llm_config = load_llm_config("API_key.txt")
openai_api_key = llm_config["api_key"]
if llm_config["base_url"]:
    os.environ["OPENAI_BASE_URL"] = llm_config["base_url"]
    print(f"Using OPENAI_BASE_URL={llm_config['base_url']}")
print(f"Using LLM model: {llm_config['model']}")

mc_port = detect_minecraft_lan_port()
print(f"Using Minecraft LAN port: {mc_port}")
viewer_port = int(os.environ.get("ADAM_VIEWER_PORT", DEFAULT_VIEWER_PORT))
viewer_url = f"http://127.0.0.1:{viewer_port}"
max_parallel_envs = 2
stop_stale_run_and_mineflayer_processes(
    [DEFAULT_GAME_SERVER_PORT + i for i in range(max_parallel_envs)]
)

ADAM = ADAM(
    mc_port=mc_port,
    llm_model_type=llm_config["model"],
    use_local_llm_service=False,
    openai_api_key=openai_api_key,
    game_server_port=DEFAULT_GAME_SERVER_PORT,
    game_visual_server_port=viewer_port,
    auto_load_ckpt=False,
    parallel=False,
)

print(f"Mineflayer viewer URL: {viewer_url}")
if open_viewer_in_browser(viewer_url):
    print("Opened Mineflayer viewer in browser.")
else:
    print("Failed to auto-open browser. Open the viewer URL manually.")
time.sleep(2)
print_gpu_process_status()
ADAM.run_visual_API()
print("Visual screenshot capture enabled.")

ADAM.explore(['iron_ingot'], ['grass'])
