import os
import re
import signal
import subprocess
import time
import webbrowser

from Adam.ADAM import ADAM

DEFAULT_VIEWER_PORT = 3007
DEFAULT_GAME_SERVER_PORT = 3000
MINEFLAYER_PATTERN = "/root/ADAM-sparse/env/mineflayer/index.js"


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
    browser_commands = (
        ["firefox", viewer_url],
        ["xdg-open", viewer_url],
        ["gio", "open", viewer_url],
    )
    for command in browser_commands:
        try:
            subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            continue
    try:
        if display:
            os.environ["DISPLAY"] = display
        return webbrowser.open(viewer_url)
    except Exception:
        return False

with open("API_key.txt", 'r') as key_file:
    openai_api_key = key_file.read()

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
    llm_model_type='gpt-4-turbo',
    use_local_llm_service=False,
    openai_api_key=openai_api_key,
    game_server_port=DEFAULT_GAME_SERVER_PORT,
    game_visual_server_port=viewer_port,
    auto_load_ckpt=True,
    parallel=True
)

print(f"Mineflayer viewer URL: {viewer_url}")
if open_viewer_in_browser(viewer_url):
    print("Opened Mineflayer viewer in browser.")
else:
    print("Failed to auto-open browser. Open the viewer URL manually.")

ADAM.explore(['iron_ingot'], ['grass'])
