import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request

import websocket

from Adam.ADAM import ADAM

DEFAULT_VIEWER_PORT = 3007
DEFAULT_GAME_SERVER_PORT = 3000
MINEFLAYER_PATTERN = "/root/ADAM-sparse/env/mineflayer/index.js"
DEFAULT_GOAL_ITEMS = ["crafting_table"]
DEFAULT_GOAL_ENVIRONMENT = ["grass"]
QUIET_BOOT = os.environ.get("ADAM_QUIET_BOOT", "1") != "0"


def boot_print(message):
    if not QUIET_BOOT:
        print(message)


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
        arg_parts = args.split()
        executable = os.path.basename(arg_parts[0]) if arg_parts else ""
        if executable.startswith("python") and any(
            os.path.basename(part) == "run.py" for part in arg_parts[1:]
        ):
            should_stop = True
        elif MINEFLAYER_PATTERN in args:
            for port in server_ports:
                if f" {port} " in f" {args} ":
                    should_stop = True
                    break
        if should_stop:
            boot_print(f"Stopping stale process PID {pid}: {args}")
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
        boot_print(f"Force killing stale process PID {pid}")
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
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                continue
            args = fields[1]
            executable = os.path.basename(args.split()[0]) if args.split() else ""
            if "chrome" not in executable and "chromium" not in executable:
                continue
            if "adam-gpu-viewer-profile" in args:
                continue
            boot_print(
                f"Detected existing non-GPU Chrome/Chromium instance; ignoring it for viewer launch: {line.strip()}"
            )
            break

    stale_gpu_chrome = []
    if process_listing:
        for line in process_listing.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                continue
            pid = int(fields[0])
            args = fields[1]
            executable = os.path.basename(args.split()[0]) if args.split() else ""
            if "adam-gpu-viewer-profile" not in args:
                continue
            if "chrome" not in executable and "chromium" not in executable:
                continue
            stale_gpu_chrome.append((pid, args))

    if stale_gpu_chrome:
        for pid, args in stale_gpu_chrome:
            boot_print(f"Stopping stale GPU Chrome viewer PID {pid}: {args}")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue
        time.sleep(2)
        for pid, args in stale_gpu_chrome:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            boot_print(f"Force killing stale GPU Chrome viewer PID {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue

    try:
        chrome_log = open("/tmp/adam-chrome-viewer.log", "ab", buffering=0)
        subprocess.Popen(
            ["/root/start-chrome-gpu.sh", viewer_url],
            env=env,
            stdout=chrome_log,
            stderr=chrome_log,
            start_new_session=True,
        )
        time.sleep(5)
        try:
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    "DISPLAY=${DISPLAY:-:1} "
                    "wid=$(xdotool search --onlyvisible --name 'Prismarine Viewer - Google Chrome' | tail -1) "
                    "&& [ -n \"$wid\" ] "
                    "&& xdotool windowactivate --sync \"$wid\" "
                    "&& xdotool windowmove \"$wid\" 320 27 "
                    "&& xdotool windowsize \"$wid\" 960 1080",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except Exception:
            pass
        return True
    except Exception:
        return False


def wait_for_tcp_port(host, port, timeout_seconds=180):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def call_chrome_devtools(websocket_url, method, params=None, origin="http://127.0.0.1:9222"):
    ws = websocket.create_connection(websocket_url, timeout=10, origin=origin)
    try:
        ws.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
        while True:
            message = json.loads(ws.recv())
            if message.get("id") != 1:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})
    finally:
        ws.close()


def get_chrome_pages():
    with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def refresh_existing_viewer_tab(viewer_url):
    pages = get_chrome_pages()
    viewer_pages = [
        page for page in pages
        if page.get("type") == "page"
        and page.get("webSocketDebuggerUrl")
        and viewer_url in page.get("url", "")
    ]
    if not viewer_pages:
        return False

    viewer_pages.sort(key=lambda page: page.get("title") != "Prismarine Viewer")
    active_page = viewer_pages[0]
    call_chrome_devtools(
        active_page["webSocketDebuggerUrl"],
        "Page.navigate",
        {"url": viewer_url},
    )

    for stale_page in viewer_pages[1:]:
        try:
            call_chrome_devtools(stale_page["webSocketDebuggerUrl"], "Page.close")
        except Exception:
            pass
    return True


def refresh_viewer_when_ready(viewer_url, viewer_port):
    def _worker():
        if not wait_for_tcp_port("127.0.0.1", viewer_port):
            print(f"Viewer port {viewer_port} did not become ready; browser was not refreshed.")
            return

        display = detect_server_display()
        env = os.environ.copy()
        if display:
            env["DISPLAY"] = display
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

        try:
            refreshed_existing_tab = refresh_existing_viewer_tab(viewer_url)
        except Exception:
            refreshed_existing_tab = False

        if not refreshed_existing_tab:
            try:
                chrome_log = open("/tmp/adam-chrome-viewer.log", "ab", buffering=0)
                subprocess.Popen(
                    ["/root/start-chrome-gpu.sh", viewer_url],
                    env=env,
                    stdout=chrome_log,
                    stderr=chrome_log,
                    start_new_session=True,
                )
            except Exception as exc:
                print(f"Failed to refresh Mineflayer viewer after startup: {exc}")
                return

        try:
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    "DISPLAY=${DISPLAY:-:1} "
                    "wid=$(xdotool search --onlyvisible --name 'Prismarine Viewer - Google Chrome' | tail -1) "
                    "&& [ -n \"$wid\" ] "
                    "&& xdotool windowactivate --sync \"$wid\" "
                    "&& xdotool key --clearmodifiers ctrl+r",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except Exception:
            pass

        print(f"Mineflayer viewer refreshed after port {viewer_port} became ready.")

    threading.Thread(target=_worker, name="ADAMViewerRefresh", daemon=True).start()


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
        if any(token in line for token in ("chrome", "chromium", "minecraft-launcher", "java")):
            interesting.append(line.strip())

    if interesting:
        print("GPU-attached graphics processes:")
        for line in interesting:
            print(line)
    else:
        print("GPU-attached graphics processes: none detected yet")


def build_visual_image_dir(goal_items):
    safe_goal = "-".join(goal_items) if goal_items else "unknown-task"
    safe_goal = re.sub(r"[^A-Za-z0-9._-]+", "-", safe_goal).strip("-") or "unknown-task"
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("Adam", f"game_image_{safe_goal}_{run_stamp}")

llm_config = load_llm_config("API_key.txt")
openai_api_key = llm_config["api_key"]
if llm_config["base_url"]:
    os.environ["OPENAI_BASE_URL"] = llm_config["base_url"]
    boot_print(f"Using OPENAI_BASE_URL={llm_config['base_url']}")
boot_print(f"Using LLM model: {llm_config['model']}")

goal_items = DEFAULT_GOAL_ITEMS
goal_environment = DEFAULT_GOAL_ENVIRONMENT

mc_port = detect_minecraft_lan_port()
boot_print(f"Using Minecraft LAN port: {mc_port}")
viewer_port = int(os.environ.get("ADAM_VIEWER_PORT", DEFAULT_VIEWER_PORT))
viewer_url = f"http://127.0.0.1:{viewer_port}"
visual_image_dir = build_visual_image_dir(goal_items)
os.environ["ADAM_VISUAL_IMAGE_DIR"] = visual_image_dir
boot_print(f"Visual screenshot directory: {visual_image_dir}")
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
    infer_sampling_num=1,
)

print(f"Mineflayer viewer URL: {viewer_url}")
if open_viewer_in_browser(viewer_url):
    print("Opened Mineflayer viewer in browser.")
    refresh_viewer_when_ready(viewer_url, viewer_port)
else:
    print("Failed to auto-open browser. Open the viewer URL manually.")
time.sleep(2)
print_gpu_process_status()
ADAM.run_visual_API()
print("Visual screenshot capture enabled.")

try:
    ADAM.explore(goal_items, goal_environment)
finally:
    ADAM.stop_visual_API()
    ADAM.env.close()
