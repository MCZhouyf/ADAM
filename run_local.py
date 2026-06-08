import os
import re
import signal
import subprocess
import sys
import time

from Adam.ADAM import ADAM
from Adam.skill_loader import skill_loader


LOCK_PATH = "/tmp/adam-run-local.lock"
MINEFLAYER_PATTERN = "/root/ADAM-sparse/env/mineflayer/index.js"
DEFAULT_ACTION = "gatherWoodLog"
DEFAULT_VIEWER_PORT = 3007
DEFAULT_OBSERVE_SECONDS = 15


SMOKE_TEST_INVENTORY = {
    "gatherWoodLog": {},
    "craftPlanks": {"oak_log": 1},
    "craftSticks": {"oak_planks": 2},
    "craftCraftingTable": {"oak_planks": 4},
    "craftWoodenPickaxe": {
        "crafting_table": 1,
        "oak_planks": 3,
        "stick": 2,
    },
}


def detect_minecraft_lan_port():
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

    raise RuntimeError("Could not find a running Minecraft LAN port. Open your world to LAN first.")


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_single_instance_lock():
    if os.path.exists(LOCK_PATH):
        with open(LOCK_PATH, "r", encoding="utf-8", errors="replace") as lock_file:
            old_pid_text = lock_file.read().strip()
        if old_pid_text.isdigit() and process_exists(int(old_pid_text)):
            raise RuntimeError(
                f"ADAM already appears to be running as PID {old_pid_text}. "
                f"Stop it first, or remove {LOCK_PATH} if that process is gone."
            )
        os.remove(LOCK_PATH)

    with open(LOCK_PATH, "w", encoding="utf-8") as lock_file:
        lock_file.write(str(os.getpid()))


def release_single_instance_lock():
    try:
        if os.path.exists(LOCK_PATH):
            with open(LOCK_PATH, "r", encoding="utf-8", errors="replace") as lock_file:
                if lock_file.read().strip() == str(os.getpid()):
                    os.remove(LOCK_PATH)
    except OSError:
        pass


def stop_stale_mineflayer_bridge(server_port=3000):
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return

    for line in output.splitlines():
        if MINEFLAYER_PATTERN not in line or f" {server_port} " not in f"{line} ":
            continue
        fields = line.strip().split(None, 1)
        if not fields or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        if pid == os.getpid():
            continue
        print(f"Stopping stale Mineflayer bridge PID {pid} on port {server_port}")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue

    time.sleep(1)


def get_openai_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    key_path = "API_key.txt"
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as key_file:
            return key_file.read().strip()

    raise RuntimeError(
        "Missing OpenAI API key. Set OPENAI_API_KEY or create /root/ADAM-sparse/API_key.txt."
    )


def parse_bot_position():
    raw = os.environ.get("ADAM_BOT_POSITION", "").strip()
    if not raw:
        return None
    parts = re.split(r"[,\s]+", raw)
    if len(parts) != 3:
        raise RuntimeError("ADAM_BOT_POSITION must have exactly three values: x y z")
    x, y, z = (float(part) for part in parts)
    return {"x": x, "y": y, "z": z}


def parse_cli_args():
    mc_port = None
    action = DEFAULT_ACTION
    for arg in sys.argv[1:]:
        if arg in {"python3", "python", "run_local.py"}:
            continue
        if arg.isdigit() and mc_port is None:
            mc_port = int(arg)
        elif action == DEFAULT_ACTION:
            action = arg
        else:
            raise RuntimeError(
                f"Unexpected argument: {arg}. Usage: python3 run_local.py [lan_port] [action_name]"
            )
    return mc_port, action


def get_smoke_test_inventory(action):
    return dict(SMOKE_TEST_INVENTORY.get(action, {}))


if __name__ == "__main__":
    acquire_single_instance_lock()
    try:
        stop_stale_mineflayer_bridge()
        mc_port_arg, action_name = parse_cli_args()
        mc_port = mc_port_arg if mc_port_arg is not None else detect_minecraft_lan_port()
        print(f"Using Minecraft LAN port: {mc_port}")
        os.environ.setdefault("OPENAI_BASE_URL", "https://xiaoai.plus/v1")
        bot_position = parse_bot_position()
        if bot_position:
            print(f"Using bot position: {bot_position}")
        else:
            print("Using live player tracking with automatic nearby safe-position search")

        agent = ADAM(
            mc_port=mc_port,
            llm_model_type=os.environ.get("ADAM_LLM_MODEL", "gpt-4-turbo"),
            use_local_llm_service=False,
            openai_api_key=get_openai_api_key(),
            game_visual_server_port=int(os.environ.get("ADAM_VIEWER_PORT", DEFAULT_VIEWER_PORT)),
            env_request_timeout=600,
            infer_sampling_num=1,
            max_try=1,
            auto_load_ckpt=True,
            parallel=False,
            reset_position=bot_position,
            track_player=bot_position is None,
        )

        options = {
            "mode": "hard",
            "inventory": get_smoke_test_inventory(action_name),
        }
        if bot_position:
            options["position"] = bot_position
        else:
            options["track_player"] = True

        reset_result = agent.env.reset(options=options)
        print(f"Running local visible smoke test: {action_name}")
        print("Injected inventory for smoke test:", options["inventory"])
        print("Note: this inventory is the bot inventory, not your player inventory.")
        reset_obs = reset_result[0][1]
        action_result = agent.env.step(skill_loader(action_name))
        time.sleep(1)
        result = agent.env.step("")
        action_obs = action_result[0][1]
        final_obs = result[0][1]
        viewer_status = (
            final_obs.get("viewerStatus")
            or action_obs.get("viewerStatus")
            or reset_obs.get("viewerStatus")
        )
        if viewer_status and str(viewer_status).startswith("http"):
            print("Mineflayer viewer URL:", viewer_status)
        else:
            print(
                "Mineflayer viewer status:",
                viewer_status or "disabled"
            )
        print("Bot inventory after local action:", final_obs["inventory"])
        print("Save marker after local action:", action_obs.get("onSave") or final_obs.get("onSave"))
        print("Chat after local action:", action_obs.get("onChat") or final_obs.get("onChat"))
        print("Error after local action:", action_obs.get("onError") or final_obs.get("onError"))
        observe_seconds = int(os.environ.get("ADAM_OBSERVE_SECONDS", DEFAULT_OBSERVE_SECONDS))
        if observe_seconds > 0:
            print(
                f"Keeping bot connected for {observe_seconds} seconds so you can see it in Minecraft."
            )
            time.sleep(observe_seconds)
        agent.env.close()
    finally:
        release_single_instance_lock()
