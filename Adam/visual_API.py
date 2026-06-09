import os
import shutil
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils as U


class VisualAPI:
    def __init__(self):
        self.viewer_url = os.environ.get("ADAM_VISUAL_API_URL", "http://127.0.0.1:3007")
        self.image_dir = os.environ.get("ADAM_VISUAL_IMAGE_DIR", "Adam/game_image")
        self.capture_interval = float(os.environ.get("ADAM_VISUAL_CAPTURE_INTERVAL", "10"))
        self.display = os.environ.get("DISPLAY", ":1")
        self.capture_tool = self.detect_capture_tool()
        self.capture_index = self.detect_next_capture_index()

    def detect_capture_tool(self):
        for tool in ("import", "xwd"):
            if shutil_which(tool):
                return tool
        raise RuntimeError("No supported X11 screenshot tool found. Install ImageMagick 'import' or xwd.")

    def find_viewer_window_id(self):
        search_commands = [
            ["xdotool", "search", "--onlyvisible", "--name", self.viewer_url],
            ["xdotool", "search", "--onlyvisible", "--name", "Mozilla Firefox"],
        ]
        env = os.environ.copy()
        env["DISPLAY"] = self.display

        for command in search_commands:
            try:
                output = subprocess.check_output(
                    command,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    timeout=5,
                )
            except Exception:
                continue
            window_ids = [line.strip() for line in output.splitlines() if line.strip()]
            if window_ids:
                return window_ids[-1]
        raise RuntimeError(f"Could not find a visible Firefox viewer window for {self.viewer_url}")

    def capture_window(self, window_id, screenshot_path):
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        if self.capture_tool == "import":
            subprocess.check_call(
                ["import", "-window", window_id, screenshot_path],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        with open(screenshot_path, "wb") as output_file:
            subprocess.check_call(
                ["xwd", "-silent", "-id", window_id],
                env=env,
                stdout=output_file,
                stderr=subprocess.DEVNULL,
            )

    def detect_next_capture_index(self):
        if not os.path.isdir(self.image_dir):
            return 1
        prefix = time.strftime("%Y%m%d")
        max_index = 0
        for name in os.listdir(self.image_dir):
            if not name.startswith(prefix + "_") or not name.endswith(".png"):
                continue
            stem = name[:-4]
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            seq = parts[-1]
            if seq.isdigit():
                max_index = max(max_index, int(seq))
        return max_index + 1

    def run(self):
        U.f_mkdir(self.image_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        print("Visual API Ready", flush=True)
        while True:
            try:
                window_id = self.find_viewer_window_id()
                date_prefix = time.strftime("%Y%m%d")
                sequence_name = f"{date_prefix}_{self.capture_index:04d}.png"
                sequence_path = os.path.join(self.image_dir, sequence_name)
                latest_path = os.path.join(self.image_dir, "tmp.png")
                self.capture_window(window_id, sequence_path)
                shutil.copyfile(sequence_path, latest_path)
                self.capture_index += 1
            except Exception as error:
                print(f"Error: {error}", flush=True)
            time.sleep(self.capture_interval)


def shutil_which(name):
    return subprocess.call(
        ["bash", "-lc", f"command -v {name} >/dev/null 2>&1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0


module = VisualAPI()
module.run()
