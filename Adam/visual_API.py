import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import websocket

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
        self.chrome_debug_url = os.environ.get("ADAM_CHROME_DEBUG_URL", "http://127.0.0.1:9222/json")

    def detect_capture_tool(self):
        for tool in ("import", "xwd"):
            if shutil_which(tool):
                return tool
        raise RuntimeError("No supported X11 screenshot tool found. Install ImageMagick 'import' or xwd.")

    def find_viewer_window_id(self):
        search_commands = [
            ["xdotool", "search", "--onlyvisible", "--name", self.viewer_url],
            ["xdotool", "search", "--onlyvisible", "--name", "Google Chrome"],
            ["xdotool", "search", "--onlyvisible", "--name", "Chrome"],
            ["xdotool", "search", "--onlyvisible", "--name", "Chromium"],
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
        raise RuntimeError(
            f"Could not find a visible browser viewer window for {self.viewer_url}"
        )

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

    def find_viewer_devtools_page(self):
        with urllib.request.urlopen(self.chrome_debug_url, timeout=5) as response:
            pages = json.loads(response.read().decode("utf-8"))
        viewer_pages = [
            page for page in pages
            if page.get("type") == "page"
            and page.get("webSocketDebuggerUrl")
            and self.viewer_url in page.get("url", "")
        ]
        if not viewer_pages:
            raise RuntimeError(f"Could not find Chrome DevTools page for {self.viewer_url}")
        viewer_pages.sort(key=lambda page: page.get("title") != "Prismarine Viewer")
        return viewer_pages[0]

    def capture_devtools_page(self, screenshot_path):
        page = self.find_viewer_devtools_page()
        ws = websocket.create_connection(
            page["webSocketDebuggerUrl"],
            timeout=10,
            origin="http://127.0.0.1:9222",
        )
        message_id = 0

        def send(method, params=None):
            nonlocal message_id
            message_id += 1
            ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
            while True:
                message = json.loads(ws.recv())
                if message.get("id") != message_id:
                    continue
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

        try:
            send("Page.enable")
            send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": int(os.environ.get("ADAM_VISUAL_CAPTURE_WIDTH", "1280")),
                    "height": int(os.environ.get("ADAM_VISUAL_CAPTURE_HEIGHT", "720")),
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            result = send("Page.captureScreenshot", {"format": "png", "fromSurface": True})
            image_data = result.get("data")
            if not image_data:
                raise RuntimeError("Chrome DevTools returned an empty screenshot")
            with open(screenshot_path, "wb") as image_file:
                image_file.write(base64.b64decode(image_data))
        finally:
            ws.close()

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
                date_prefix = time.strftime("%Y%m%d")
                sequence_name = f"{date_prefix}_{self.capture_index:04d}.png"
                sequence_path = os.path.join(self.image_dir, sequence_name)
                latest_path = os.path.join(self.image_dir, "tmp.png")
                try:
                    self.capture_devtools_page(sequence_path)
                except Exception as devtools_error:
                    window_id = self.find_viewer_window_id()
                    self.capture_window(window_id, sequence_path)
                    print(f"DevTools screenshot failed, used X11 fallback: {devtools_error}", flush=True)
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
