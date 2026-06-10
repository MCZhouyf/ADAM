import os

import requests


def get_image_description(image_path=None, local_mllm_port=7000):
    if image_path is None:
        image_dir = os.environ.get("ADAM_VISUAL_IMAGE_DIR", "Adam/game_image")
        image_path = os.path.join(image_dir, "tmp.png")
    text = 'Please describe this Minecraft image'
    url = 'http://localhost:' + str(local_mllm_port) + '/send_image_text'
    data = {'text': text}

    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(url, data=data, files=files, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as error:
        return f"Visual description unavailable: {error}"
