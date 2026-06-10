import requests


def get_image_description(image_path='Adam/game_image/tmp.png', local_mllm_port=7000):
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
