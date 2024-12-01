import requests

# URL для вашего API
url = 'http://127.0.0.1:8000/asr/'

audio_file_path = 'Запись.m4a'

with open(audio_file_path, 'rb') as audio_file:
    files = {'file': (audio_file_path, audio_file, 'audio/mpeg')}

    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
