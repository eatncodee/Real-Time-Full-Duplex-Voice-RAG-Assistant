import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_stt(file_path):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": os.getenv("SARVAM_API_KEY", "")
    }
    
    with open(file_path, "rb") as f:
        files = {
            "file": (file_path, f, "audio/wav")
        }
        data = {
            "model": "saaras:v1" # or saaras:v3
        }
        response = requests.post(url, headers=headers, files=files, data=data)
        
    print(response.status_code)
    print(response.text)

if __name__ == "__main__":
    test_stt("test_audio.wav")
