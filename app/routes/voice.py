from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.rag import chat_with_function_calling
from gtts import gTTS
import io
import re
import httpx
import os
from fastapi import UploadFile, File
from cartesia import Cartesia


router=APIRouter()

cleint=Cartesia(api_key=os.getenv("Cartesia_key"),)


client.tts.bytes(
    model_id="sonic-2",
    transcript="Hello, world!",
    voice={
        "mode": "id",
        "id": "694f9389-aac1-45b6-b726-9d9369183238",
    },
    language="en",
    output_format={
        "container": "wav",
        "sample_rate": 44100,
        "encoding": "pcm_f32le",
    },
)

async def procces_with_rag(user_text:str,conversation_history :list | None=None):
    response=chat_with_function_calling(user_message=user_text,conversation_history=conversation_history,temprature=0.9)

    if response:
        return response
    

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def clean_text_for_tts(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'#+\s?', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'^\s*[-+*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def text_to_speech(text:str):
    clean_text = clean_text_for_tts(text)    
    sentences=split_into_sentences(clean_text)

    for i,sentence in enumerate(sentences):
        tts =gTTS(text=sentence, lang='en', slow=False)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        yield audio_bytes

user_histories={}
@router.websocket("/ws/voice/{userId}")
async def voice_chat(websocket: WebSocket,userId:str):
    await websocket.accept()
    print("✅ Voice client connected")
    audio_output=None
    try:
        while True:
            current_history=user_histories.get(userId,[])
            data = await websocket.receive_json()
            user_text=data.get("text","")
            print(f"📥 Received {len(user_text)}")
            if not user_text:
                continue

            result =await procces_with_rag(user_text,current_history)

            if result :
                new_history=result.get("conversation_history",[])
                user_histories[userId]=new_history
                data=result.get("answer","")
                async for audio_chunk in text_to_speech(data):
                    if audio_chunk:
                        await websocket.send_bytes(audio_chunk)
                        print(f"📤 Sent chunk: {len(audio_chunk)} bytes")
        
    except WebSocketDisconnect:
        print("❌ Voice client disconnected")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

@router.post("/test-stt/")
async def test_sarvam_stt(file: UploadFile = File(...)):
    """
    Test endpoint for Sarvam AI STT mapping.
    Send an audio file as form-data via this endpoint to test transcription.
    """
    sarvam_url = "https://api.sarvam.ai/speech-to-text"
    api_key = os.getenv("SARVAM_API_KEY")
    
    if not api_key:
        return {"error": "SARVAM_API_KEY environment variable is missing. Please add it to your .env file."}
    
    headers = {
        "api-subscription-key": api_key
    }
    
    file_bytes = await file.read()
    
    files = {
        "file": (file.filename, file_bytes, file.content_type)
    }
    data = {
        "model": "saaras:v1" # Use saaras:v1 or saaras:v3 as supported by your plan
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                sarvam_url,
                headers=headers,
                data=data,
                files=files,
                timeout=30.0
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

