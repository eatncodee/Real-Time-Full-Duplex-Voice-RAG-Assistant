from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.rag import chat_with_function_calling
import io,wave
import re
import httpx
import os
import numpy as np
import time
from fastapi import UploadFile, File
from cartesia import AsyncCartesia
from sarvamai import AsyncSarvamAI

# STT--TTT-TTS--
# sarvamai-openai-cartesia


router=APIRouter()

client=AsyncCartesia(api_key=os.getenv("Cartesia_key"),)
sarvam_client=AsyncSarvamAI(api_subscription_key=os.getenv("Sarvam_key"))

class SilenceDetector:
    def __init__(self, threshold=700, silence_duration=0.8):
        self.threshold = threshold          # Min volume to count as "speech"
        self.silence_duration = silence_duration  # Seconds of silence to trigger stop
        self.silence_start_time = None
        self.has_spoken = False             # Ensures we don't trigger on initial silence
    def is_user_finished(self, audio_chunk_bytes):
        # 1. Convert bytes to Int16 (Frontend MUST send 16-bit PCM)
        audio_data = np.frombuffer(audio_chunk_bytes, dtype=np.int16)

        if len(audio_data) == 0:
            return False
        # 2. Calculate Volume (RMS)
        rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        # 3. Decision Logic
        if rms > self.threshold:
            self.has_spoken = True
            self.silence_start_time = None  # Reset timer because they are talking
            return False
        else:
            # If they have said something and now it's quiet
            if self.has_spoken:
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()

                # Check if silence has lasted long enough
                if (time.time() - self.silence_start_time) >= self.silence_duration:
                    self.has_spoken = False # Reset for next turn
                    self.silence_start_time = None
                    return True # TRIGGER: User is done

            return False


async def procces_with_rag(user_text:str,conversation_history :list | None=None):
    response=await chat_with_function_calling(user_message=user_text,conversation_history=conversation_history,temprature=0.9)

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


async def speech_to_text(audio_bytes:bytes) ->dict:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(16000)    # 16kHz — Sarvam requires this
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)

    response = await sarvam_client.speech_to_text.transcribe(
        file=("audio.wav", wav_buffer, "audio/wav"),
        model="saarika:v2.5",     
        language_code="unknown",  
    )
    return {
        "transcript": response.transcript,
        "language_code": response.language_code 
    }

user_histories={}
@router.websocket("/ws/voice/{userId}")
async def voice_chat(websocket: WebSocket,userId:str):
    await websocket.accept()
    print("✅ Voice client connected")
    detector = SilenceDetector(threshold=700, silence_duration=0.8)
    master_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                chunk = message["bytes"]
                master_buffer.extend(chunk)

                if detector.is_user_finished(chunk):
                    print("🤫 Silence detected. Processing...")

                    audio_to_process = bytes(master_buffer)
                    master_buffer.clear()

                    stt_result = await speech_to_text(audio_to_process)

                    user_text = stt_result["transcript"]
                    language  = stt_result["language_code"]
                    if not user_text:
                        continue

                    print(f"📝 Transcript ({language}): {user_text}")

                    current_history = user_histories.get(userId, [])
                    result = await procces_with_rag(user_text, current_history)

                    if result:
                        new_history = result.get("conversation_history", [])
                        user_histories[userId] = new_history
                        data = result.get("answer", "")

                        async for audio_chunk in text_to_speech(data):
                            if audio_chunk is not None:
                                await websocket.send_bytes(audio_chunk)
                                print(f"📤 Sent chunk: {len(audio_chunk)} bytes")
        
    except WebSocketDisconnect:
        print("❌ Voice client disconnected")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def text_to_speech(text: str):
    clean_text = clean_text_for_tts(text)
    sentences = split_into_sentences(clean_text)

    ws = await client.tts.websocket()
    try:
        for sentence in sentences:
            stream = await ws.send(
                model_id="sonic-3",
                transcript=sentence,
                voice={"mode": "id", "id": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": 44100
                },
            )
            async for output in stream:
                if output.audio is not None:
                    yield output.audio
    finally:
        await ws.close()

