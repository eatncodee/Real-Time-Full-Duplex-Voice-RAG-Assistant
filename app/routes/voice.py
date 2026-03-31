from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.streaming import stream_rag_response
import io,wave
import re
import httpx
import os
import numpy as np
import asyncio
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
    def __init__(self, threshold=700, silence_duration=0.4):
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
            if self.has_spoken:
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()

                if (time.time() - self.silence_start_time) >= self.silence_duration:
                    self.has_spoken = False # Reset for next turn
                    self.silence_start_time = None
                    return True # TRIGGER: User is done

            return False

    

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
    try:
        await websocket.send_json({"type": "status", "message": "✅ Connected & Ready"})
        print(f"✅ Voice client connected and handshake sent: {userId}")
    except Exception:
        print(f"⚠️ Client {userId} disconnected during handshake. Skipping.")
        return    

    
    detector = SilenceDetector(threshold=700, silence_duration=0.4)
    master_buffer = bytearray()


    sentence_queue = asyncio.Queue()
    sentence_buffer = ""
    
    # Start the TTS worker immediately so the Cartesia handshake happens NOW
    tts_task = asyncio.create_task(cartesia_tts_worker(sentence_queue, websocket))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                chunk = message["bytes"]
                master_buffer.extend(chunk)

                if not detector.has_spoken:
                    if len(master_buffer) > 32000:
                        master_buffer = master_buffer[-32000:]

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

                    async def on_text_chunk(text: str):
                        nonlocal sentence_buffer
                        
                        # 1. 🟢 Preserve RAW text in the buffer (Don't clean yet!)
                        sentence_buffer += text
                        
                        # 2. ⚡ EMERGENCY SPLIT: Latency Guard
                        words = sentence_buffer.split()
                        if len(words) > 12:
                            # Clean the phrase ONLY when it's ready to be spoken
                            raw_phrase = " ".join(words[:10])
                            clean_phrase = clean_text_for_tts(raw_phrase)
                            
                            if clean_phrase:
                                await sentence_queue.put(clean_phrase)
                                
                            sentence_buffer = " ".join(words[10:])
                            return

                        # 3. 🌬️ NATURAL SPLIT: Breath-based logic
                        pattern = r'(?<=[.!?,\n])\s+'
                        parts = re.split(pattern, sentence_buffer)
                        
                        for part in parts[:-1]:
                            # 🟢 Clean the phrase right before sending to Cartesia
                            clean_phrase = clean_text_for_tts(part)
                            if clean_phrase:
                                await sentence_queue.put(clean_phrase)
                        
                        sentence_buffer = parts[-1]


                    # --- 4. START OPENAI / RAG ---
                    result = await stream_rag_response(
                        user_message=user_text, 
                        conversation_history=current_history, 
                        websocket=websocket,
                        text_chunk_callback=on_text_chunk
                    )

                    # --- 5. CLEANUP & FLUSH ---
                    # Push any leftover words in the buffer
                    if sentence_buffer.strip():
                        await sentence_queue.put(sentence_buffer.strip())

                    # Send the "Poison Pill" so the worker knows OpenAI is done
                    await sentence_queue.put(None)

                    # Wait for the Cartesia worker to finish speaking the last sentence
                    await tts_task

                    if result:
                        new_history = result.get("conversation_history", [])
                        user_histories[userId] = new_history
                        data = result.get("answer", "")

        
    except WebSocketDisconnect:
        print("❌ Voice client disconnected")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def clean_text_for_tts(text):
    text = re.sub(r'\*+', '', text)      # Bold/Italic
    text = re.sub(r'_+', '', text)       # Underline/Italic
    text = re.sub(r'#+\s?', '', text)    # Headers
    text = re.sub(r'`+', '', text)       # Code blocks
    
    # 2. Remove List Markers (the starts of lines)
    text = re.sub(r'^\s*[-+*]\s+', '', text, flags=re.MULTILINE)
    # 3. 🟢 PROD ADDITION: Remove URLs (AI loves to yap links)
    text = re.sub(r'http[s]?://\S+', '', text)
    # 4. 🟢 PROD ADDITION: Remove LaTeX/Math symbols if they appear
    text = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', text)
    # 5. Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def cartesia_tts_worker(sentence_queue: asyncio.Queue, websocket):
    ws = await client.tts.websocket()
    try:
        while True:
            sentence = await sentence_queue.get()
            
# 2. The "Poison Pill": Stops the worker when OpenAI is completely finished
            if sentence is None:
                sentence_queue.task_done()
                break
                
            clean_sentence = clean_text_for_tts(sentence)
            if not clean_sentence.strip():
                sentence_queue.task_done()
                continue

            stream = await ws.send(
                model_id="sonic-3",
                transcript=clean_sentence,
                voice={"mode": "id", "id": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": 44100
                },
            )
            
            async for output in stream:
                if output.audio is not None:
                    await websocket.send_bytes(output.audio)
                    
            sentence_queue.task_done()
            
    except Exception as e:
        print(f"⚠️ TTS Worker Error: {e}")
    finally:
        await ws.close()