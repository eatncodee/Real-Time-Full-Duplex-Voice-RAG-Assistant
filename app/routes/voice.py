from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.streaming import stream_rag_response
from app.services.VAD import SilenceDetector
from app.services.STT import speech_to_text
import re
import httpx
import os
import json
import asyncio
from fastapi import UploadFile, File
from cartesia import AsyncCartesia
import random
from app.services.fillers import FILLERS


# STT--TTT-TTS--
# sarvamai-openai-cartesia


router=APIRouter()

client=AsyncCartesia(api_key=os.getenv("Cartesia_key"),)



user_histories={}
@router.websocket("/ws/voice/{userId}")
async def voice_chat(websocket: WebSocket,userId:str):
    await websocket.accept()
    detector = SilenceDetector(threshold=700, silence_duration=0.8)
    master_buffer = bytearray()

    is_processing=False
    interrupt_event = asyncio.Event()
    sentence_queue = asyncio.Queue()
    tts_task = asyncio.create_task(cartesia_tts_worker(sentence_queue, websocket, interrupt_event))
    active_ai_task=None
    was_speaking = False

    try:
        await websocket.send_json({"type": "status", "message": "✅ Connected & Ready"})
        print(f"✅ Voice client connected and handshake sent: {userId}")
        welcome_parts = ["Hello!", "I'm your assistant.", "How can I help you?"]
        for part in welcome_parts:
            await sentence_queue.put(part)
    except Exception:
        print(f"⚠️ Client {userId} disconnected during handshake. Skipping.")
        return    


    async def handle_full_brain_process(audio_bytes, userId, websocket, sentence_queue):
        nonlocal is_processing
        is_processing = True # Lock the brain
        
        try:
            # 1. Background STT 
            stt_result = await speech_to_text(audio_bytes)
            user_text = stt_result["transcript"]
            
            if not user_text:
                return

            # 2. Background RAG Response
            await run_ai_response(user_text, userId, websocket, sentence_queue)

        except Exception as e:
            print(f"🧠 Brain Error: {e}")
        finally:
            is_processing = False # Unlock for the next turn

    async def run_ai_response(user_text, userId, websocket, sentence_queue):
        sentence_buffer = ""
        full_ai_response = ""

        async def on_text_chunk(text: str):
            nonlocal sentence_buffer
            nonlocal full_ai_response

            sentence_buffer += text
            full_ai_response += text
            
            # 2. ⚡ EMERGENCY SPLIT: Latency Guard
            words = sentence_buffer.split()
            if len(words) > 12:
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
                clean_phrase = clean_text_for_tts(part)
                if clean_phrase:
                    await sentence_queue.put(clean_phrase)
            
            sentence_buffer = parts[-1]

        try:
            current_history = list(user_histories.get(userId, []))
            current_history.append({
                "role": "user",
                "content": user_text
            })
            user_histories[userId] = current_history
            result = await stream_rag_response(
                user_message=user_text,
                conversation_history=current_history,
                websocket=websocket,
                text_chunk_callback=on_text_chunk
            )

            # 3. If it finished naturally, save the full answer
            if sentence_buffer.strip():
                await sentence_queue.put(sentence_buffer.strip())
            
            if result:
                current_history.append({"role": "assistant", "content": full_ai_response})
                user_histories[userId] = current_history

        except asyncio.CancelledError:
            # ⚡ BARGE-IN DETECTED
            print(f"👋 AI Task interrupted. Saving partial response: {full_ai_response[:30]}...")
            
            # 4. Save the partial response so the AI knows where it left off
            partial_content = full_ai_response.strip() + "..." 
            current_history.append({"role": "assistant", "content": partial_content})
            user_histories[userId] = current_history
            
            raise



    try:
        while True:
            message = await websocket.receive()
        
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "mic_on":
                        print("🖱️ Mic Clicked: Forcing Backend Silence...")
                        interrupt_event.set()
                        
                        # 🧹 Clean up
                        if active_ai_task:
                            active_ai_task.cancel()
                            active_ai_task = None
                        
                        master_buffer.clear()
                        
                        while not sentence_queue.empty():
                            try:
                                sentence_queue.get_nowait()
                                sentence_queue.task_done()
                            except asyncio.QueueEmpty: break
                except Exception as e:
                    print(f"⚠️ Non-JSON text received: {e}")

            
            if "bytes" in message:
                chunk = message["bytes"]
                master_buffer.extend(chunk)

                # THE "INSTANT KILL" LOGIC
                if detector.has_spoken and not was_speaking:
                    print("🔊 USER SPOKE: KILLING EVERYTHING.")
                    
                    # A. Stop the Backend Mouth
                    interrupt_event.set()

                    # B. Tell the Frontend to stop playing current audio
                    await websocket.send_json({"type": "interrupt", "message": "🔇 Shutting up!"})

                    ai_was_thinking = (active_ai_task and not active_ai_task.done())
                    ai_has_more_to_say = not sentence_queue.empty()

                    if is_processing or ai_was_thinking or ai_has_more_to_say:
                        print("🧹 AI was busy. Wiping buffer and queue...")
                        if active_ai_task:
                            active_ai_task.cancel()
                            active_ai_task = None
                        
                        master_buffer.clear()
                        master_buffer.extend(chunk)

                        while not sentence_queue.empty():
                            try:
                                sentence_queue.get_nowait()
                                sentence_queue.task_done()
                            except asyncio.QueueEmpty: break

                was_speaking = detector.has_spoken

                # --- (Standard Rolling Window) ---
                if not detector.has_spoken:
                    if len(master_buffer) > 32000:
                        master_buffer = master_buffer[-32000:]

                # --- (Processing Trigger) ---
                if detector.is_user_finished(chunk) and not is_processing:
                    was_speaking = False # Reset for the next turn
                    await websocket.send_json({"type": "status", "message": "🤫 Processing..."})
                    audio_to_process = bytes(master_buffer)
                    master_buffer.clear()
                    active_ai_task = asyncio.create_task(
                        handle_full_brain_process(audio_to_process, userId, websocket, sentence_queue)
                    )

    except WebSocketDisconnect:
        print("❌ Voice client disconnected")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if active_ai_task and not active_ai_task.done():
            active_ai_task.cancel()
        await sentence_queue.put(None) 
        await tts_task
        print("🧹 Cleanup complete.")


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


async def cartesia_tts_worker(sentence_queue: asyncio.Queue, websocket, interrupt_event: asyncio.Event):
    ws = await client.tts.websocket()
    try:
        while True:
            sentence = await sentence_queue.get()
            
            if sentence is None:
                sentence_queue.task_done()
                break
                
            interrupt_event.clear()

            if not sentence.strip():
                sentence_queue.task_done()
                continue

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
                if interrupt_event.is_set():
                    print("🔇 TTS Worker: Interrupt received, killing current stream...")
                    break

                if output.audio is not None:
                    await websocket.send_bytes(output.audio)
                    
            sentence_queue.task_done()
            
    except Exception as e:
        print(f"⚠️ TTS Worker Error: {e}")
    finally:
        await ws.close()