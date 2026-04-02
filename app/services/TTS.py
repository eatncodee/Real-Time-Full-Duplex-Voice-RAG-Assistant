import io,wave
import re
import os
import asyncio
import time
from cartesia import AsyncCartesia


client=AsyncCartesia(api_key=os.getenv("Cartesia_key"),)


async def cartesia_tts_worker(sentence_queue: asyncio.Queue, websocket, interrupt_event: asyncio.Event):
    ws = await client.tts.websocket()
    try:
        while True:
            sentence = await sentence_queue.get()
            
            if sentence is None:
                sentence_queue.task_done()
                break
                
            interrupt_event.clear()

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