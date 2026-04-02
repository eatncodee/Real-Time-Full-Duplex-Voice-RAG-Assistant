import io,wave
import re
import httpx
import numpy as np
import asyncio
import time






class SilenceDetector:
    def __init__(self, threshold=700, silence_duration=0.4):
        self.threshold = threshold          # Min volume to count as "speech"
        self.silence_duration = silence_duration  # Seconds of silence to trigger stop
        self.silence_start_time = None
        self.has_spoken = False             # Ensures we don't trigger on initial silence
    def is_user_finished(self, audio_chunk_bytes):
        audio_data = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
        if len(audio_data) == 0:
            return False
        # 2. Calculate Volume (RMS)
        rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        # 3. Decision Logic
        if rms > self.threshold:
            self.has_spoken = True
            self.silence_start_time = None  # Reset timer
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
