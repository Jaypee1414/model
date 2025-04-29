# server.py

import os
import asyncio
import numpy as np
import soundfile as sf
import sounddevice as sd
import scipy.io.wavfile as wav

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

from faster_whisper import WhisperModel
from TTS.api import TTS

# === Setup ===

sample_rate = 16000
reference_wav = "voice_ref.wav"  # Path to your cloned voice reference

# Load models (slow on cold start)
whisper_model = WhisperModel("tiny.en", device="cpu")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")

# === Helper Functions ===

def save_audio_from_bytes(raw_audio_bytes, filename="recording.wav"):
    audio_np = np.frombuffer(raw_audio_bytes, dtype=np.int16)
    wav.write(filename, sample_rate, audio_np)
    return filename

def transcribe_audio(filename):
    print("🧠 Transcribing...")
    try:
        segments, _ = whisper_model.transcribe(filename)
        text = " ".join([segment.text for segment in segments])
        print("📝 Transcribed:", text)
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def speak_text(text, output_file="tts_output.wav"):
    print("🧬 Generating cloned voice...")
    try:
        tts.tts_to_file(text=text, file_path=output_file, speaker_wav=reference_wav, language="en")
        data, samplerate = sf.read(output_file)
        print("▶️ Playing audio...")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"❌ TTS error: {e}")

# === FastAPI App ===

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket Connected!")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"🎤 Received {len(data)} bytes of audio")

            audio_file = save_audio_from_bytes(data)
            transcribed_text = transcribe_audio(audio_file)

            if transcribed_text.strip():
                speak_text(transcribed_text)
            else:
                print("🤐 No clear speech detected.")

            await websocket.send_text("✅ Audio processed")
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        await websocket.close()
        print("🔌 WebSocket Disconnected.")

# === Entry Point ===

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on 0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port)
