# server.py

from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import asyncio
import time
import os

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

# === Setup ===

whisper_model = WhisperModel("tiny.en", device="cpu")

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")

reference_wav = "voice_ref.wav"  # <-- your cloned voice sample

sample_rate = 16000

# === Helper Functions ===

def save_audio_from_bytes(raw_audio_bytes, filename="recording.wav"):
    # Convert bytes buffer to numpy array
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
        print("▶️ Playing audio...\n")
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
            # Receive raw audio bytes sent by client
            data = await websocket.receive_bytes()
            print(f"🎤 Received audio buffer: {len(data)} bytes")

            # Process the received audio
            audio_file = save_audio_from_bytes(data)
            transcribed_text = transcribe_audio(audio_file)

            if transcribed_text.strip():
                speak_text(transcribed_text)
            else:
                print("🤐 Nothing was spoken clearly.")

            # Optionally: send a reply back to client
            await websocket.send_text("✅ Audio processed")

            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
    finally:
        await websocket.close()
        print("🔌 WebSocket Disconnected!")

# === Main entry ===

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=port)

