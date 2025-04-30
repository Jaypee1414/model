from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from TTS.api import TTS
import numpy as np
import soundfile as sf
import os
import uuid
import wave

app = FastAPI()

# Load models
whisper_model = WhisperModel("tiny.en", device="cpu")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/download-audio/{filename}")
def download_audio(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav", filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

# --- 🧠 WebSocket for Streaming Audio ---
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()

    audio_id = str(uuid.uuid4())
    raw_audio_path = os.path.join(UPLOAD_DIR, f"{audio_id}.raw")
    wav_audio_path = os.path.join(UPLOAD_DIR, f"{audio_id}.wav")
    out_audio_path = os.path.join(OUTPUT_DIR, f"{audio_id}_tts.wav")

    buffer = bytearray()
    sample_rate = 16000
    sample_width = 2  # 16-bit PCM
    channels = 1

    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)

    except WebSocketDisconnect:
        # Save raw to WAV
        with wave.open(wav_audio_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(buffer)

        # Transcribe
        segments, _ = whisper_model.transcribe(wav_audio_path)
        transcription = " ".join([seg.text for seg in segments])

        if not transcription.strip():
            await websocket.send_json({"error": "No speech detected."})
            return

        # Generate TTS
        tts_model.tts_to_file(text=transcription, file_path=out_audio_path)

        await websocket.send_json({
            "transcription": transcription,
            "tts_audio_url": f"/download-audio/{os.path.basename(out_audio_path)}"
        })
