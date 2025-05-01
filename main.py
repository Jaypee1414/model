from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import numpy as np
import tempfile
import uuid
import os

app = FastAPI()

whisper_model = WhisperModel("tiny.en", device="cpu")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

@app.get("/")
def read_root():
    return {"message": "Whisper + TTS is running on Render"}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = b""
    while True:
        try:
            chunk = await websocket.receive_bytes()
            audio_data += chunk
        except Exception as e:
            print("WebSocket error:", e)
            break

        if len(audio_data) > 32000:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, np.frombuffer(audio_data, dtype=np.int16), 16000)
                transcription = transcribe_audio(temp_audio.name)
                tts_path = synthesize_tts(transcription)
                await websocket.send_json({
                    "transcription": transcription,
                    "tts_audio_url": f"/tts/{os.path.basename(tts_path)}"
                })
                audio_data = b""

@app.get("/tts/{filename}")
async def get_tts(filename: str):
    return FileResponse(f"tts_audio/{filename}")

def transcribe_audio(filepath: str) -> str:
    segments, _ = whisper_model.transcribe(filepath)
    return " ".join([seg.text for seg in segments])

def synthesize_tts(text: str) -> str:
    os.makedirs("tts_audio", exist_ok=True)
    out_path = f"tts_audio/{uuid.uuid4()}.wav"
    tts.tts_to_file(text=text, file_path=out_path)
    return out_path

# 🔽 This ensures the app binds correctly on Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
