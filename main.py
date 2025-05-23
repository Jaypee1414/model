# from fastapi import FastAPI, WebSocket, HTTPException
# from fastapi.responses import FileResponse
# from faster_whisper import WhisperModel
# from TTS.api import TTS
# import soundfile as sf
# import numpy as np
# import uvicorn
# import tempfile
# import uuid
# import os

# app = FastAPI()

# # Global model variables
# whisper_model = None
# tts = None

# # === SYNC MODEL LOADING DURING STARTUP ===
# @app.on_event("startup")
# async def load_models():
#     global whisper_model, tts
#     print("Loading Whisper and TTS models...")
#     whisper_model = WhisperModel("tiny.en", device="cpu")  # lightweight
#     tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # could be swapped if too slow
#     print("Models loaded successfully.")

# # === HEALTH CHECK ===
# @app.get("/")
# def read_root():
#     if whisper_model is None or tts is None:
#         return {"status": "starting", "message": "Models are still loading..."}
#     return {"status": "ready", "message": "Whisper + TTS is ready."}

# @app.get("/health")
# def health_check():
#     return {"ok": whisper_model is not None and tts is not None}

# # === WEBSOCKET ENDPOINT ===
# @app.websocket("/ws/audio")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     if whisper_model is None or tts is None:
#         await websocket.send_text("Models are still loading, please wait.")
#         await websocket.close()
#         return

#     audio_data = b""
#     while True:
#         try:
#             chunk = await websocket.receive_bytes()
#             audio_data += chunk
#         except Exception as e:
#             print("WebSocket error:", e)
#             break

#         if len(audio_data) > 32000:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
#                 sf.write(temp_audio.name, np.frombuffer(audio_data, dtype=np.int16), 16000)
#                 transcription = transcribe_audio(temp_audio.name)
#                 tts_path = synthesize_tts(transcription)
#                 await websocket.send_json({
#                     "transcription": transcription,
#                     "tts_audio_url": f"/tts/{os.path.basename(tts_path)}"
#                 })
#                 audio_data = b""

# # === AUDIO FILE RETURN ===
# @app.get("/tts/{filename}")
# async def get_tts(filename: str):
#     file_path = f"tts_audio/{filename}"
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(file_path)

# # === WHISPER TRANSCRIPTION ===
# def transcribe_audio(filepath: str) -> str:
#     segments, _ = whisper_model.transcribe(filepath)
#     return " ".join([seg.text for seg in segments])

# # === TTS SYNTHESIS ===
# def synthesize_tts(text: str) -> str:
#     os.makedirs("tts_audio", exist_ok=True)
#     out_path = f"tts_audio/{uuid.uuid4()}.wav"
#     tts.tts_to_file(text=text, file_path=out_path)
#     return out_path

# # === ENTRY POINT ===
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import numpy as np
import tempfile
import uuid
import os

app = FastAPI()

# Global model variables
whisper_model = None
tts = None

# === SYNC MODEL LOADING DURING STARTUP ===
@app.on_event("startup")
async def load_models():
    global whisper_model, tts
    print("Loading Whisper and TTS models...")
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")  # optimized for low memory
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    print("Models loaded successfully.")

# === HEALTH CHECK ===
@app.get("/")
def read_root():
    if whisper_model is None or tts is None:
        return {"status": "starting", "message": "Models are still loading..."}
    return {"status": "ready", "message": "Whisper + TTS is ready."}

@app.get("/health")
def health_check():
    return {"ok": whisper_model is not None and tts is not None}

# === WEBSOCKET ENDPOINT ===
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if whisper_model is None or tts is None:
        await websocket.send_text("Models are still loading, please wait.")
        await websocket.close()
        return

    audio_data = b""
    while True:
        try:
            chunk = await websocket.receive_bytes()
            audio_data += chunk
        except Exception as e:
            print("WebSocket error:", e)
            break

        if len(audio_data) > 32000:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as temp_audio:
                    sf.write(temp_audio.name, np.frombuffer(audio_data, dtype=np.int16), 16000)
                    print("Audio file saved:", temp_audio.name)

                transcription = transcribe_audio(temp_audio.name)
                print("Transcription:", transcription)

                tts_path = synthesize_tts(transcription)
                print("TTS audio generated:", tts_path)

                await websocket.send_json({
                    "transcription": transcription,
                    "tts_audio_url": f"/tts/{os.path.basename(tts_path)}"
                })
            except Exception as e:
                print("Processing error:", e)
                await websocket.send_text("Error processing audio.")

            audio_data = b""

# === AUDIO FILE RETURN ===
@app.get("/tts/{filename}")
async def get_tts(filename: str):
    file_path = f"/tmp/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# === WHISPER TRANSCRIPTION ===
def transcribe_audio(filepath: str) -> str:
    segments, _ = whisper_model.transcribe(filepath)
    return " ".join([seg.text for seg in segments])

# === TTS SYNTHESIS ===
def synthesize_tts(text: str) -> str:
    out_path = f"/tmp/{uuid.uuid4()}.wav"
    tts.tts_to_file(text=text, file_path=out_path)
    return out_path
