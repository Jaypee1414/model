import sounddevice as sd
import scipy.io.wavfile as wav

duration = 30  # seconds
fs = 16000  # 16kHz

print("🎤 Recording voice reference (speak now)...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
wav.write("voice_ref.wav", fs, recording)
print("✅ Saved as voice_ref.wav")