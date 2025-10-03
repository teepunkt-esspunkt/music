# --- CONFIG (edit these) ---
VOICE = "v2/de_speaker_3"
TEXT_TEMP = 0.80
WAVE_TEMP = 0.95
LYRICS = " ♪ Alita, Tamara, Da bibi, Kunawa ♪ \n [singing] alita tamara, da bibi kunawa "
OUT_DIR_NAME = "outputs"      # just the folder name
OUT_NAME = "bark_take7.wav"
# ---------------------------

import os, sys, torch, torchaudio
from pathlib import Path

# Bark import (supports both layouts)
try:
    from bark import generate_audio, preload_models, SAMPLE_RATE
except ImportError:
    from bark.api import generate_audio, preload_models, SAMPLE_RATE  # type: ignore

# Make all relative paths resolve from THIS file's folder
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / OUT_DIR_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / OUT_NAME

def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes; ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    print(f"➡️  Writing to: {OUT_PATH}")
    print("➡️  Preloading Bark models (first run will download)…")
    preload_models()

    prompt = " ".join(LYRICS.split()).strip()
    if not prompt:
        raise ValueError("LYRICS is empty after stripping.")

    print("🎤  Generating vocal with Bark…")
    audio_np = generate_audio(prompt, history_prompt=VOICE,
                              text_temp=float(TEXT_TEMP),
                              waveform_temp=float(WAVE_TEMP))

    if audio_np is None or len(audio_np) == 0:
        raise RuntimeError("Bark returned empty audio. Try adjusting the prompt/temps.")

    wav = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    wav = torch.clamp(wav, -0.99, 0.99)

    torchaudio.save(str(OUT_PATH), wav.cpu(), int(SAMPLE_RATE)) # type: ignore
    dur = wav.shape[-1] / float(SAMPLE_RATE)
    print(f"✅  Saved → {OUT_PATH}  ({dur:.2f}s @ {SAMPLE_RATE} Hz)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
