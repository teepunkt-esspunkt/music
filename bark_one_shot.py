# --- CONFIG (edit these) ---
VOICE = "v2/en_speaker_6"
TEXT_TEMP = 0.80
WAVE_TEMP = 0.95
LYRICS = (
    " [cheeky] do you know what i am saying? "

)
OUT_DIR = "outputs"           # folder created if missing
OUT_NAME = "bark_take5.wav"   # change per run if you want
# ---------------------------

import os
import sys
import torch
import torchaudio

# Bark import (supports both layouts)
try:
    from bark import generate_audio, preload_models, SAMPLE_RATE  # new-ish
except ImportError:
    from bark.api import generate_audio, preload_models, SAMPLE_RATE  # older

def main():
    # Ensure UTF-8 (Windows consoles can be quirky with ♪)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)

    print("➡️  Preloading Bark models (first run will download ~GBs)…")
    preload_models()

    # Light cleanup of the prompt (helps avoid odd whitespace artifacts)
    prompt = " ".join(LYRICS.split()).strip()
    if not prompt:
        raise ValueError("LYRICS is empty after stripping.")

    print("🎤  Generating vocal with Bark…")
    audio_np = generate_audio(
        prompt,
        history_prompt=VOICE,
        text_temp=float(TEXT_TEMP),
        waveform_temp=float(WAVE_TEMP),
    )  # numpy float32, mono

    if audio_np is None or len(audio_np) == 0:
        raise RuntimeError("Bark returned empty audio. Try adjusting the prompt/temps.")

    # Convert to torch tensor and save at Bark’s native 24 kHz
    wav = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)  # (1, T)

    # Optional: very gentle peak clamp to avoid accidental clipping
    wav = torch.clamp(wav, -0.99, 0.99)

    torchaudio.save(out_path, wav.cpu(), int(SAMPLE_RATE))
    dur = wav.shape[-1] / float(SAMPLE_RATE)
    print(f"✅  Saved → {out_path}  ({dur:.2f}s @ {SAMPLE_RATE} Hz)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
