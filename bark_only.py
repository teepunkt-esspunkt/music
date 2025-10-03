import argparse, torch, os
import numpy as np
import soundfile as sf

# Bark imports (works with either API layout)
try:
    from bark import generate_audio, preload_models, SAMPLE_RATE
except ImportError:
    from bark.api import generate_audio, preload_models, SAMPLE_RATE # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics", required=True, help="What to sing/rap")
    ap.add_argument("--voice", default="v2/en_speaker_6", help="Bark voice preset")
    ap.add_argument("--out", default="outputs/bark_vocal.wav", help="Output wav")
    ap.add_argument("--text_temp", type=float, default=0.7, help="Lower=clearer, Higher=creative")
    ap.add_argument("--wave_temp", type=float, default=0.7, help="Lower=stable tone, Higher=expressive")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preload_models()

    # Nudge Bark into singing/flow: musical tags + line breaks
    prompt = f"[music]\n{args.lyrics.strip()}\n[/music]"

    audio = generate_audio(
        prompt,
        history_prompt=args.voice,
        text_temp=args.text_temp,
        waveform_temp=args.wave_temp,
    )

    # Bark is 24 kHz mono float32
    sf.write(args.out, audio, SAMPLE_RATE)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
