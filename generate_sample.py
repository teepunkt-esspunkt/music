import argparse, os, torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def load_model():
    model_name = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MusicGen.get_pretrained(model_name, device=device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", required=True, help="Text prompt")
    ap.add_argument("-d", "--duration", type=int, default=10, help="Seconds")
    ap.add_argument("-o", "--out", default="outputs/sample.wav", help="Output .wav")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # manual seeding for this Audiocraft version
    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = load_model()

    # keep it simple & compatible
    model.set_generation_params(
        duration=args.duration,
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        cfg_coef=3.0,
    )

    wav = model.generate([args.prompt], progress=True)[0]
    audio_write(args.out, wav, model.sample_rate, format="wav", strategy="loudness")
    print(f"Saved → {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
