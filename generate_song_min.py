import os, argparse, torch, math
import torchaudio
from audiocraft.models import MusicGen

# ---- Bark (lyrics -> sung vocal) ----
try:
    from bark import generate_audio, preload_models, SAMPLE_RATE  # SAMPLE_RATE = 24000
except ImportError:
    from bark.api import generate_audio, preload_models, SAMPLE_RATE

def gen_bark_vocal(lyrics: str, voice_preset: str = "v2/en_speaker_6"):
    preload_models()  # downloads Bark models on first run
    # Hint Bark to sing: add [music] tags + punctuation/line breaks
    prompt = f"[music] {lyrics.strip()} [/music]"
    wav_24k = generate_audio(prompt, history_prompt=voice_preset)  # np.ndarray, mono
    wav_24k = torch.tensor(wav_24k, dtype=torch.float32).unsqueeze(0)  # (1, T)
    return wav_24k, 24000

# ---- MusicGen (style -> instrumental) ----
def gen_musicgen_instrumental(style_prompt: str, seconds: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
    model.set_generation_params(
        duration=seconds,
        use_sampling=True, top_k=250, top_p=0.0, temperature=1.0,
        cfg_coef=3.0, two_step_cfg=True
    )
    wav_32k = model.generate([style_prompt], progress=True)[0]  # (1, T), 32k
    return wav_32k, 32000  # ← IMPORTANT: 32k, not Bark’s 24k

def resample(wav: torch.Tensor, sr_in: int, sr_out: int):
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out)

def pad_or_trim(wav: torch.Tensor, target_len: int):
    cur = wav.shape[-1]
    if cur == target_len:
        return wav
    if cur > target_len:
        return wav[..., :target_len]
    return torch.nn.functional.pad(wav, (0, target_len - cur))

def rms_normalize(wav: torch.Tensor, target_rms: float):
    rms = torch.sqrt(torch.mean(wav**2) + 1e-8)
    return wav * (target_rms / max(rms, 1e-8))

def mix_to_stereo(vocal_32k: torch.Tensor, instr_32k: torch.Tensor,
                  vocal_gain_db: float = 0.0, instr_gain_db: float = -2.0):
    T = max(vocal_32k.shape[-1], instr_32k.shape[-1])
    v = pad_or_trim(vocal_32k, T)
    i = pad_or_trim(instr_32k, T)

    v = rms_normalize(v, 0.08)
    i = rms_normalize(i, 0.10)

    v = v * (10 ** (vocal_gain_db / 20.0))
    i = i * (10 ** (instr_gain_db / 20.0))

    mix = v + i  # mono mix
    left  = mix * 0.95
    right = mix * 1.05
    stereo = torch.stack([left.squeeze(0), right.squeeze(0)], dim=0)  # (2, T)
    return torch.clamp(stereo, -0.99, 0.99)

def save_wav_torch(path: str, wav: torch.Tensor, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, wav.cpu(), sr)
    print(f"Saved → {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, help="Style prompt for instrumental")
    ap.add_argument("--lyrics", required=True, help="Lyrics to be sung")
    ap.add_argument("-o", "--out", default="outputs/song.wav", help="Output WAV path")
    ap.add_argument("-v", "--voice", default="v2/en_speaker_6", help="Bark voice preset")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Bark vocal @24k -> resample to 32k
    vocal_24k, sr_v = gen_bark_vocal(args.lyrics, args.voice)
    vocal_32k = resample(vocal_24k, sr_v, 32000).to(device)


    # 2) MusicGen instrumental with matching duration
    seconds = int(math.ceil(vocal_32k.shape[-1] / 32000.0))
    instr_32k, sr_i = gen_musicgen_instrumental(args.style, seconds)
    instr_32k = instr_32k.to(device)


    # 3) Mix & export (32k stereo)
    mix_32k = mix_to_stereo(vocal_32k, instr_32k, vocal_gain_db=1.5, instr_gain_db=-2.0).to("cpu")
    save_wav_torch(args.out, mix_32k, 32000)



if __name__ == "__main__":
    main()
