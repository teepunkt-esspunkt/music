# --- CONFIG YOU CAN COMMIT (safe) ---
TEXT = "This is my first StyleTTS2 test."
REF_AUDIO = ""          # optional override; else uses config.default_ref_audio
OUT_DIR = ""            # optional override; else uses config.out_dir
OUT_NAME = "take1.wav"  # "" → auto timestamp

# --- CODE (don’t edit below) ---
import os, sys, datetime, yaml
from pathlib import Path

try:
    from styletts2 import tts as styletts_api
except ImportError:
    print("Missing styletts2 package → run: pip install styletts2")
    sys.exit(1)

CFG_PATH = "config.yaml"
cfg = {}
if os.path.isfile(CFG_PATH):
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

CHECKPOINT = cfg.get("checkpoint") or os.getenv("STYLETTS2_CHECKPOINT")
CONFIG     = cfg.get("config") or os.getenv("STYLETTS2_CONFIG")
DEFAULT_REF = cfg.get("default_ref_audio", "")
DEFAULT_OUT_DIR = cfg.get("out_dir", "outputs")

if not (CHECKPOINT and CONFIG):
    print("Missing checkpoint/config in config.yaml (or env vars).")
    sys.exit(1)

out_dir = OUT_DIR or DEFAULT_OUT_DIR
os.makedirs(out_dir, exist_ok=True)
if OUT_NAME:
    out_path = os.path.join(out_dir, OUT_NAME)
else:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"styletts2_{stamp}.wav")

ref_audio = REF_AUDIO or DEFAULT_REF or None

# Create the TTS engine
engine = styletts_api.StyleTTS2(
    model_checkpoint_path=CHECKPOINT,
    config_path=CONFIG
)

# Run inference
wav = engine.inference(
    text=TEXT,
    target_voice_path=ref_audio,
    output_wav_file=out_path,
    output_sample_rate=24000,
    alpha=0.3,
    beta=0.7,
    diffusion_steps=5,
    embedding_scale=1,
)

print("[StyleTTS2] done →", out_path)
