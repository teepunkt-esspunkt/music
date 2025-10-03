# --- CONFIG YOU CAN COMMIT (safe) ---
TEXT = "naw, it aint all playing, but i mean every word, do you know what im saying?"
REF_AUDIO = ""          # optional override; else uses config.default_ref_audio
OUT_DIR = ""            # optional override; else uses config.out_dir
OUT_NAME = "take1.wav"  # "" → auto timestamp

# --- PRESETS (choose 1–4) ---
PRESETS = {
    1: dict(ALPHA=0.35, BETA=0.85, DIFF_STEPS=10, EMB_SCALE=1),  # narration
    2: dict(ALPHA=0.30, BETA=0.70, DIFF_STEPS=12, EMB_SCALE=1),  # conversational
    3: dict(ALPHA=0.25, BETA=0.55, DIFF_STEPS=16, EMB_SCALE=2),  # rap/read
    4: dict(ALPHA=0.20, BETA=0.45, DIFF_STEPS=18, EMB_SCALE=2),  # sing-hint
}

PRESET = 4   # <<< pick one here (1–4)

# --- APPLY PRESET ---
p = PRESETS[PRESET]
ALPHA, BETA, DIFF_STEPS, EMB_SCALE = p["ALPHA"], p["BETA"], p["DIFF_STEPS"], p["EMB_SCALE"]
SR = 24000

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
    target_voice_path=ref_audio or None,
    output_wav_file=out_path,
    output_sample_rate=SR,
    alpha=ALPHA,
    beta=BETA,
    diffusion_steps=int(DIFF_STEPS),
    embedding_scale=int(EMB_SCALE),
)

print(f"[StyleTTS2] preset {PRESET} done →", out_path)
