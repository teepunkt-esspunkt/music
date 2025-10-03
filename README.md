# Music AI Experiments

WIP  
This repository contains experiments with combining TTS and Music Generation.  
It’s a work-in-progress playground for testing out AI-generated vocals and instrumentals.  
So far I tried **Bark** and **MusicGen**.

---

## 📂 Files

- `bark_only.py` → Simple script to generate raw vocals with Bark. CLI usage, pass lyrics as arguments.
- `bark_one_shot.py` → Config-based script to generate a single vocal take from text input.  
   Edit variables in the file and run directly. (uses VS Code `settings.json` for venv)
- `generate_sample.py` → MusicGen sample generator (instrumental only).
- `generate_song_min.py` → Minimal example combining Bark vocals + MusicGen instrumentals.

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/teepunkt-esspunkt/music.git
cd music
```

### 2. Create and activate a virtual environment (Python 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate # on Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. VS Code settings (optional but recommended)
Create a .vscode/settings.json file in the repo to always use the right virtual environment:
```bash
{
  "python.defaultInterpreterPath": "C:\\Users\\YOURNAME\\music\\venv\\Scripts\\python.exe"
}
```
(Set the path to your venv.)
This ensures imports (torch, audiocraft, bark, etc.) are recognized and you don’t have to re-select the interpreter every time.

## 📝 Example Usage

### Bark only
```bash
python bark_only.py --lyrics "Hello world, this is Bark singing"
```

### Bark with custom voice (edit config in script)
```bash
python bark_one_shot.py --lyrics "Riding through the night" --voice v2/en_speaker_6
```

### MusicGen only
```bash
python generate_sample.py -p "lofi hip hop with warm Rhodes and vinyl crackle" -d 8 -o outputs/sample.wav
```

### Combine vocals + instrumental
```bash
python generate_song_min.py --style "lofi hip hop with warm Rhodes and vinyl crackle" --lyrics "Drifting through the city lights, I move slow" -o outputs/my_song.wav
```

## ⚠️ Notes
- Bark is CPU/GPU heavy and may generate strange or noisy outputs — it’s still experimental.
- MusicGen works better for instrumentals; syncing vocals to beat is still naive.
- Outputs are stored in the `outputs/` folder.


