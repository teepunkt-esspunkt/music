# Use from an activated venv in your StyleTTS2 repo or in this repo
python -m pip install --upgrade pip wheel setuptools
# Install CUDA 12.1 Torch for RTX 40xx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
