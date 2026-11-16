#!/bin/bash
set -e
exec > >(tee -a /workspace/setup.log) 2>&1
echo "START $(date)"
mkdir -p /workspace && cd /workspace

apt-get update && apt-get install -y zstd pciutils curl
PIP=/venv/main/bin/pip
$PIP install omnivoice faster-whisper soundfile fastapi uvicorn "av<17" \
    nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 5
ollama pull gemma4:26b

# предзагрузка весов, чтобы первая пачка не ждала
/venv/main/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('k2-fsa/OmniVoice')
snapshot_download('Systran/faster-whisper-large-v3')"

curl -fsSL https://raw.githubusercontent.com/vladimirzykov689-ctrl/audio_generator/main/server.py -o /workspace/server.py

export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
nohup /venv/main/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --app-dir /workspace > /workspace/server.log 2>&1 &
sleep 3
echo "READY $(date)"
