#!/bin/bash
set -e
trap 'echo "FAILED line $LINENO"' ERR
exec > >(tee -a /workspace/setup.log) 2>&1
echo "START $(date)"
cd /workspace

apt-get update && apt-get install -y zstd pciutils curl
/venv/main/bin/pip install omnivoice faster-whisper soundfile fastapi uvicorn \
    python-multipart "av<17" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /workspace/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -s localhost:11434 >/dev/null && break; sleep 2; done
ollama pull gemma4:26b

/venv/main/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('k2-fsa/OmniVoice')
snapshot_download('Systran/faster-whisper-large-v3')"

export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
nohup /venv/main/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --app-dir /workspace > /workspace/server.log 2>&1 &
for i in $(seq 1 30); do curl -s localhost:8000/health >/dev/null && break; sleep 2; done
curl -s localhost:8000/health

curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
nohup cloudflared tunnel --url http://localhost:8000 > /workspace/tunnel.log 2>&1 &
for i in $(seq 1 20); do grep -q trycloudflare /workspace/tunnel.log && break; sleep 2; done

echo "READY $(date)"
echo "TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /workspace/tunnel.log | head -1)"
