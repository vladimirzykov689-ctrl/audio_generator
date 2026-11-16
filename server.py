import os, json, time, gc, uuid, hashlib, zipfile, threading, requests, torch, soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

OLLAMA = "http://127.0.0.1:11434"
MODEL = "gemma4:26b"
SR = 24000
WPS = 1.4                      # слов в секунду для вашего голоса
ROOT = "/workspace/jobs"; REF = "/workspace/refs"
os.makedirs(ROOT, exist_ok=True); os.makedirs(REF, exist_ok=True)

SYSTEM = ("Ты пишешь текст для озвучки на русском. Только устная речь: короткие фразы, "
          "без markdown, списков, эмодзи и скобок. Числа и даты пиши словами. "
          "Верни только текст озвучки, без пояснений.")

app = FastAPI()
JOBS = {}
GPU_LOCK = threading.Lock()    # одна пачка за раз

class Item(BaseModel):
    id: str
    prompt: Optional[str] = None
    text: Optional[str] = None          # если задан, LLM пропускается
    seconds: float = 20
    voice_url: str                      # http(s) URL или путь на диске
    ref_text: Optional[str] = None
    language: str = "ru"

class Batch(BaseModel):
    jobs: List[Item]

def get_voice(src):
    if not src.startswith("http"): return src
    p = f"{REF}/{hashlib.md5(src.encode()).hexdigest()}.wav"
    if not os.path.exists(p):
        open(p, "wb").write(requests.get(src, timeout=60).content)
    return p

def build_subtitles(words, max_words_line=3, max_lines=3, max_dur=3.5, gap=0.5):
    clean = lambda t: "".join(c for c in t.lower() if c.isalnum() or c in "-'")
    groups, cur = [], []
    for x in words:
        t = clean(x.word.strip())
        if not t: continue
        if cur and (x.start - cur[-1][2] > gap or x.end - cur[0][1] > max_dur
                    or len(cur) >= max_words_line * max_lines):
            groups.append(cur); cur = []
        cur.append((t, x.start, x.end))
    if cur: groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        lines = [" ".join(t for t, _, _ in g[j:j+max_words_line])
                 for j in range(0, len(g), max_words_line)]
        start = 0.0 if i == 0 else g[0][1]
        end = groups[i+1][0][1] if i+1 < len(groups) else g[-1][2]
        out.append({"start": round(start, 2), "end": round(end, 2),
                    "duration": round(end - start, 2), "text": "\n".join(lines)})
    return out

def gen_text(it):
    n = round(it.seconds * WPS)
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL, "stream": False, "think": False, "keep_alive": "10m",
        "options": {"num_ctx": 4096},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"{it.prompt}\n\nОбъём текста: около {n} слов."}]},
        timeout=300)
    return r.json()["message"]["content"].strip()

def run_batch(job_id, items):
    st = JOBS[job_id]; d = f"{ROOT}/{job_id}"; os.makedirs(d, exist_ok=True)
    with GPU_LOCK:
        try:
            # --- этап 1: тексты ---
            st["status"] = "texts"
            for it in items:
                try:
                    if not it.text: it.text = gen_text(it)
                    st["items"][it.id] = {"status": "text_ready"}
                except Exception as e:
                    st["items"][it.id] = {"status": "error", "error": f"text: {e}"}
            requests.post(f"{OLLAMA}/api/generate", json={"model": MODEL, "keep_alive": 0})
            time.sleep(3); gc.collect(); torch.cuda.empty_cache()

            # --- этапы 2-3: озвучка и субтитры ---
            st["status"] = "voice"
            from omnivoice import OmniVoice
            from faster_whisper import WhisperModel
            tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
            asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
            for it in items:
                if st["items"][it.id]["status"] == "error": continue
                try:
                    kw = {"text": it.text, "ref_audio": get_voice(it.voice_url)}
                    if it.ref_text: kw["ref_text"] = it.ref_text
                    wav = tts.generate(**kw)[0].squeeze()
                    wp = f"{d}/{it.id}.wav"; sf.write(wp, wav, SR)
                    segs, _ = asr.transcribe(wp, language=it.language, word_timestamps=True)
                    subs = build_subtitles([x for s in segs for x in s.words])
                    json.dump(subs, open(f"{d}/{it.id}.json", "w"), ensure_ascii=False, indent=2)
                    st["items"][it.id] = {"status": "done", "duration": round(wav.size / SR, 1),
                                          "text": it.text}
                except Exception as e:
                    st["items"][it.id] = {"status": "error", "error": str(e)}
            del tts, asr; gc.collect(); torch.cuda.empty_cache()

            json.dump({i.id: i.text for i in items}, open(f"{d}/texts.json", "w"),
                      ensure_ascii=False, indent=2)
            with zipfile.ZipFile(f"{d}/result.zip", "w") as z:
                for f in os.listdir(d):
                    if f != "result.zip": z.write(f"{d}/{f}", f)
            st["status"] = "done"
        except Exception as e:
            st["status"] = "error"; st["error"] = str(e)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/batch")
def batch(b: Batch):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "queued", "items": {}}
    threading.Thread(target=run_batch, args=(job_id, b.jobs), daemon=True).start()
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS: raise HTTPException(404)
    return JOBS[job_id]

@app.get("/result/{job_id}")
def result(job_id: str):
    p = f"{ROOT}/{job_id}/result.zip"
    if not os.path.exists(p): raise HTTPException(404, "ещё не готово")
    return FileResponse(p, filename=f"{job_id}.zip")
