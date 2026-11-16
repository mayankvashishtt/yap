"""Call room server. Everyone joins from a browser; the Mac does all the listening, translating and speaking.

PYTHONPATH=. .venv/bin/python -m yap.server        then open http://localhost:8770

Browser -> server : JSON {"type":"join", name, lang, voice: base64 PCM16 16 kHz | demo: "priya"}
                    then binary frames: PCM16, 16 kHz mono, 512 samples
Server -> browser : {"type":"roster"}, {"type":"speaking"}, {"type":"live"} (same-language pass-through),
                    {"type":"caption"} (what was heard + translation + timings), {"type":"audio"} (translated voice)
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import queue
import threading
import time

import librosa
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = 512                 # 32 ms at 16 kHz, what Silero VAD wants
PREROLL = 10                # frames kept from before speech starts (~320 ms)
MAX_UTT = 15 * 16000        # cut very long turns so the others aren't left waiting

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
ids = itertools.count(1)
people: dict = {}           # pid -> Person
jobs: "queue.Queue" = queue.Queue()
LOOP: asyncio.AbstractEventLoop = None
ENGINE = None
READY = threading.Event()


class Person:
    def __init__(self, pid, ws, name, lang):
        from silero_vad import VADIterator, load_silero_vad
        self.pid, self.ws, self.name, self.lang = pid, ws, name, lang
        self.vad = VADIterator(load_silero_vad(), sampling_rate=16000, threshold=0.5,
                               min_silence_duration_ms=550, speech_pad_ms=120)
        self.pre, self.buf, self.speaking = [], [], False

    def info(self):
        return {"id": self.pid, "name": self.name, "lang": self.lang}


def b64pcm(x: np.ndarray) -> str:
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


async def send(p: Person, msg: dict):
    try:
        await p.ws.send_text(json.dumps(msg, ensure_ascii=False))
    except Exception:
        pass


async def broadcast(msg, who=None):
    for p in list(people.values()):
        if who is None or who(p):
            await send(p, msg)


def post(coro):
    asyncio.run_coroutine_threadsafe(coro, LOOP)


# ---------------------------------------------------------------- worker: all model calls happen here
def worker():
    global ENGINE
    from .engine import Engine
    ENGINE = Engine()
    for name in ("priya", "mike"):
        path = os.path.join(ROOT, "voices", f"{name}.wav")
        if os.path.exists(path):
            ENGINE.enroll(f"demo:{name}", librosa.load(path, sr=16000)[0])
    READY.set()
    post(broadcast({"type": "status", "ready": True}))
    while True:
        job = jobs.get()
        if job[0] == "enroll":
            _, pid, wav = job
            ENGINE.enroll(pid, wav)
            continue
        _, pid, wav, t_end = job
        p = people.get(pid)
        if p is None or len(wav) < 16000 * 0.4:
            continue
        targets = sorted({q.lang for q in people.values() if q.pid != pid and q.lang != p.lang})
        voice = getattr(p, "voice_id", pid)
        cid = f"c{next(ids)}"
        ms, first = {}, {}
        for ev in ENGINE.interpret_stream(wav, voice, p.lang, targets):
            if ev[0] == "heard":
                _, heard, locked, m = ev
                ms.update(m)
                if not heard:
                    break
                post(broadcast({"type": "caption", "cid": cid, "speaker": p.name, "pid": pid, "lang": p.lang,
                                "heard": heard, "locked": locked}))
            elif ev[0] == "text":
                _, tgt, text, m = ev
                ms.update(m)
                post(broadcast({"type": "caption_tr", "cid": cid, "lang": tgt, "text": text}))
            elif ev[0] == "audio":
                _, tgt, chunk, last = ev
                lat = round((time.time() - t_end) * 1000)
                if tgt not in first:
                    first[tgt] = lat
                    post(broadcast({"type": "timing", "cid": cid, "lang": tgt, "first_audio_ms": lat,
                                    "ms": {k: round(v) for k, v in ms.items()}}))
                post(broadcast({"type": "audio", "cid": cid, "speaker": p.name, "pid": pid, "sr": 24000, "lang": tgt,
                                "latency_ms": lat, "last": last, "pcm": b64pcm(chunk)},
                               who=lambda q, tgt=tgt: q.lang == tgt and q.pid != pid))


# ---------------------------------------------------------------- web
@app.on_event("startup")
async def startup():
    global LOOP
    LOOP = asyncio.get_running_loop()
    threading.Thread(target=worker, daemon=True).start()


@app.get("/")
async def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    p = None
    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("text"):
                msg = json.loads(m["text"])
                if msg["type"] == "join":
                    pid = f"p{next(ids)}"
                    p = Person(pid, ws, msg["name"][:24] or pid, msg["lang"] if msg["lang"] in ("hi", "en") else "en")
                    if msg.get("demo") in ("priya", "mike"):
                        p.voice_id = f"demo:{msg['demo']}"
                    elif msg.get("voice"):
                        wav = np.frombuffer(base64.b64decode(msg["voice"]), "<i2").astype(np.float32) / 32768
                        jobs.put(("enroll", pid, wav))
                    people[pid] = p
                    await send(p, {"type": "joined", "id": pid, "ready": READY.is_set()})
                    await broadcast({"type": "roster", "people": [q.info() for q in people.values()]})
                elif msg["type"] == "mute" and p:
                    p.vad.reset_states(); p.buf, p.speaking = [], False
            elif m.get("bytes") is not None and p is not None:
                await on_audio(p, m["bytes"])
    except WebSocketDisconnect:
        pass
    finally:
        if p:
            people.pop(p.pid, None)
            await broadcast({"type": "roster", "people": [q.info() for q in people.values()]})


async def on_audio(p: Person, data: bytes):
    x = np.frombuffer(data, "<i2").astype(np.float32) / 32768
    for i in range(0, len(x) - FRAME + 1, FRAME):
        f = x[i:i + FRAME]
        ev = p.vad(torch.from_numpy(f), return_seconds=False)
        if p.speaking:
            p.buf.append(f)
        else:
            p.pre = (p.pre + [f])[-PREROLL:]
        if ev and "start" in ev and not p.speaking:
            p.speaking, p.buf = True, list(p.pre)
            await broadcast({"type": "speaking", "pid": p.pid, "on": True})
        # same-language listeners hear the speaker directly
        if p.speaking:
            await broadcast({"type": "live", "pid": p.pid, "pcm": b64pcm(f)},
                            who=lambda q: q.lang == p.lang and q.pid != p.pid)
        long_turn = p.speaking and len(p.buf) * FRAME >= MAX_UTT
        if (ev and "end" in ev and p.speaking) or long_turn:
            wav = np.concatenate(p.buf)
            p.speaking, p.buf = False, []
            if long_turn:
                p.vad.reset_states()
            jobs.put(("utt", p.pid, wav, time.time()))
            await broadcast({"type": "speaking", "pid": p.pid, "on": False})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8770)), log_level="warning")
