"""Two scripted people hold a standup over the real call server, through real WebSockets, in real time.

PYTHONPATH=. .venv/bin/python -m yap.bots [ws://127.0.0.1:8770/ws]

Priya speaks Hinglish, Mike speaks English. Each line is streamed like a mic (32 ms frames, real time,
then silence). We record what each one hears, measure the delay from 'stopped talking' to 'first
translated sound', and check the numbers survived. Output: runs/<time>/report.json + conversation.wav
"""
import asyncio
import base64
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = [
    ("priya", "hi", "गुड मॉर्निंग माइक! कल मैंने पेमेंट वाला बग फिक्स कर दिया।", None),
    ("mike", "en", "Nice! Did the refund for the Delhi customer go through?", None),
    ("priya", "hi", "हाँ, साढ़े तीन हज़ार रुपये का रिफंड चला गया।", "3,500|three thousand five hundred"),
    ("mike", "en", "Great. The client call is at a quarter to nine tomorrow, please don't be late.", None),
    ("priya", "hi", "ठीक है। और क्लाइंट का नंबर नोट करो, नौ आठ सात, डबल पाँच, चार तीन दो एक शून्य।", "9875543210|nine eight seven five five four three two one zero"),
    ("mike", "en", "Got it. Last thing, the invoice is one lakh fifty thousand rupees, send it by Friday.", "1,50,000|डेढ़ लाख|एक लाख पचास हज़ार"),
]
VOICE = {"priya": ("h", "hf_alpha"), "mike": ("a", "am_michael")}


def make_lines():
    from kokoro import KPipeline
    import librosa
    pipes = {}
    out = []
    for who, lang, text, check in SCRIPT:
        code, voice = VOICE[who]
        pipes.setdefault(code, KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M"))
        a = np.concatenate([np.asarray(x) for _, _, x in pipes[code](text, voice=voice)])
        out.append(librosa.resample(a, orig_sr=24000, target_sr=16000).astype(np.float32))
    return out


class Bot:
    def __init__(self, name, lang):
        self.name, self.lang, self.heard, self.caps, self.tr, self.timing = name, lang, [], [], {}, {}

    async def connect(self, url):
        self.ws = await websockets.connect(url, max_size=2**25)
        await self.ws.send(json.dumps({"type": "join", "name": self.name.title(), "lang": self.lang, "demo": self.name}))
        self.task = asyncio.create_task(self.listen())

    async def listen(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if m["type"] == "audio":
                pcm = np.frombuffer(base64.b64decode(m["pcm"]), "<i2").astype(np.float32) / 32768
                if self.heard and self.heard[-1]["cid"] == m["cid"]:
                    self.heard[-1]["pcm"] = np.concatenate([self.heard[-1]["pcm"], pcm])
                else:
                    self.heard.append({"t": time.time(), "cid": m["cid"], "from": m["speaker"], "pcm": pcm,
                                       "latency_ms": m["latency_ms"], "text": self.tr.get(m["cid"], "")})
            elif m["type"] == "caption":
                self.caps.append(m)
            elif m["type"] == "timing":
                self.timing[m["cid"]] = m
            elif m["type"] == "caption_tr" and m["lang"] == self.lang:
                self.tr[m["cid"]] = m["text"]
            elif m["type"] == "joined":
                self.ready = m["ready"]

    async def say(self, wav):
        for i in range(0, len(wav), 512):
            f = wav[i:i + 512]
            if len(f) < 512:
                f = np.pad(f, (0, 512 - len(f)))
            await self.ws.send((f * 32767).astype("<i2").tobytes())
            await asyncio.sleep(0.032)
        silence = np.zeros(512, dtype="<i2").tobytes()
        for _ in range(25):                     # 0.8 s of silence so the VAD closes the turn
            await self.ws.send(silence)
            await asyncio.sleep(0.032)


async def main(url):
    lines = make_lines()
    bots = {"priya": Bot("priya", "hi"), "mike": Bot("mike", "en")}
    for b in bots.values():
        await b.connect(url)
    await asyncio.sleep(1.0)
    report, convo = [], []
    for (who, lang, text, check), wav in zip(SCRIPT, lines):
        other = bots["mike" if who == "priya" else "priya"]
        n_before = len(other.heard)
        t_stop = time.time() + len(wav) / 16000
        await bots[who].say(wav)
        while len(other.heard) == n_before and time.time() - t_stop < 30:
            await asyncio.sleep(0.05)
        if len(other.heard) == n_before:
            report.append({"who": who, "said": text, "error": "nothing heard in 30 s"}); continue
        h = other.heard[-1]
        await asyncio.sleep(len(h["pcm"]) / 24000 + 0.5)          # let the rest of the reply stream in
        cap = next((c for c in other.caps if c["cid"] == h["cid"]), {})
        ok = None
        if check:
            ok = any(k.lower() in (h["text"] + " " + cap.get("locked", "")).lower() for k in check.split("|"))
        report.append({"who": who, "said": text, "heard_as": cap.get("heard"), "locked": cap.get("locked"),
                       "other_hears": h["text"], "delay_s": round(h["t"] - t_stop, 2),
                       "server_latency_s": h["latency_ms"] / 1000, "steps_ms": other.timing.get(h["cid"], {}).get("ms"), "numbers_ok": ok})
        convo += [wav, np.zeros(4000, np.float32),
                  np.interp(np.arange(0, len(h["pcm"]), 1.5), np.arange(len(h["pcm"])), h["pcm"]).astype(np.float32),
                  np.zeros(8000, np.float32)]
        print(f"{who:>5}: {text}\n   -> {h['text']}   ({report[-1]['delay_s']} s"
              + ("" if ok is None else f", numbers {'OK' if ok else 'WRONG'}") + ")", flush=True)
        await asyncio.sleep(1.5)
    d = os.path.join(ROOT, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    json.dump(report, open(os.path.join(d, "report.json"), "w"), ensure_ascii=False, indent=1)
    sf.write(os.path.join(d, "conversation.wav"), np.concatenate(convo), 16000)
    delays = [r["delay_s"] for r in report if "delay_s" in r]
    print(f"\nmedian delay {np.median(delays):.2f} s, max {max(delays):.2f} s, numbers ok "
          f"{sum(1 for r in report if r.get('numbers_ok'))}/{sum(1 for r in report if r.get('numbers_ok') is not None)}  -> {d}")
    for b in bots.values():
        await b.ws.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8770/ws"))
