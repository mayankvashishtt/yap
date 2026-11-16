"""Builds docs/demo.mp4 from a real yap.bots run: real audio, real transcripts, waiting gaps trimmed."""
import json, os, re, subprocess, sys, html
import numpy as np, soundfile as sf, librosa, imageio_ffmpeg

RUN = sys.argv[1]
CLIPS = "runs/clips"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = "demo/build"; os.makedirs(OUT, exist_ok=True)
rep = json.load(open(os.path.join(RUN, "report.json")))
SR = 24000

CSS = """
*{box-sizing:border-box;margin:0}
body{width:1920px;height:1080px;background:#0d0f12;color:#eef1f4;font-family:-apple-system,"Helvetica Neue",sans-serif;overflow:hidden}
.top{position:absolute;left:80px;top:60px;font-weight:800;font-size:44px;letter-spacing:-.02em}.top span{color:#ff7a3d}
.badge{position:absolute;right:80px;top:66px;font-size:24px;color:#3ddc97;border:2px solid #1f4a38;border-radius:99px;padding:8px 22px}
.people{position:absolute;left:80px;top:190px;width:520px;display:flex;flex-direction:column;gap:22px}
.p{display:flex;align-items:center;gap:22px;padding:24px;border:2px solid #252a31;border-radius:22px;background:#15181d}
.p.talk{border-color:#3ddc97;box-shadow:0 0 0 6px #3ddc9722}.p.listen{border-color:#6aa8ff;box-shadow:0 0 0 6px #6aa8ff22}
.av{width:78px;height:78px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:34px;color:#0d0f12}
.n{font-weight:700;font-size:34px}.l{font-size:24px;color:#8b94a0}
.state{font-size:24px;margin-top:4px}
.main{position:absolute;left:660px;right:80px;top:190px}
.label{font-size:26px;color:#8b94a0;margin-bottom:18px;text-transform:uppercase;letter-spacing:.08em}
.big{font-size:60px;line-height:1.25;font-weight:700}
.orig{font-size:36px;color:#aab3bd;line-height:1.4;margin-top:34px}
.hl{color:#ffb547}
.foot{position:absolute;left:660px;right:80px;bottom:70px;font-size:26px;color:#8b94a0}
.center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:26px}
.h1{font-size:84px;font-weight:800;letter-spacing:-.03em;line-height:1.1}
.sub{font-size:36px;color:#8b94a0}
.mono{font-family:Menlo,monospace;font-size:38px;color:#ff7a3d}
"""
PEOPLE = {"priya": ("Priya", "हिन्दी", "#3ddc97"), "mike": ("Mike", "English", "#6aa8ff")}

def hl(t):
    t = html.escape(t)
    return re.sub(r"(₹[\d,]+|\d{1,2}:\d\d|\d{6,}|[\d,]{4,})", r'<span class="hl">\1</span>', t)

def page(body):
    return f"<!doctype html><meta charset=utf-8><style>{CSS}</style><body>{body}</body>"

def people(talk=None, listen=None, state=""):
    out = []
    for k, (n, l, c) in PEOPLE.items():
        cls = "talk" if k == talk else "listen" if k == listen else ""
        st = state if k in (talk, listen) else ""
        out.append(f'<div class="p {cls}"><div class="av" style="background:{c}">{n[0]}</div><div><div class="n">{n}</div><div class="l">{l}</div><div class="state">{st}</div></div></div>')
    return '<div class="people">' + "".join(out) + "</div>"

HEAD = '<div class="top">yap<span>.</span></div><div class="badge">running on this MacBook · no cloud</div>'
scenes = []   # (html, audio array or seconds of silence)
scenes.append((page('<div class="center"><div class="h1">Priya speaks Hindi.<br>Mike hears English.<br><span style="color:#ff7a3d">In Priya\'s own voice.</span></div><div class="sub">A live call translator that runs fully on a laptop</div></div>'), 3.2))

picks = [2, 4, 5]   # ₹3,500 · phone number · 1.5 lakh  (+ the 8:45 line)
picks = [2, 3, 4, 5]
for i in picks:
    r = rep[i]
    who = r["who"]; other = "mike" if who == "priya" else "priya"
    said = librosa.load(f"{CLIPS}/{i+1:02d}_said.wav", sr=SR)[0]
    heard = librosa.load(f"{CLIPS}/{i+1:02d}_heard.wav", sr=SR)[0]
    n_w, l_w, _ = PEOPLE[who]; n_o, l_o, _ = PEOPLE[other]
    a = page(HEAD + people(talk=who, state="speaking…") +
             f'<div class="main"><div class="label">{n_w} says, in {l_w}</div><div class="big">{hl(r["said"])}</div></div>'
             f'<div class="foot">Whisper is listening…</div>')
    b = page(HEAD + people(talk=None, listen=other, state=f"hearing {n_w}, in {l_o}") +
             f'<div class="main"><div class="label">{n_o} hears, in {l_o}, in {n_w}\'s voice</div><div class="big">{hl(r["other_hears"])}</div>'
             f'<div class="orig">🔒 numbers locked before translating: {hl(r["locked"])}</div></div>'
             f'<div class="foot">{r["delay_s"]:.1f} s after {n_w} stopped talking (wait trimmed in this video)</div>')
    scenes.append((a, said)); scenes.append((a, 0.35)); scenes.append((b, heard)); scenes.append((b, 0.7))

scenes.append((page('<div class="center"><div class="h1">yap<span style="color:#ff7a3d">.</span></div><div class="sub">Hinglish · lakh &amp; crore · your own voice · 100% local</div><div class="mono">github.com/mayankvashishtt/yap</div><div class="sub" style="font-size:28px">Whisper · Gemma 3 · Kokoro · Chatterbox · MLX · open source</div></div>'), 3.5))

# render unique pages
cache, frames, audio = {}, [], []
for h, a in scenes:
    if h not in cache:
        k = len(cache); hp = os.path.abspath(f"{OUT}/s{k}.html"); open(hp, "w").write(h)
        png = os.path.abspath(f"{OUT}/s{k}.png")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080",
                        "--virtual-time-budget=1500", f"--screenshot={png}", "file://" + hp], capture_output=True)
        cache[h] = png
    seg = a if isinstance(a, np.ndarray) else np.zeros(int(a * SR), np.float32)
    seg = np.concatenate([seg, np.zeros(int(0.15 * SR), np.float32)])
    frames.append((cache[h], len(seg) / SR)); audio.append(seg)
wav = np.concatenate(audio); wav = wav / (np.abs(wav).max() + 1e-9) * 0.89
sf.write(f"{OUT}/audio.wav", wav, SR)
with open(f"{OUT}/list.txt", "w") as f:
    for p, d in frames:
        f.write(f"file '{p}'\nduration {d:.3f}\n")
    f.write(f"file '{frames[-1][0]}'\n")
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{OUT}/list.txt", "-i", f"{OUT}/audio.wav",
                "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                "-shortest", "-movflags", "+faststart", "docs/demo.mp4"], check=True)
print("docs/demo.mp4", round(len(wav) / SR, 1), "s")
