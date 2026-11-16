"""Do translated lines keep the speaker's voice? Measured with Chatterbox's own voice encoder."""
import numpy as np, soundfile as sf, torch, librosa
from kokoro import KPipeline
from chatterbox.vc import ChatterboxVC
from chatterbox.models.voice_encoder import VoiceEncoder

P = {"h": KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M"), "a": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")}
def say(l, voice, text, path):
    a = np.concatenate([np.asarray(x) for _, _, x in P[l](text, voice=voice)]); sf.write(path, a, 24000); return path

# the two characters' own voices (their "enrolment" clip, what the app records when you join)
say("h", "hf_alpha", "नमस्ते, मैं प्रिया हूँ। मैं बैंगलोर से हूँ और पेमेंट्स टीम में काम करती हूँ। आज की मीटिंग में मैं अपना अपडेट दूँगी।", "voices/priya.wav")
say("a", "am_michael", "Hi, I'm Mike. I work on the mobile app from Austin, and I'll share my update in today's standup.", "voices/mike.wav")

vc = ChatterboxVC.from_pretrained("mps")
ve = VoiceEncoder(); ve.load_state_dict(torch.load(__import__("glob").glob(
    __import__("os").path.expanduser("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/ve.pt"))[0], map_location="cpu")); ve.eval()
def emb(p):
    w, _ = librosa.load(p, sr=16000); return ve.embeds_from_wavs([w], sample_rate=16000)[0]

cases = [  # (who, translated text in the OTHER language, generic voice used before conversion)
    ("priya", "a", "I'm blocked on the payment API, can someone help me after lunch?", "af_heart"),
    ("mike", "h", "मेरा ऐप वाला काम पूरा हो गया, कल रिलीज़ करेंगे।", "hm_omega"),
]
E = {w: emb(f"voices/{w}.wav") for w in ("priya", "mike")}
for who, l, text, generic in cases:
    raw = say(l, generic, text, f"voices/{who}_translated_raw.wav")
    w = vc.generate(raw, target_voice_path=f"voices/{who}.wav")
    out = f"voices/{who}_translated_in_own_voice.wav"; sf.write(out, w.squeeze(0).cpu().numpy(), vc.sr)
    e_raw, e_out = emb(raw), emb(out)
    other = "mike" if who == "priya" else "priya"
    cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    print(f"{who}: generic voice vs {who} {cos(e_raw, E[who]):.2f} | converted vs {who} {cos(e_out, E[who]):.2f} | converted vs {other} {cos(e_out, E[other]):.2f}")
