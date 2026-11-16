import time, glob, os, numpy as np, torch, librosa
import yap.engine as eng
from chatterbox.models.voice_encoder import VoiceEncoder
E = eng.Engine()
E.enroll("priya", librosa.load("voices/priya.wav", sr=16000)[0]); E.enroll("mike", librosa.load("voices/mike.wav", sr=16000)[0])
ve = VoiceEncoder(); ve.load_state_dict(torch.load(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/ve.pt"))[0], map_location="cpu")); ve.eval()
emb = lambda w, sr: ve.embeds_from_wavs([librosa.resample(w, orig_sr=sr, target_sr=16000)], sample_rate=16000)[0]
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
ref = {k: emb(librosa.load(f"voices/{k}.wav", sr=16000)[0], 16000) for k in ("priya", "mike")}
lines = [("priya", "en", "I'm blocked on the payment API, can someone help me after lunch?"), ("mike", "hi", "मेरा ऐप वाला काम पूरा हो गया, कल रिलीज़ करेंगे।"),
         ("priya", "en", "Yeah, I sent back three thousand five hundred rupees."), ("mike", "hi", "कल सुबह पौने नौ बजे क्लाइंट कॉल है।")]
gen = [(who, E.speak_generic(t, l)) for who, l, t in lines]
for steps in (10, 6, 4, 2):
    eng.VC_STEPS = steps
    E.convert(gen[0][1], "priya")
    ts, sims, gap = [], [], []
    for who, g in gen:
        t = time.time(); a = E.convert(g, who); torch.mps.synchronize(); ts.append(time.time() - t)
        e = emb(a, 24000); other = "mike" if who == "priya" else "priya"
        sims.append(cos(e, ref[who])); gap.append(cos(e, ref[who]) - cos(e, ref[other]))
    print(f"steps {steps:>2}: convert {np.mean(ts):.2f}s  match own voice {np.mean(sims):.2f}  (own minus other {np.mean(gap):+.2f})", flush=True)
