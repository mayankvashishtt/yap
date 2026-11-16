"""The interpreter: one spoken line in, the same line in another language and the same voice out.

hear (Whisper turbo, MLX)  ->  number lock  ->  translate (Gemma 3, MLX)  ->  numbers back to words
->  speak (Kokoro, fast generic voice)  ->  voice convert to the speaker (Chatterbox S3Gen, MPS)

Everything runs on the Mac. Every output is watermarked as AI audio (Perth), like Chatterbox ships it.
Models are not thread safe: call Engine from one worker thread only.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import numpy as np
import torch

from .numlock import lock, speak_numbers

STT_MODEL = "mlx-community/whisper-large-v3-turbo"
MT_MODEL = os.environ.get("YAP_MT", "mlx-community/gemma-3-4b-it-4bit")
LANGS = {"hi": ("Hindi", "h", "hf_alpha", "Hindi (Devanagari script)"),
         "en": ("English", "a", "af_heart", "English")}
MT_PROMPT = ("You are a live interpreter on a voice call. Translate the speaker's line into {tgt}. "
             "Keep it casual and short, like spoken language. Copy every number exactly as written "
             "(₹3,500 stays ₹3,500, 8:45 stays 8:45, 9875543210 stays 9875543210). "
             "Output only the translation.\n\nLine: {text}")
SR_IN = 16000
VC_STEPS = int(os.environ.get("YAP_VC_STEPS", "4"))   # 4 = same voice match as 10, 3x faster (bench/vc_steps.py)
# tells Whisper this is an Indian call, so 'lakh' isn't heard as 'lock'
HINT = {"en": "Indian English call. Amounts in rupees, lakh and crore, like 1 lakh 50 thousand rupees.",
        "hi": "हिंदी और English मिली हुई बातचीत। रुपये, लाख, करोड़, साढ़े तीन हज़ार, सवा, पौने।"}


class Engine:
    def __init__(self, device="mps"):
        import mlx_whisper
        from chatterbox.vc import ChatterboxVC
        from kokoro import KPipeline
        from mlx_lm import load
        self._whisper = mlx_whisper
        self.mt, self.tok = load(MT_MODEL)
        self.kokoro = {lang: KPipeline(lang_code=v[1], repo_id="hexgrad/Kokoro-82M") for lang, v in LANGS.items()}
        self.vc = ChatterboxVC.from_pretrained(device)
        self.device = device
        self.voices = {}                    # speaker id -> s3gen reference (their voice)
        self.warm()

    def warm(self):
        silent = np.zeros(SR_IN, dtype=np.float32)
        self.hear(silent, "en")
        self.translate("hello", "hi")
        for lang in LANGS:
            self.speak_generic("ठीक है" if lang == "hi" else "okay", lang)

    # ---------------------------------------------------------------- steps
    def enroll(self, speaker: str, wav16k: np.ndarray):
        """Remember a speaker's voice from ~5-15 s of their speech."""
        from chatterbox.models.s3gen import S3GEN_SR
        ref = librosa.resample(wav16k.astype(np.float32), orig_sr=SR_IN, target_sr=S3GEN_SR)
        ref = ref[: self.vc.DEC_COND_LEN]
        with torch.inference_mode():
            self.voices[speaker] = self.vc.s3gen.embed_ref(ref, S3GEN_SR, device=self.device)

    def hear(self, wav16k: np.ndarray, lang: str) -> str:
        r = self._whisper.transcribe(wav16k.astype(np.float32), path_or_hf_repo=STT_MODEL, language=lang,
                                     condition_on_previous_text=False, initial_prompt=HINT.get(lang))
        return r["text"].strip()

    def translate(self, text: str, tgt: str) -> str:
        from mlx_lm import generate
        msgs = [{"role": "user", "content": MT_PROMPT.format(tgt=LANGS[tgt][3], text=text)}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
        return generate(self.mt, self.tok, prompt=prompt, max_tokens=160).strip()

    def speak_generic(self, text: str, lang: str) -> np.ndarray:
        parts = [np.asarray(a) for _, _, a in self.kokoro[lang](text, voice=LANGS[lang][2])]
        return np.concatenate(parts) if parts else np.zeros(2400, dtype=np.float32)   # 24 kHz

    def convert(self, wav24k: np.ndarray, speaker: str) -> np.ndarray:
        """Make generic speech sound like `speaker`. Returns 24 kHz float32."""
        if speaker not in self.voices:
            return wav24k
        from chatterbox.models.s3tokenizer import S3_SR
        a16 = librosa.resample(wav24k.astype(np.float32), orig_sr=24000, target_sr=S3_SR)
        with torch.inference_mode():
            tokens, _ = self.vc.s3gen.tokenizer(torch.from_numpy(a16).float().to(self.device)[None])
            wav, _ = self.vc.s3gen.inference(speech_tokens=tokens, ref_dict=self.voices[speaker],
                                               n_cfm_timesteps=VC_STEPS)
            wav = wav.squeeze(0).detach().cpu().numpy()
        return self.vc.watermarker.apply_watermark(wav, sample_rate=self.vc.sr).astype(np.float32)

    # ---------------------------------------------------------------- streaming
    @staticmethod
    def sentences(text: str):
        """Split for streaming; tiny pieces are merged so each chunk is worth a round trip."""
        import re
        parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()]
        out = []
        for p in parts:
            if out and len(out[-1]) < 25:
                out[-1] += " " + p
            else:
                out.append(p)
        return out or [text]

    def interpret_stream(self, wav16k, speaker, src, targets, use_lock=True):
        """Yields ('heard', text, locked, ms) then per target ('text', tgt, translation, ms) and
        ('audio', tgt, chunk, is_last) as soon as each sentence is ready."""
        t = time.time(); heard = self.hear(wav16k, src); hear_ms = (time.time() - t) * 1000
        locked = lock(heard) if use_lock else heard
        yield ("heard", heard, locked, {"hear": hear_ms})
        if not heard:
            return
        for tgt in targets:
            if tgt == src:
                continue
            t = time.time(); tr = self.translate(locked, tgt)
            yield ("text", tgt, tr, {f"translate_{tgt}": (time.time() - t) * 1000})
            chunks = self.sentences(speak_numbers(tr, tgt) if use_lock else tr)
            for i, c in enumerate(chunks):
                yield ("audio", tgt, self.convert(self.speak_generic(c, tgt), speaker), i == len(chunks) - 1)

    # ---------------------------------------------------------------- whole line
    def interpret(self, wav16k: np.ndarray, speaker: str, src: str, targets, use_lock=True):
        """Returns {'heard', 'locked', 'out': {lang: {'text', 'said', 'audio'}}, 'ms': {...}}."""
        ms = {}
        t = time.time(); heard = self.hear(wav16k, src); ms["hear"] = (time.time() - t) * 1000
        locked = lock(heard) if use_lock else heard
        out = {}
        for tgt in targets:
            if tgt == src:
                continue
            t = time.time(); tr = self.translate(locked, tgt); ms[f"translate_{tgt}"] = (time.time() - t) * 1000
            said = speak_numbers(tr, tgt) if use_lock else tr
            t = time.time(); g = self.speak_generic(said, tgt); ms[f"speak_{tgt}"] = (time.time() - t) * 1000
            t = time.time(); a = self.convert(g, speaker); ms[f"voice_{tgt}"] = (time.time() - t) * 1000
            out[tgt] = {"text": tr, "said": said, "audio": a}
        return {"heard": heard, "locked": locked, "out": out, "ms": ms}
