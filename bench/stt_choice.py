"""Which Whisper should Yap use? Runs Yap's own hear() step (Indian-call hint included) on the
bolbench audio and checks the numbers, plus how long each line takes to transcribe.

PYTHONPATH=.:../bolbench .venv/bin/python -m bench.stt_choice ../bolbench [models] [conds] [out_name]
"""
import json
import os
import sys
import time

import mlx_whisper
import numpy as np
import soundfile as sf

from bolbench.items import read
from yap.engine import HINT

MODELS = {"turbo": "mlx-community/whisper-large-v3-turbo", "large-v3": "mlx-community/whisper-large-v3-mlx"}
CONDS = ["clean", "phone", "street", "call"]


def main(bb, models=None, conds=None, out_name="stt_choice"):
    items = [json.loads(l) for l in open(os.path.join(bb, "data/items.jsonl"))]
    suspect = set(json.load(open(os.path.join(bb, "results/summary.json")))["suspect"])
    items = [it for it in items if it["id"] not in suspect]
    out = {}
    for name, repo in MODELS.items():
        if models and name not in models:
            continue
        mlx_whisper.transcribe(np.zeros(16000, np.float32), path_or_hf_repo=repo, language="hi")   # warm up
        for c in conds or CONDS:
            ok, secs, rows = 0, [], []
            for it in items:
                x, _ = sf.read(os.path.join(bb, "data/audio", c, it["id"].replace("/", "_") + ".t0.wav"), dtype="float32")
                t = time.perf_counter()
                txt = mlx_whisper.transcribe(x, path_or_hf_repo=repo, language="hi", condition_on_previous_text=False,
                                             initial_prompt=HINT["hi"])["text"].strip()
                secs.append(time.perf_counter() - t)
                good = read(it["kind"], txt) == it["gold"]
                ok += good
                rows.append({"id": it["id"], "ok": good, "heard": txt})
            out[f"{name}|{c}"] = {"ok": ok, "n": len(items), "median_s": round(float(np.median(secs)), 2), "rows": rows}
            print(f"{name:9} {c:6} numbers {ok}/{len(items)} ({100*ok/len(items):.0f}%)  median {np.median(secs):.2f} s/line", flush=True)
    json.dump(out, open(f"results/{out_name}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1].split(",") if len(a) > 1 else None, a[2].split(",") if len(a) > 2 else None,
         a[3] if len(a) > 3 else "stt_choice")
