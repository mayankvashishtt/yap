"""Number lock: numbers never go through the translator as words.

Before translation, every spoken number in the line is parsed (साढ़े तीन हज़ार रुपये -> ₹3,500,
पौने नौ बजे -> 8:45, नौ आठ सात डबल पाँच ... -> 9875543210) so the model only has to copy digits.
After translation, digits are turned back into words the target voice says naturally
(₹3,500 -> साढ़े तीन हज़ार रुपये / three thousand five hundred rupees).
"""
from __future__ import annotations

import re

from .numbers import DEV_DIGITS, _kind, en_int, en_under_100, hi_clock, hi_fraction, hi_int, norm, value, HI_WORDS, EN_UNITS

WORD = re.compile(r"\d[\d,]*(?:\.\d+)?|[ऀ-ॣ॰-ॿ‌‍]+|[A-Za-z]+|₹")
CURRENCY = re.compile(r"^(रु|रू|rup|rs$|inr$)")
TIME_AFTER = ("बजे", "बज", "baje", "oclock", "o")
SMALL_WORDS = {"दो", "एक", "do", "ek", "one", "a"}       # also mean 'give' / 'a': only lock them with context


def indian(n: float) -> str:
    """3500 -> 3,500   150000 -> 1,50,000"""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return f"{head},{tail}"


def _runs(text):
    """Number-word runs with their character spans."""
    toks = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text.translate(DEV_DIGITS))]
    runs, cur = [], []
    for i, (t, s, e) in enumerate(toks):
        k = _kind(t) if not re.fullmatch(r"\d[\d,]*(?:\.\d+)?", t) else ("num", float(t.replace(",", "")))
        if k:
            cur.append((k, t, s, e, i))
        elif cur and norm(t) in ("and", "और") and i + 1 < len(toks) and _kind(toks[i + 1][0]):
            continue
        else:
            if cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs, toks


NUMWORD = r"(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|" \
          r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|half|and a half)"


AFTER = r"\s*(?:\d|rupees?\b|rs\b|₹|and\b|" + NUMWORD[3:]


def fix_indian_units(text: str) -> str:
    """English speech models don't know lakh/crore: 'one lock 50,000' -> 'one lakh 50,000'."""
    text = re.sub(rf"\b({NUMWORD})\s+(?:lock|lack|luck|lac|lak|lakhs?|locks)\b(?={AFTER})", r"\1 lakh", text, flags=re.I)
    text = re.sub(rf"\b({NUMWORD})\s+(?:core|cores|crow|crows|crore|crores|karod|krore)\b(?={AFTER})", r"\1 crore", text, flags=re.I)
    return text


def lock(text: str) -> str:
    """Replace spoken numbers with digits before translation."""
    text = fix_indian_units(text)
    runs, toks = _runs(text)
    out, pos = [], 0
    for run in runs:
        kinds = [r[0] for r in run]
        s, e, last_i = run[0][2], run[-1][3], run[-1][4]
        nxt = norm(toks[last_i + 1][0]) if last_i + 1 < len(toks) else ""
        prev = toks[run[0][4] - 1][0] if run[0][4] > 0 else ""
        nums = [k for k in kinds if k[0] == "num"]
        has_rep = any(k[0] == "rep" for k in kinds)
        single_digits = len(nums) >= 4 and all(k[1] < 10 for k in nums if k[0] == "num")
        if not any(k[0] in ("num", "alone") for k in kinds) and not (
                any(k[0] == "pre" for k in kinds) and any(k[0] == "mult" for k in kinds)):
            continue                                                     # a bare 'हज़ार' after garbled words: don't guess
        raw_groups = [t for _, t, *_ in run]
        if len(run) >= 2 and all(re.fullmatch(r"\d+", t) for t in raw_groups):
            out.append(text[pos:s]); out.append("".join(raw_groups)); pos = e   # 987 5543210 -> one number
            continue
        if has_rep or single_digits:                                     # phone / OTP: digit by digit
            digits, rep = "", 1
            for k in kinds:
                if k[0] == "rep":
                    rep = k[1]
                elif k[0] == "num":
                    digits += str(int(k[1])) * rep; rep = 1
            rep_txt = digits
        else:
            if len(run) == 1 and run[0][1] in SMALL_WORDS and not (CURRENCY.match(nxt) or nxt in TIME_AFTER):
                continue                                                 # 'कर दो' = 'do it', not 2
            if len(run) == 1 and run[0][0][0] in ("pre",):
                continue
            v = value([(k[0], k[1], t) for k, t, *_ in run if k[0] != "rep"])
            if nxt in ("बजे", "बज", "baje"):
                h = int(v); m = int(round((v - h) * 60))
                rep_txt = f"{h}:{m:02d}"
            else:
                rep_txt = indian(v) if v == int(v) else f"{v:g}"
                if CURRENCY.match(nxt) or prev == "₹":
                    rep_txt = "₹" + rep_txt
                    # swallow the currency word that follows
                    e = toks[last_i + 1][2] if CURRENCY.match(nxt) else e
        out.append(text[pos:s]); out.append(rep_txt); pos = e
    out.append(text[pos:])
    return re.sub(r"₹\s*₹", "₹", "".join(out))


def _digits_spoken(d: str, lang: str) -> str:
    words = HI_WORDS if lang == "hi" else dict(enumerate(EN_UNITS))
    return " ".join(words[int(c)] for c in d)


def speak_numbers(text: str, lang: str) -> str:
    """Digits -> words the target voice should say. lang: 'hi' or 'en'."""
    def time_(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if lang == "hi":
            return hi_clock(h, mi)
        return f"{en_int(h)}" + ("" if mi == 0 else f" {en_under_100(mi) if mi >= 10 else 'oh ' + EN_UNITS[mi]}")

    def amount(m):
        raw = m.group(2).replace(",", "")
        cur = bool(m.group(1))
        digits_only = not cur and raw.isdigit() and len(raw) >= 6 and "," not in m.group(2)
        if digits_only:                       # phone / OTP: say digit by digit
            return _digits_spoken(re.sub(r"\D", "", m.group(0)), lang)
        n = float(raw)
        if n != int(n):
            return m.group(0)
        n = int(n)
        if lang == "hi":
            w = hi_fraction(n) if n >= 1000 and hi_fraction(n) else hi_int(n)
            return w + (" रुपये" if cur else "")
        return en_int(n) + (" rupees" if cur else "")

    text = re.sub(r"(\d{1,2}):(\d{2})", time_, text)
    # phone-like: groups of digits separated by spaces (98765 43210) -> one digit string
    text = re.sub(r"\b(\d{3,5})[ -](\d{3,5})(?:[ -](\d{2,5}))?\b",
                  lambda m: _digits_spoken("".join(g for g in m.groups() if g), lang), text)
    text = re.sub(r"(₹\s?)?(\d[\d,]*(?:\.\d+)?)", amount, text)
    text = text.replace("₹", "rupees " if lang == "en" else "रुपये ")
    return text
