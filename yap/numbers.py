"""Indian numbers: write them as spoken Hinglish, and read them back out of any transcript.

Spoken side (what the TTS reads):  साढ़े तीन हज़ार, डेढ़ लाख, fifteen hundred, डबल पाँच ...
Reading side (what we grade):      Devanagari words, English words, ASCII digits, Devanagari digits,
                                   Indian grouping (1,50,000), decimals (3.5 हज़ार), clock times (5:30).

Spelling in STT output drifts (पाँच/पांच, छह/छः/छे, चौवालीस/चवालीस), so words are matched after a
loose normalisation (no nukta, one nasal mark, short = long vowels). Distinct numbers never collide
after normalisation: that is asserted at import time.
"""
from __future__ import annotations

import re
import unicodedata

HI = """0 शून्य|ज़ीरो|जीरो
1 एक
2 दो
3 तीन
4 चार
5 पाँच|पांच
6 छह|छः|छे|छै
7 सात
8 आठ
9 नौ
10 दस
11 ग्यारह
12 बारह
13 तेरह
14 चौदह
15 पंद्रह|पन्द्रह
16 सोलह
17 सत्रह|सत्तरह
18 अठारह|अट्ठारह
19 उन्नीस
20 बीस
21 इक्कीस
22 बाईस|बाइस
23 तेईस|तेइस
24 चौबीस
25 पच्चीस
26 छब्बीस
27 सत्ताईस|सताईस
28 अट्ठाईस|अठाईस
29 उनतीस|उन्तीस
30 तीस
31 इकतीस|इकत्तीस
32 बत्तीस
33 तैंतीस|तेंतीस
34 चौंतीस|चौतीस
35 पैंतीस|पेंतीस
36 छत्तीस
37 सैंतीस|सेंतीस
38 अड़तीस
39 उनतालीस|उन्तालीस
40 चालीस
41 इकतालीस
42 बयालीस|बियालीस
43 तैंतालीस|तेंतालीस
44 चौवालीस|चवालीस
45 पैंतालीस|पेंतालीस
46 छियालीस|छयालीस
47 सैंतालीस|सेंतालीस
48 अड़तालीस
49 उनचास|उनन्चास
50 पचास
51 इक्यावन|इकावन
52 बावन
53 तिरपन|तरेपन
54 चौवन|चव्वन
55 पचपन
56 छप्पन
57 सत्तावन
58 अट्ठावन|अठावन
59 उनसठ
60 साठ
61 इकसठ
62 बासठ
63 तिरसठ|तिरेसठ
64 चौंसठ
65 पैंसठ
66 छियासठ
67 सड़सठ|सरसठ
68 अड़सठ
69 उनहत्तर
70 सत्तर
71 इकहत्तर
72 बहत्तर
73 तिहत्तर
74 चौहत्तर
75 पचहत्तर
76 छिहत्तर
77 सतहत्तर
78 अठहत्तर|अठत्तर
79 उन्यासी|उनासी|उनयासी
80 अस्सी
81 इक्यासी
82 बयासी
83 तिरासी
84 चौरासी
85 पचासी
86 छियासी
87 सत्तासी
88 अट्ठासी
89 नवासी|नब्बासी
90 नब्बे
91 इक्यानवे|इक्यानबे
92 बानवे|बानबे
93 तिरानवे|तिरानबे
94 चौरानवे|चौरानबे
95 पचानवे|पंचानवे|पचानबे
96 छियानवे|छियानबे
97 सत्तानवे|सत्तानबे
98 अट्ठानवे|अट्ठानबे
99 निन्यानवे|निन्यानबे"""

HI_WORDS = {}                   # n -> canonical spoken word
for _line in HI.splitlines():
    _n, _ws = _line.split(" ", 1)
    HI_WORDS[int(_n)] = _ws.split("|")[0]

EN_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen nineteen").split()
EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}

MULT_HI = {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000, "करोड़": 10**7, "करोड": 10**7}
MULT_EN = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
           "crore": 10**7, "crores": 10**7, "k": 1000}
# fraction words: (kind, value). "pre" words modify the next number: साढ़े तीन = 3.5, सवा दो = 2.25,
# पौने नौ = 8.75. "alone" words are complete numbers: डेढ़ = 1.5, ढाई = 2.5.
FRAC = {"साढ़े": ("pre", 0.5), "साढे": ("pre", 0.5), "सवा": ("pre", 0.25), "पौने": ("pre", -0.25),
        "पोने": ("pre", -0.25), "डेढ़": ("alone", 1.5), "डेढ": ("alone", 1.5), "ढाई": ("alone", 2.5),
        "अढ़ाई": ("alone", 2.5), "saade": ("pre", 0.5), "sawa": ("pre", 0.25), "paune": ("pre", -0.25),
        "dedh": ("alone", 1.5), "dhai": ("alone", 2.5)}
REPEAT = {"डबल": 2, "double": 2, "ट्रिपल": 3, "triple": 3}

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(w: str) -> str:
    """Loose spelling key for a Devanagari or Latin word."""
    w = unicodedata.normalize("NFC", w.lower())
    w = w.replace("़", "")                         # nukta
    w = w.replace("ँ", "ं")                   # chandrabindu -> anusvara
    w = w.replace("ढ", "ड").replace("ट्ठ", "ठ").replace("ड्ड", "ड")   # साढ़े ~ साड़े, अट्ठावन ~ अठावन
    for long, short in (("ी", "ि"), ("ू", "ु"), ("ै", "े"), ("ौ", "ो"), ("ई", "इ"), ("ऊ", "उ")):
        w = w.replace(long, short)
    return w


LEX = {}                        # norm(word) -> value (0..99)
for _line in HI.splitlines():
    _n, _ws = _line.split(" ", 1)
    for _w in _ws.split("|"):
        k = norm(_w)
        assert LEX.get(k, int(_n)) == int(_n), f"spelling collision: {_w}"
        LEX[k] = int(_n)
for _i, _w in enumerate(EN_UNITS):
    LEX[_w] = _i
for _n, _w in EN_TENS.items():
    LEX[_w] = _n
LEX["oh"] = 0                   # "nine eight oh one"
LEX.update({"सिफ़र": 0})
MULT = {norm(k): v for k, v in {**MULT_HI, **MULT_EN}.items()}
FRACN = {norm(k): v for k, v in FRAC.items()}
REPN = {norm(k): v for k, v in REPEAT.items()}


# ---------------------------------------------------------------- speaking

def hi_int(n: int) -> str:
    """Integer in Hindi words, Indian grouping: 150000 -> एक लाख पचास हज़ार."""
    if n < 100:
        return HI_WORDS[n]
    parts = []
    for size, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        q, n = divmod(n, size)
        if q:
            parts.append(f"{hi_int(q)} {word}")
    if n:
        parts.append(HI_WORDS[n])
    return " ".join(parts)


def en_under_100(n: int) -> str:
    if n < 20:
        return EN_UNITS[n]
    t, u = divmod(n, 10)
    return EN_TENS[t * 10] + (f" {EN_UNITS[u]}" if u else "")


def en_int(n: int, hundreds_style: bool = False) -> str:
    """English words, Indian grouping. hundreds_style: 1500 -> fifteen hundred."""
    if hundreds_style and 1000 < n < 10000 and n % 100 == 0 and n % 1000:
        return f"{en_under_100(n // 100)} hundred"
    if n < 100:
        return en_under_100(n)
    parts = []
    for size, word in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand"), (100, "hundred")):
        q, n = divmod(n, size)
        if q:
            parts.append(f"{en_int(q)} {word}")
    if n:
        parts.append(en_under_100(n))
    return " ".join(parts)


def hi_fraction(n: int):
    """Spoken fractional form if one exists: 3500 -> साढ़े तीन हज़ार, 150000 -> डेढ़ लाख, 1250 -> सवा हज़ार.

    Returns None when the number has no natural fractional form."""
    for size, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n < size or n % (size // 4):
            continue
        q, r = divmod(n, size)
        if r == 0:
            continue
        if q == 1 and r == size // 2:
            return f"डेढ़ {word}"
        if q == 2 and r == size // 2:
            return f"ढाई {word}"
        if r == size // 2 and q >= 3 and q < 100:
            return f"साढ़े {hi_int(q)} {word}"
        if r == size // 4 and q < 100:
            return f"सवा {hi_int(q) if q > 1 else ''} {word}".replace("  ", " ")
        if r == 3 * size // 4 and q + 1 < 100:
            return f"पौने {hi_int(q + 1)} {word}"
    return None


def hi_clock(h: int, m: int):
    """Clock time as spoken: 5:30 -> साढ़े पाँच, 2:15 -> सवा दो, 8:45 -> पौने नौ, 1:30 -> डेढ़."""
    if m == 0:
        return hi_int(h)
    if m == 30:
        return {1: "डेढ़", 2: "ढाई"}.get(h, f"साढ़े {hi_int(h)}")
    if m == 15:
        return f"सवा {hi_int(h)}"
    if m == 45:
        return f"पौने {hi_int(h % 12 + 1)}"
    return f"{hi_int(h)} बजकर {hi_int(m)} मिनट"


# ---------------------------------------------------------------- reading

DEVA = "[\u0900-\u0963\u0970-\u097F\u200c\u200d]+"


def tokens(text: str):
    text = unicodedata.normalize("NFC", text).translate(DEV_DIGITS)
    text = re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", text)
    return re.findall(r"₹|\d+:\d+|\d[\d,]*(?:\.\d+)?|" + DEVA + "|[A-Za-z]+", text)


def _kind(tok):
    """Classify one token: ('num', v) | ('mult', v) | ('pre', v) | ('alone', v) | ('rep', k) | None."""
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", tok):
        return ("num", float(tok.replace(",", "")))
    k = norm(tok)
    if k in LEX:
        return ("num", float(LEX[k]))
    if k in MULT:
        return ("mult", MULT[k])
    if k in FRACN:
        return FRACN[k]
    if k in REPN:
        return ("rep", REPN[k])
    return None


def phrases(text: str):
    """Group a transcript into number phrases: a list of lists of (kind, value, raw token).

    Fillers that sit between number words (और, and, रुपये ...) end a phrase, except 'and' / 'और'
    inside a phrase ('one lakh and fifty thousand')."""
    out, cur = [], []
    toks = tokens(text)
    for i, t in enumerate(toks):
        if re.fullmatch(r"\d+:\d+", t):
            if cur:
                out.append(cur); cur = []
            out.append([("clock", t, t)])
            continue
        k = _kind(t)
        if k is None:
            if cur and norm(t) in ("and", "और") and i + 1 < len(toks) and _kind(toks[i + 1]):
                continue
            if cur:
                out.append(cur); cur = []
            continue
        cur.append((k[0], k[1], t))
    if cur:
        out.append(cur)
    return out


def value(ph) -> float:
    """Evaluate one phrase as a quantity: डेढ़ लाख = 150000, साढ़े तीन हज़ार = 3500,
    fifteen hundred = 1500, एक लाख पचास हज़ार = 150000, 3.5 हज़ार = 3500, 1,50,000 = 150000,
    पौने नौ = 8.75, दो सौ पचास = 250."""
    total, group, pending = 0.0, None, 0.0
    for kind, v, _ in ph:
        if kind == "clock":
            h, m = v.split(":")
            return int(h) + int(m) / 60
        if kind == "pre":
            pending += v
        elif kind == "alone":
            group = (group or 0) + v
        elif kind == "num":
            if pending:
                v, pending = v + pending, 0.0
            group = v if group is None else group + v
        elif kind == "mult":
            base = group if group is not None else 1 + pending
            pending = 0.0
            if v == 100:
                group = base * 100              # binds to the small group: दो सौ, fifteen hundred
            else:
                total += base * v
                group = None
    return total + (group or 0)


def amounts(text: str):
    """Every quantity said in the text, in order."""
    return [value(p) for p in phrases(text) if any(k[0] in ("num", "alone", "clock", "pre") for k in p)]


def digit_string(text: str) -> str:
    """Read a spoken digit sequence (phone, OTP): every digit word, 'double five' -> 55,
    two-digit words kept as two digits (अट्ठानवे -> 98), digit runs kept as written."""
    out, rep = [], 1
    for t in tokens(text):
        if re.fullmatch(r"\d[\d,]*", t):
            out.append(t.replace(",", "") * rep); rep = 1
            continue
        k = _kind(t)
        if not k:
            continue
        if k[0] == "rep":
            rep = k[1]
        elif k[0] == "num":
            s = str(int(k[1])) if k[1] == int(k[1]) else ""
            out.append(s * rep); rep = 1
    return "".join(out)
