"""How consistent is Orpheus across a multi-sentence reply? Compares chunking and sampling settings.

    uv run consistency_eval.py [--trials 3]

Uses the running Orpheus server (or starts one). Scores each rendering over ~1.5 s windows:
  identity  mean pairwise speaker-embedding similarity between windows (higher = same-sounding person)
  pitch     std of median F0 across windows in Hz (lower = steadier)
  loudness  std of RMS across windows in dB (lower = steadier)
Nothing is written to disk.
"""

import argparse
import itertools
import re
import time

import numpy as np
from openai import OpenAI

import orpheus_tts as o
import spike

REPLY = ("Oh, nice, the tests are passing again. I tightened the retry limit to five attempts and added a test for "
         "the timeout path. The backoff starts at half a second and doubles each time. If you'd rather keep it at "
         "three attempts, that's a one-line change. Want me to copy the prompt to your clipboard?")
SR = 24000


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


class Renderer:
    def __init__(self):
        import torch
        from huggingface_hub import hf_hub_download
        from snac import SNAC
        from tokenizers import Tokenizer

        self.torch = torch
        self.tok = Tokenizer.from_file(hf_hub_download(o.ORPHEUS_MODEL, "tokenizer.json"))
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
        self.client = OpenAI(base_url=o.BASE_URL, api_key="local")

    def render(self, text: str, voice: str, temperature: float, seed: int | None) -> tuple[np.ndarray, float]:
        ids = o._START + self.tok.encode(f"{voice}: {text}").ids + o._END
        extra = {"repetition_penalty": 1.1, "stop_token_ids": [o._STOP], "skip_special_tokens": False}
        if seed is not None:
            extra["seed"] = seed
        t0 = time.time()
        first = None
        out = ""
        for ch in self.client.completions.create(model=o.ORPHEUS_MODEL, prompt=ids, max_tokens=2000, stream=True,
                                                 temperature=temperature, top_p=0.8, extra_body=extra):
            piece = ch.choices[0].text if ch.choices else ""
            out += piece
            if first is None and out.count("<custom_token_") >= 28:  # first audio frame is decodable here
                first = time.time() - t0
        codes = []
        for m in o._TOKEN.finditer(out):
            c = int(m.group(1)) - 10 - (len(codes) % 7) * 4096
            if c > 0:
                codes.append(c)
        codes = codes[: len(codes) // 7 * 7]

        class D:
            pass

        d = D()
        d._torch, d._snac = self.torch, self.snac
        audio = o.OrpheusTTSService._decode(d, codes)
        return (audio[0, 0].float().cpu().numpy() if audio is not None else np.zeros(1)), (first or 0.0)


def f0(x: np.ndarray) -> float | None:
    """Median pitch of voiced 80 ms frames (autocorrelation), or None if unvoiced."""
    vals = []
    for i in range(0, len(x) - 1920, 960):
        fr = x[i : i + 1920] - x[i : i + 1920].mean()
        if np.sqrt((fr**2).mean()) < 0.01:
            continue
        ac = np.correlate(fr, fr, "full")[len(fr) - 1 :]
        lo, hi = SR // 400, SR // 70
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / (ac[0] + 1e-9) > 0.35:
            vals.append(SR / lag)
    return float(np.median(vals)) if len(vals) >= 3 else None


def score(audio: np.ndarray, vp) -> dict:
    win = int(1.5 * SR)
    wins = [audio[i : i + win] for i in range(0, len(audio) - win // 2, win)]
    wins = [w for w in wins if np.sqrt((w**2).mean()) > 0.01]
    embs = []
    for w in wins:
        pcm16 = (np.interp(np.arange(0, len(w), 1.5), np.arange(len(w)), w) * 32767).astype(np.int16).tobytes()
        embs.append(vp.embed(pcm16, 16000))
    sims = [float(a @ b) for a, b in itertools.combinations(embs, 2)]
    pitches = [p for p in (f0(w) for w in wins) if p]
    louds = [20 * np.log10(np.sqrt((w**2).mean()) + 1e-9) for w in wins]
    return {"identity": np.mean(sims) if sims else np.nan, "pitch_std": np.std(pitches) if pitches else np.nan,
            "loud_std": float(np.std(louds)), "secs": len(audio) / SR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--voice", default="tara")
    args = ap.parse_args()
    proc = o.launch_server()
    o.wait_ready(proc)
    r = Renderer()
    vp = spike.VoicePrint()
    r.render("Warming up.", args.voice, 0.6, None)

    configs = [
        ("per sentence · T0.6 (current)", "sentence", 0.6, None),
        ("per sentence · T0.4", "sentence", 0.4, None),
        ("per sentence · T0.4 · fixed seed", "sentence", 0.4, 7),
        ("whole reply · T0.6", "whole", 0.6, None),
        ("whole reply · T0.4", "whole", 0.4, None),
        ("first sentence + rest · T0.4", "first+rest", 0.4, None),
    ]
    print(f"{'config':36} {'identity':>9} {'pitch Hz':>9} {'loud dB':>8} {'1st audio':>10} {'length':>7}")
    for label, mode, temp, seed in configs:
        rows = []
        for _ in range(args.trials):
            sents = sentences(REPLY)
            parts = {"sentence": sents, "whole": [REPLY],
                     "first+rest": [sents[0], " ".join(sents[1:])]}[mode]
            audios, firsts = [], []
            for p in parts:
                a, first = r.render(p, args.voice, temp, seed)
                audios.append(a)
                firsts.append(first)
            res = score(np.concatenate(audios), vp)
            res["first"] = firsts[0]
            rows.append(res)
        m = {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]}
        print(f"{label:36} {m['identity']:9.3f} {m['pitch_std']:9.1f} {m['loud_std']:8.2f} "
              f"{m['first'] * 1000:8.0f}ms {m['secs']:6.1f}s")
    if proc is not None:
        o.stop_server(proc)


if __name__ == "__main__":
    main()
