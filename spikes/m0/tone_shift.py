"""Find tonal shifts between consecutive lines from one speaker.

    uv run tone_shift.py latest                 # Passe's replies in the newest event log, per session
    uv run tone_shift.py path/to/spike-….jsonl  # a specific event log
    uv run tone_shift.py transcript.txt         # one line per unit
    uv run tone_shift.py --check                # run the hand-labelled sample and score it

Output: a JSON array like [{"index": 3, "from": "neutral", "to": "sarcastic", "confidence": 0.87}], where index is
the 0-based line where the new tone starts. Lines are labelled by gpt-oss-120b (Baseten) from a fixed tone set;
each line is labelled three times and majority-voted; shifts are then computed in code. Lines under four words are
too short to judge and carry the previous tone forward.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TONES = ["neutral", "friendly", "warm", "playful", "affectionate", "sarcastic", "formal", "casual",
         "authoritative", "apologetic", "excited", "irritated", "sad"]
MIN_WORDS = 4
MIN_CONFIDENCE = 0.6
MILD = {"neutral", "friendly", "casual", "warm"}  # neighbours: a shift among these must be clear-cut
MILD_CONFIDENCE = 0.85
TAG = re.compile(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>")

SYSTEM = f"""You label the tone of each line of a spoken assistant's replies.
Tones (use exactly one per line): {", ".join(TONES)}.
Judge the speaker's manner, not the topic. Sound tags like <chuckle> or <sigh> are performed sounds and count as
evidence of tone. Reply with JSON only: {{"labels": [{{"line": <n>, "tone": "<tone>", "confidence": <0..1>}}]}},
one entry per input line, in order."""


def label(lines: list[str], votes: int = 3) -> list[dict]:
    """Majority vote over several labelling passes: a line whose tone flips between passes ends up with low
    confidence (confidence × share of agreeing votes) instead of producing a phantom shift."""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(votes) as pool:
        runs = list(pool.map(lambda _: label_once(lines), range(votes)))
    out = []
    for i in range(len(lines)):
        tones = Counter(r[i].get("tone") for r in runs)
        tone, n = tones.most_common(1)[0]
        confs = [float(r[i].get("confidence") or 0) for r in runs if r[i].get("tone") == tone]
        out.append({"tone": tone, "confidence": round(sum(confs) / len(confs) * n / votes, 3)})
    return out


def label_once(lines: list[str]) -> list[dict]:
    from openai import OpenAI

    import spike

    client = OpenAI(api_key=spike.load_api_key(), base_url="https://inference.baseten.co/v1")
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b", reasoning_effort="medium", max_completion_tokens=4000, temperature=0, seed=7,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": numbered}],
    )
    raw = r.choices[0].message.content or "{}"
    labels = {int(x["line"]): x for x in json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("labels", [])}
    return [labels.get(i, {"tone": "neutral", "confidence": 0.0}) for i in range(len(lines))]


def shifts(lines: list[str], labels: list[dict]) -> list[dict]:
    out, prev, prev_conf = [], None, 0.0
    for i, (line, lab) in enumerate(zip(lines, labels)):
        tone = lab.get("tone") if lab.get("tone") in TONES else "neutral"
        conf = float(lab.get("confidence") or 0.0)
        if len(TAG.sub("", line).split()) < MIN_WORDS or conf < MIN_CONFIDENCE:
            continue  # too short or too unsure to call: carry the previous tone forward
        if prev is not None and tone != prev:
            if {tone, prev} <= MILD and min(prev_conf, conf) < MILD_CONFIDENCE:
                continue  # e.g. neutral vs friendly at 0.7: noise, not a shift
            out.append({"index": i, "from": prev, "to": tone, "confidence": round(min(prev_conf, conf), 2)})
        prev, prev_conf = tone, conf
    return out


def passe_lines(log: Path) -> list[list[str]]:
    """Passe's replies from an event log, one list per session (a new session resets the comparison)."""
    sessions, cur = [], []
    for rec in (json.loads(x) for x in log.read_text().splitlines() if x.strip()):
        if rec["kind"] in ("session_start", "session_end") and cur:
            sessions.append(cur)
            cur = []
        elif rec["kind"] == "assistant_turn" and (rec.get("full") or "").strip():
            cur.append(rec["full"].strip())
    if cur:
        sessions.append(cur)
    return sessions


SAMPLE = [  # (line, tone it starts if it's a shift) — hand-labelled
    ("Sure. What should the prompt focus on first, the retry logic or the tests?", None),
    ("Got it. I added the retry limit and a note about the timeout path.", None),
    ("Okay.", None),
    ("Oh wow, that actually worked first try! I love it when that happens.", "excited"),
    ("Brilliant. Nothing says confidence like a test suite that only passes on Tuesdays.", "sarcastic"),
    ("Right, five attempts with backoff starting at half a second.", "neutral"),
    ("You must not merge this until the migration is reviewed. That is not optional.", "authoritative"),
    ("Sorry, I should have caught that earlier. I'll fix the draft now.", "apologetic"),
    ("Updated the draft with the corrected file path.", "neutral"),
]


def check() -> None:
    lines = [x for x, _ in SAMPLE]
    expected = {i for i, (_, t) in enumerate(SAMPLE) if t}
    found = shifts(lines, label(lines))
    got = {s["index"] for s in found}
    tp = len(got & expected)
    print(json.dumps(found, indent=1))
    print(f"\nexpected shifts at {sorted(expected)}, found {sorted(got)}")
    print(f"precision {tp / max(1, len(got)):.2f} · recall {tp / max(1, len(expected)):.2f}")
    sys.exit(0 if got == expected else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="latest")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    import spike

    if args.source == "latest":
        logs = sorted(spike.LOG_DIR.glob("spike-*.jsonl"))
        if not logs:
            raise SystemExit(f"no event logs in {spike.LOG_DIR}")
        src = logs[-1]
    else:
        src = Path(args.source).expanduser()
    groups = passe_lines(src) if src.suffix == ".jsonl" else [[x for x in src.read_text().splitlines() if x.strip()]]
    result = []
    offset = 0
    for lines in groups:
        if len(lines) > 1:
            result += [{**s, "index": s["index"] + offset} for s in shifts(lines, label(lines))]
        offset += len(lines)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
