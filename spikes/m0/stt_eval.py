"""Speech-to-text accuracy check: record once (in memory), compare front-ends, segmentation and models.

    uv run stt_eval.py              # read the sentences below aloud, in order, at a normal pace

Nothing is written to disk. Prints word error rate (WER) per configuration plus the transcripts.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path

import spike

SENTENCES = [
    "I'm testing the voice loop to see how fast it responds.",
    "Can you draft a prompt that adds retry logic with exponential backoff?",
    "Run the tests with pytest and tell Claude Code to fix the failing ones.",
    "Let's refactor the uploader and keep the existing API stable.",
    "Simplify the last answer from the agent for me.",
    "Okay, sounds good, let's use that.",
]
VOCAB = "Passe, Claude Code, pytest, prompt, testing, refactor, retry logic, backoff, uploader, API, agent, commit"
PARAKEET = Path("~/.cache/passe-spike/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8").expanduser()
SR = 16000


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " ")).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))


async def record(args, seconds: float) -> bytes:
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SR, input=True, frames_per_buffer=512,
                     input_device_index=args.input_device)
    print(f"Recording {seconds:.0f}s. Read the sentences now:")
    for i, s in enumerate(SENTENCES, 1):
        print(f"  {i}. {s}")
    chunks, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        chunks.append(stream.read(512, False))
        await asyncio.sleep(0)
    stream.close()
    pa.terminate()
    return b"".join(chunks)


async def front_ends(raw: bytes) -> dict[str, bytes]:
    import numpy as np
    from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter

    def gain(pcm: bytes, db: float = 12.0) -> bytes:
        x = np.frombuffer(pcm, np.int16).astype(np.float32) * 10 ** (db / 20)
        return np.clip(x, -32768, 32767).astype(np.int16).tobytes()

    rn = RNNoiseFilter()
    await rn.start(SR)
    out = b"".join([await rn.filter(raw[i : i + 640]) for i in range(0, len(raw), 640)])
    await rn.stop()
    out = (out + b"\x00" * len(raw))[: len(raw)]
    return {"raw+gain": gain(raw), "rnnoise+gain": gain(out)}


def segments(pcm: bytes, stop_secs: float) -> list[tuple[int, int]]:
    """Speech regions (sample offsets) like Pipecat's VAD: confidence ≥ 0.7, closed after stop_secs of silence."""
    import numpy as np
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    vad = SileroVADAnalyzer()
    vad.set_sample_rate(SR)
    n = vad.num_frames_required()
    regions, start, silent = [], None, 0
    for i in range(0, len(pcm) // 2 - n, n):
        conf = float(np.asarray(vad.voice_confidence(pcm[i * 2 : (i + n) * 2])).max())
        if conf >= 0.7:
            start = i if start is None else start
            silent = 0
        elif start is not None:
            silent += n
            if silent >= stop_secs * SR:
                regions.append((max(0, start - int(0.2 * SR)), i))
                start, silent = None, 0
    if start is not None:
        regions.append((max(0, start - int(0.2 * SR)), len(pcm) // 2))
    return regions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=40)
    p.add_argument("--input-device", type=int, default=None)
    args = p.parse_args()

    import numpy as np
    import sherpa_onnx
    from faster_whisper import WhisperModel

    whisper = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
    list(whisper.transcribe(np.zeros(SR, np.float32), language="en")[0])
    parakeet = None
    if (PARAKEET / "tokens.txt").exists():
        parakeet = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(PARAKEET / "encoder.int8.onnx"),
            decoder=str(PARAKEET / "decoder.int8.onnx"),
            joiner=str(PARAKEET / "joiner.int8.onnx"),
            tokens=str(PARAKEET / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=4,
        )

    raw = asyncio.run(record(args, args.seconds))
    fronts = asyncio.run(front_ends(raw))
    ref = words(" ".join(SENTENCES))

    def run_whisper(x, prompt):
        segs, _ = whisper.transcribe(x, language="en", initial_prompt=prompt, hotwords=prompt)
        return " ".join(s.text for s in segs)

    def run_parakeet(x):
        st = parakeet.create_stream()
        st.accept_waveform(SR, x)
        parakeet.decode_stream(st)
        return st.result.text

    models = {"whisper-turbo": lambda x: run_whisper(x, None),
              "whisper-turbo+vocab": lambda x: run_whisper(x, VOCAB)}
    if parakeet:
        models["parakeet-tdt-0.6b"] = run_parakeet

    # Segment once on the cleaned audio (that's what the VAD hears in the spike), reuse for every config.
    seg_sets = {"fragments(0.2s)": segments(fronts["rnnoise+gain"], 0.2),
                "turns(0.8s)": segments(fronts["rnnoise+gain"], 0.8)}
    rows = []
    for front, pcm in fronts.items():
        x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        for seg_name, segs in seg_sets.items():
            for model_name, fn in models.items():
                t = time.time()
                text = " ".join(fn(x[a:b]) for a, b in segs)
                ms = (time.time() - t) * 1000 / max(1, len(segs))
                rows.append((wer(ref, words(text)), front, seg_name, model_name, ms, text))

    rows.sort(key=lambda r: r[0])
    print(f"\n{len(seg_sets['fragments(0.2s)'])} fragments / {len(seg_sets['turns(0.8s)'])} turns detected\n")
    print(f"{'WER':>6}  {'front-end':13} {'segmentation':16} {'model':20} {'ms/seg':>7}")
    for w, front, seg, model, ms, _ in rows:
        mark = "  ← current spike" if (front, seg, model) == ("rnnoise+gain", "fragments(0.2s)", "whisper-turbo") else ""
        print(f"{w:6.1%}  {front:13} {seg:16} {model:20} {ms:7.0f}{mark}")
    print("\nBest:    ", rows[0][5].strip())
    cur = next(r for r in rows if r[1:4] == ("rnnoise+gain", "fragments(0.2s)", "whisper-turbo"))
    print("Current: ", cur[5].strip())
    print("\nNothing was written to disk.")


if __name__ == "__main__":
    spike._ensure_cuda12_libs()
    main()
