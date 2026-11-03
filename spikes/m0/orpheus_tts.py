"""Orpheus TTS (3B, GPU) as a Pipecat TTS service.

Orpheus is a Llama-3.2-3B that speaks in SNAC audio codes: 7 tokens per 85 ms frame of 24 kHz audio. A local
vLLM server generates the tokens (streamed); the 24 kHz SNAC decoder turns them into audio on the GPU.
Supports inline emotion tags: <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

ORPHEUS_MODEL = "unsloth/orpheus-3b-0.1-ft"  # public mirror of canopylabs/orpheus-3b-0.1-ft
FEMALE_VOICES = ["tara", "leah", "jess", "mia", "zoe"]
EMOTION_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]
PORT = int(os.environ.get("PASSE_ORPHEUS_PORT", "18100"))  # away from common dev-server ports (8000–8100)
BASE_URL = f"http://127.0.0.1:{PORT}/v1"

_START = [128259]  # start of human turn
_END = [128009, 128260, 128261, 128257]  # end of text, end of human turn, start of AI turn, start of speech
_STOP = 128258  # end of speech
_TOKEN = re.compile(r"<custom_token_(\d+)>")
_SR = 24000
_FRAME = 2048  # samples per SNAC frame (7 tokens)


def audio_token_budget(text: str) -> int:
    """Enough audio tokens for the text, with room to spare: ~86 tokens per second of speech, and slow, warm
    delivery runs at ~10 characters a second, so allow ~12 tokens per character (a fixed 1600 cut long replies off
    at 19.5 s)."""
    return min(7000, 400 + 12 * len(text))


def server_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=1) as r:
            return r.status == 200 and ORPHEUS_MODEL in r.read().decode(errors="replace")  # really Orpheus
    except OSError:
        return False


def launch_server(gpu_fraction: float = 0.3) -> subprocess.Popen | None:
    """Start vLLM for Orpheus in the background (its own .venv-vllm, so its torch can't clash with the spike's).

    Returns immediately; call wait_ready() before first use. Returns None if a server is already up."""
    if server_up():
        return None
    vllm = Path(__file__).parent / ".venv-vllm" / "bin" / "vllm"
    if not vllm.exists():
        raise SystemExit("Orpheus needs vLLM: cd spikes/m0 && uv venv --python 3.12 .venv-vllm && "
                         "VIRTUAL_ENV=.venv-vllm uv pip install vllm")
    proc = subprocess.Popen(
        [str(vllm), "serve", ORPHEUS_MODEL, "--port", str(PORT), "--gpu-memory-utilization", str(gpu_fraction),
         "--max-model-len", "8192", "--dtype", "bfloat16"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
    )
    # vLLM logs every request. Drain its output continuously, or the pipe fills up and the server blocks.
    import collections
    import threading

    proc.log_tail = collections.deque(maxlen=200)

    def drain():
        for line in proc.stderr:
            proc.log_tail.append(line.decode(errors="replace").rstrip())

    threading.Thread(target=drain, daemon=True).start()
    return proc


def wait_ready(proc: subprocess.Popen | None) -> None:
    t0 = time.monotonic()
    while not server_up():
        if proc is not None and proc.poll() is not None:
            time.sleep(0.5)  # let the drain thread catch the last lines
            err = "\n".join(list(getattr(proc, "log_tail", []))[-30:])
            raise SystemExit(f"Orpheus server exited:\n{err}")
        print(f"  Orpheus booting… {time.monotonic() - t0:.0f}s", end="\r", flush=True)
        time.sleep(1)
    if time.monotonic() - t0 > 1:
        print(f"  Orpheus ready (waited {time.monotonic() - t0:.0f}s)          ")


def stop_server(proc: subprocess.Popen | None) -> None:
    """Stop the server: ours via its process group, or one left over from an earlier run."""
    import signal

    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    subprocess.run(["pkill", "-INT", "-f", f"vllm serve {ORPHEUS_MODEL}"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def start_server(gpu_fraction: float = 0.3) -> subprocess.Popen | None:
    print(f"Starting Orpheus on vLLM (~{gpu_fraction * 32:.0f} GB VRAM)…", flush=True)
    proc = launch_server(gpu_fraction)
    wait_ready(proc)
    return proc


class OrpheusTTSService(TTSService):
    Settings = TTSSettings
    _settings: TTSSettings

    def __init__(self, *, base_url: str = BASE_URL, temperature: float = 0.6, top_p: float = 0.8,
                 repetition_penalty: float = 1.1, settings: TTSSettings | None = None, **kwargs):
        default = TTSSettings(model=ORPHEUS_MODEL, voice="tara", language=Language.EN)
        if settings is not None:
            default.apply_update(settings)
        super().__init__(push_start_frame=True, push_stop_frames=True, settings=default, **kwargs)
        import torch
        from huggingface_hub import hf_hub_download
        from openai import AsyncOpenAI
        from snac import SNAC
        from tokenizers import Tokenizer

        self._torch = torch
        self._tok = Tokenizer.from_file(hf_hub_download(ORPHEUS_MODEL, "tokenizer.json"))
        self._snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
        self._client = AsyncOpenAI(base_url=base_url, api_key="local")
        self._sampling = {"temperature": temperature, "top_p": top_p}
        self._extra = {"repetition_penalty": repetition_penalty, "stop_token_ids": [_STOP],
                       "skip_special_tokens": False}
        self._resampler = create_stream_resampler()
        self.fx = None  # optional voice_fx(...) applied to every chunk

    def _decode(self, codes: list[int]):
        """SNAC-decode whole frames of 7 codes → float audio (1, 1, frames * 2048), or None if invalid."""
        torch = self._torch
        n = len(codes) // 7
        c0, c1, c2 = [], [], []
        for j in range(n):
            f = codes[7 * j : 7 * j + 7]
            c0.append(f[0])
            c1 += [f[1], f[4]]
            c2 += [f[2], f[3], f[5], f[6]]
        layers = [torch.tensor(c, dtype=torch.int32, device="cuda").unsqueeze(0) for c in (c0, c1, c2)]
        if any(bool(((t < 0) | (t > 4096)).any()) for t in layers):
            return None
        with torch.inference_mode():
            return self._snac.decode(layers)

    async def _frame(self, audio, start: int, end: int, context_id: str) -> TTSAudioRawFrame:
        x = audio[0, 0, start:end].float().cpu().numpy()
        if self.fx is not None:
            x = self.fx(x)
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
        pcm = await self._resampler.resample(pcm, _SR, self.sample_rate)
        return TTSAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            voice = self._settings.voice or "tara"
            ids = _START + self._tok.encode(f"{voice}: {text}").ids + _END
            stream = await self._client.completions.create(
                model=ORPHEUS_MODEL, prompt=ids, max_tokens=audio_token_budget(text), stream=True,
                extra_body=self._extra,
                **self._sampling,
            )
            codes: list[int] = []
            async for chunk in stream:
                for m in _TOKEN.finditer(chunk.choices[0].text if chunk.choices else ""):
                    code = int(m.group(1)) - 10 - (len(codes) % 7) * 4096
                    if code <= 0:
                        continue
                    codes.append(code)
                    # Sliding window of 4 frames; emit the 2nd. Context on both sides keeps frame edges clean.
                    if len(codes) % 7 == 0 and len(codes) >= 28:
                        audio = await asyncio.to_thread(self._decode, codes[-28:])
                        if audio is not None:
                            await self.stop_ttfb_metrics()
                            yield await self._frame(audio, _FRAME, 2 * _FRAME, context_id)
            # The window lags two frames behind; flush them instead of clipping the end of every sentence.
            whole = len(codes) // 7 * 7
            if whole >= 28:
                audio = await asyncio.to_thread(self._decode, codes[whole - 28 : whole])
                if audio is not None:
                    yield await self._frame(audio, 2 * _FRAME, 4 * _FRAME, context_id)
            elif whole:
                audio = await asyncio.to_thread(self._decode, codes[:whole])
                if audio is not None:
                    yield await self._frame(audio, 0, whole // 7 * _FRAME, context_id)
        except Exception as e:
            logger.exception("Orpheus TTS failed")
            yield ErrorFrame(error=f"Orpheus TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()


if __name__ == "__main__":
    # Quick check: python orpheus_tts.py "Hello there <laugh> this is Orpheus." [voice]
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    print("server up:", server_up(), file=sys.stderr)


def synthesize(text: str, voice: str = "tara") -> np.ndarray:
    """Whole utterance → float32 24 kHz audio, outside a pipeline (for auditions)."""

    async def run():
        svc = OrpheusTTSService(settings=TTSSettings(voice=voice))
        svc._sample_rate = _SR

        class Passthrough:
            async def resample(self, audio, _in, _out):
                return audio

        async def noop(*_a, **_k):
            pass

        svc._resampler = Passthrough()
        svc.start_tts_usage_metrics = svc.stop_ttfb_metrics = noop
        chunks = [f.audio async for f in svc.run_tts(text, "audition") if isinstance(f, TTSAudioRawFrame)]
        return np.frombuffer(b"".join(chunks), np.int16).astype(np.float32) / 32768

    return asyncio.run(run())


_HUMS = [  # spelled-out murmurs Orpheus drones → something it performs well, keeping the meaning
    (re.compile(r"(?<![\w<])(?:mhm+|m+-hm+|uh-?huh)\b[\s.…,!?]*", re.I), "Yeah, "),  # agreeing
    (re.compile(r"(?<![\w<])hm{2,}\b[\s.…,!?]*", re.I), "Well… "),  # thinking
    (re.compile(r"(?<![\w<])m{2,}h*\b[\s.…,!?]*", re.I), "<sigh> "),  # pleased / contented
]


class _HumSub:
    def sub(self, _repl, text: str) -> str:  # same call shape as re.sub so callers stay simple
        for rx, repl in _HUMS:
            text = rx.sub(repl, text)
        return text


HUM = _HumSub()


_TAG_NAMES = "|".join(t[1:-1] for t in EMOTION_TAGS)
_BROKEN_TAG = re.compile(r"<\s*(?:" + _TAG_NAMES + r")?\w*(?![\w>])|<(?!(?:" + _TAG_NAMES + r")>)[^>\s]*>?", re.I)
_TAG_RUN = re.compile(r"((<(?:" + _TAG_NAMES + r")>)\s*)\2(\s*\2)+", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def speakable(text: str) -> str:
    """Make text safe to perform: lists become one flowing sentence, broken/unknown tags go, sound runs cap at 2."""
    lines = [_BULLET.sub("", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    joined = ""
    for ln in lines:
        if joined and joined[-1] not in ".!?…:,;":
            joined += ","
        joined += (" " if joined else "") + ln
    joined = _BROKEN_TAG.sub(lambda m: m.group(0) if m.group(0).lower() in EMOTION_TAGS else "", joined)
    return re.sub(r"\s{2,}", " ", _TAG_RUN.sub(r"\2 \2", joined)).strip()


def emotion_safe_markdown_filter():
    """Pipecat's markdown filter treats <laugh> as HTML and strips it; protect Orpheus's tags around it."""
    from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

    tags = {t: f"⟦{t[1:-1]}⟧" for t in EMOTION_TAGS}

    class EmotionSafeMarkdownFilter(MarkdownTextFilter):
        async def filter(self, text: str) -> str:
            text = speakable(text)
            # Orpheus drones spelled-out hums ("Mm…", "mmmh", "mhm"); make them a performed sigh instead.
            text = HUM.sub("<sigh> ", text)
            for tag, safe in tags.items():
                text = text.replace(tag, safe)
            text = await super().filter(text)
            for tag, safe in tags.items():
                text = text.replace(safe, tag)
            return text

    return EmotionSafeMarkdownFilter()


def paragraph_aggregator(max_chars: int = 400, **kwargs):
    """Hold whole sentences until ~a paragraph (or the end of the reply), then synthesize them in one go.

    Orpheus samples each request independently, so sentence-by-sentence synthesis drifts in pitch, energy and
    pace between sentences. Measured on a five-sentence reply (consistency_eval.py): one generation cut pitch
    drift from 33 to 18 Hz and loudness drift from 3.3 to 1.9 dB. The cost is waiting for the LLM to finish the
    reply (or the paragraph) before speaking; max_chars caps that for long explanations."""
    from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
    from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

    class ParagraphTextAggregator(SimpleTextAggregator):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._held = ""

        async def _check_sentence_with_lookahead(self, char, *, protected_end=0):
            done = await super()._check_sentence_with_lookahead(char, protected_end=protected_end)
            if done is None:
                return None
            self._held = f"{self._held} {done.text}".strip()
            if len(self._held) < max_chars:
                return None
            out, self._held = self._held, ""
            return Aggregation(text=out, type=done.type)

        async def flush(self):
            held, self._held = self._held, ""  # take it first: the base flush() calls reset()
            rest = await super().flush()
            text = f"{held} {rest.text if rest else ''}".strip()
            return Aggregation(text=text, type=TextType.SENTENCE) if text else None

        async def handle_interruption(self):
            self._held = ""
            await super().handle_interruption()

        async def reset(self):
            self._held = ""
            await super().reset()

    return ParagraphTextAggregator(**kwargs)


def voice_fx(style: str, pitch_semitones: float = 0.0):
    """Post-processing for a style. Returns process(float32 audio) -> float32 audio, or None for 'natural'.

    Live use keeps to zero-latency filters. Pitch shifting (rubberband) needs ~1 s of lookahead when streaming,
    so it's only applied to whole clips (auditions) for now."""
    if style == "natural" and not pitch_semitones:
        return None
    from pedalboard import Compressor, Gain, HighShelfFilter, LowShelfFilter, Pedalboard, PeakFilter, PitchShift

    chain = []
    if pitch_semitones:
        chain.append(PitchShift(semitones=pitch_semitones))
    if style == "warm":
        chain += [
            LowShelfFilter(cutoff_frequency_hz=200, gain_db=3.5),  # chest and warmth
            PeakFilter(cutoff_frequency_hz=3200, gain_db=-2.0, q=0.9),  # less forward, less "announcer"
            HighShelfFilter(cutoff_frequency_hz=6500, gain_db=-4.5),  # softer s-sounds and edge
            Compressor(threshold_db=-24, ratio=2.5, attack_ms=8, release_ms=120),  # close-mic intimacy
            Gain(gain_db=5.0),  # back to the natural loudness after the compressor
        ]
    board = Pedalboard(chain)

    def process(audio: np.ndarray, whole_clip: bool = False) -> np.ndarray:
        return board.process(audio.astype(np.float32), _SR, reset=whole_clip)

    return process
