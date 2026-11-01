"""Passe M0 spike: a local voice loop with a live dashboard.

Mic → Silero VAD → smart-turn v3 → faster-whisper (GPU) → Baseten LLM → Kokoro → headphones.

Shows turn-taking latency, interruptions, token use/cost and GPU load while you chat.
Writes nothing to disk: no logs, recordings or transcripts. Everything lives in memory and is gone on exit.

    uv run spike.py                 # chat; Ctrl+C to quit and print a summary
    uv run spike.py --list-devices  # pick --input-device / --output-device
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


def _ensure_cuda12_libs() -> None:
    """CTranslate2 needs CUDA 12 cuBLAS/cuDNN; the venv ships them. Re-exec with them on the loader path."""
    import nvidia.cublas
    import nvidia.cudnn

    libs = [str(Path(p) / "lib") for m in (nvidia.cublas, nvidia.cudnn) for p in m.__path__]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(lib in current.split(":") for lib in libs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([current] if current else []))
    os.execv(sys.executable, [sys.executable, *sys.argv])


SYSTEM_PROMPT = """You are Passe, a voice companion for a software developer. This is a spoken conversation:
your replies are converted to speech.

- Keep every reply to one or two short sentences unless asked for more.
- Plain spoken English only: never use asterisks, markdown, lists, headings, code blocks, emoji or URLs.
- You can chat about anything, and you are especially good at helping shape a prompt for a coding agent:
  ask one clarifying question at a time and summarise changes briefly.
- If you get interrupted, don't repeat yourself; respond to what the user just said.

Tools (all dummies in this test, but treat them as real):
- Before a tool that takes a moment, say a few words about what you're doing ("Checking git.").
- After a tool returns, give the outcome in one sentence. Don't read raw output.
- run_tests runs in the background: say you've started it and keep talking with the user. When its result
  arrives later, tell the user right away.
- Keep the prompt you're shaping with the user in update_draft. Never read the draft aloud; say what changed.
- Only call finalize_prompt after the user clearly approves the draft (e.g. "let's use that").
- If you say you'll do something with a tool, call the tool in that same reply. Don't announce and wait.
- set_voice: call it straight away without saying anything first (an "Okay." is played for you). After it
  returns, your whole reply is a short check in the new voice, like "Is this better?"."""


# ---------------------------------------------------------------------------------------------------------
# In-memory stats


@dataclass
class Turn:
    n: int
    user_text: str = ""
    bot_text: str = ""
    total_s: float | None = None  # user stopped talking → bot started speaking
    end_of_turn_s: float | None = None  # user stopped talking → turn released (VAD + smart-turn + STT)
    stt_s: float | None = None
    llm_s: float | None = None
    tts_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    interrupted: bool = False


@dataclass
class Stats:
    started: float = field(default_factory=time.monotonic)
    state: str = "starting"
    turns: list[Turn] = field(default_factory=list)
    transcript: deque = field(default_factory=lambda: deque(maxlen=10))
    live_bot: str = ""
    interruptions: int = 0
    interrupt_reaction_ms: list[float] = field(default_factory=list)
    backchannels: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    llm_calls: int = 0
    tts_chars: int = 0
    gpu: dict = field(default_factory=dict)
    gpu_util_samples: list[float] = field(default_factory=list)
    gpu_util_active: list[float] = field(default_factory=list)
    vram_peak_mb: float = 0.0
    proc_vram_peak_mb: float = 0.0
    cpu_pct: float = 0.0
    mic_dbfs: float = -120.0
    mic_volume: float = 0.0
    voice_rejected: int = 0
    voice_last_sim: float | None = None
    tool_calls: list = field(default_factory=list)
    draft: str = ""
    draft_version: int = 0
    draft_sent: bool = False
    voice: str = ""
    recording_since: float | None = None
    recording_file: str | None = None
    tts_cps: float = 15.0  # characters per second of speech, measured at startup
    response_full: str = ""
    response_speech_start: float | None = None
    last_barge: float | None = None
    warnings: deque = field(default_factory=lambda: deque(maxlen=4))
    _bot_speaking: bool = False
    _barge_t: float | None = None

    def turn(self) -> Turn:
        if not self.turns:
            self.turns.append(Turn(n=1))
        return self.turns[-1]

    def new_turn(self) -> Turn:
        t = Turn(n=len(self.turns) + 1)
        self.turns.append(t)
        return t

    def cost(self, price_in: float, price_out: float) -> float:
        return (self.prompt_tokens * price_in + self.completion_tokens * price_out) / 1e6


# ---------------------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Passe M0 voice spike")
    p.add_argument("--model", default="openai/gpt-oss-120b", help="Baseten model id")
    p.add_argument("--reasoning", default="low", help="reasoning_effort (low|medium|high|none to omit)")
    p.add_argument("--price-in", type=float, default=0.10, help="$ per 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.50, help="$ per 1M output tokens")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice")
    p.add_argument("--speed", type=float, default=1.15, help="speaking rate (1.0 = Kokoro default, 0.5–2.0)")
    p.add_argument("--output-latency", type=float, default=0.2,
                   help="seconds between playback and your ears (Bluetooth ≈ 0.2) for the interruption estimate")
    p.add_argument("--tts-device", choices=["cuda", "cpu"], default="cuda", help="where Kokoro runs")
    p.add_argument("--stt-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    p.add_argument("--vad-stop", type=float, default=0.2, help="VAD stop_secs (smart-turn decides the rest)")
    p.add_argument("--min-volume", type=float, default=0.6, help="VAD volume gate 0–1 (lower for quiet mics)")
    p.add_argument("--vad-confidence", type=float, default=0.7, help="VAD speech confidence threshold 0–1")
    p.add_argument("--mic-test", action="store_true", help="show live mic level + VAD confidence, no chat")
    p.add_argument("--seconds", type=float, default=0, help="with --mic-test: run N seconds, print per-second lines")
    p.add_argument("--denoise", choices=["rnnoise", "none"], default="rnnoise", help="input noise suppression")
    p.add_argument("--gain-db", type=float, default=12.0, help="digital mic gain after denoising (dB)")
    p.add_argument("--calibrate", action="store_true", help="measure quiet / you / background speech, suggest flags")
    p.add_argument("--voice-lock", action="store_true", help="enroll your voice at start; drop turns that aren't you")
    p.add_argument("--voice-threshold", type=float, default=0.6, help="voice-lock similarity threshold 0–1")
    p.add_argument("--enroll-secs", type=float, default=10.0, help="seconds of speech to record for voice lock")
    p.add_argument(
        "--interrupt",
        choices=["vad", "words"],
        default="vad",
        help="vad: any speech interrupts at once. words: need 3+ transcribed words, which ignores 'mhm' but"
        " (with segmented Whisper) only takes effect once you pause",
    )
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--no-dashboard", action="store_true", help="plain Pipecat logs instead of the dashboard")
    return p.parse_args()


def list_devices() -> None:
    import pyaudio

    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        io = ("in " if d["maxInputChannels"] else "   ") + ("out" if d["maxOutputChannels"] else "   ")
        print(f"{i:3}  {io}  {d['name']}")
    pa.terminate()


def load_api_key() -> str:
    if key := os.environ.get("BASETEN_API_KEY"):
        return key.strip()
    path = Path(os.environ.get("BASETEN_API_KEY_FILE", "~/baseten_apikey")).expanduser()
    return path.read_text().strip()


# ---------------------------------------------------------------------------------------------------------
# GPU / CPU sampling


async def sample_hardware(stats: Stats) -> None:
    import psutil
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    me = psutil.Process(os.getpid())
    me.cpu_percent(None)
    pid = os.getpid()
    while True:
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        proc_mb = 0.0
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory:
                proc_mb = p.usedGpuMemory / 2**20
        stats.gpu = {
            "util": util.gpu,
            "vram_mb": mem.used / 2**20,
            "vram_total_mb": mem.total / 2**20,
            "proc_mb": proc_mb,
            "power_w": power,
            "temp_c": temp,
        }
        stats.gpu_util_samples.append(util.gpu)
        if stats.state in ("you", "thinking", "speaking"):
            stats.gpu_util_active.append(util.gpu)
        stats.vram_peak_mb = max(stats.vram_peak_mb, mem.used / 2**20)
        stats.proc_vram_peak_mb = max(stats.proc_vram_peak_mb, proc_mb)
        stats.cpu_pct = me.cpu_percent(None)
        await asyncio.sleep(0.25)


# ---------------------------------------------------------------------------------------------------------
# Dashboard


STATE_STYLE = {
    "starting": ("STARTING", "yellow"),
    "listening": ("● LISTENING", "green"),
    "you": ("● YOU'RE TALKING", "bold green"),
    "thinking": ("◌ THINKING", "yellow"),
    "speaking": ("▶ SPEAKING", "cyan"),
}


def fmt_s(v: float | None) -> str:
    return "—" if v is None else f"{v * 1000:.0f}"


def render(stats: Stats, args: argparse.Namespace):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    label, style = STATE_STYLE.get(stats.state, (stats.state, "white"))
    up = time.monotonic() - stats.started
    gate_ok = stats.mic_volume >= args.min_volume
    head = Text.assemble(
        (f" {label} ", f"reverse {style}"),
        "  mic ",
        (meter(stats.mic_volume), "green" if gate_ok else "dim"),
        (f" vol {stats.mic_volume:.2f}/{args.min_volume:.2f} {stats.mic_dbfs:5.0f} dBFS", "" if gate_ok else "dim"),
        "\n",
        f"   {args.model} · reasoning {args.reasoning} · tts {args.tts_device} · interrupt={args.interrupt}"
        f" · mic {args.denoise}+{args.gain_db:g}dB · voice {stats.voice or args.voice} {args.speed:g}x"
        f" · up {up:,.0f}s",
        (f"   ● REC {time.monotonic() - stats.recording_since:,.0f}s" if stats.recording_since else "", "bold red"),
    )

    convo = Text()
    for who, line, *rest in stats.transcript:
        who_style = {"you": "bold green", "other": "dim red"}.get(who, "bold cyan")
        convo.append(f"{who:>5}  ", style=who_style)
        convo.append(line, style="dim" if who == "other" else "")
        if rest:
            convo.append(" ⟂ ", style="bold magenta")
            convo.append(rest[0], style="dim strike")
        convo.append("\n")
    if stats.live_bot:
        convo.append("passe  ", style="bold cyan")
        convo.append(stats.live_bot + "\n", style="dim")

    t = Table(expand=True, box=None, header_style="bold", pad_edge=False)
    for col in ("#", "total", "end-of-turn", "stt", "llm", "tts", "tok in", "out", "reason", ""):
        t.add_column(col, justify="right" if col not in ("#", "") else "left")
    for tr in stats.turns[-8:]:
        total_style = "" if tr.total_s is None else ("green" if tr.total_s < 1.0 else "yellow" if tr.total_s < 1.5 else "red")
        t.add_row(
            str(tr.n),
            Text(fmt_s(tr.total_s), style=total_style),
            fmt_s(tr.end_of_turn_s),
            fmt_s(tr.stt_s),
            fmt_s(tr.llm_s),
            fmt_s(tr.tts_s),
            str(tr.prompt_tokens or "—"),
            str(tr.completion_tokens or "—"),
            str(tr.reasoning_tokens or "—"),
            Text("interrupted", style="magenta") if tr.interrupted else "",
        )

    totals = [tr.total_s for tr in stats.turns if tr.total_s is not None]
    react = stats.interrupt_reaction_ms
    g = stats.gpu or {}
    avg_util = sum(stats.gpu_util_samples) / len(stats.gpu_util_samples) if stats.gpu_util_samples else 0
    avg_active = sum(stats.gpu_util_active) / len(stats.gpu_util_active) if stats.gpu_util_active else 0
    summary = Table.grid(expand=True, padding=(0, 2))
    summary.add_column()
    summary.add_column()
    summary.add_column()
    summary.add_row(
        Text.assemble(
            ("Turn-taking\n", "bold"),
            f"turns {len(totals)} · median {fmt_s(median(totals))} ms · p90 {fmt_s(pct(totals, 0.9))} ms\n",
            f"interruptions {stats.interruptions} · stop reaction "
            f"{('%.0f ms' % median(react)) if react else '—'} (pipeline; Bluetooth adds ~150–250)\n",
            f"backchannels ignored {stats.backchannels}",
            (
                f"\nvoice lock ≥{args.voice_threshold:.2f} · rejected {stats.voice_rejected} · last match "
                + ("—" if stats.voice_last_sim is None else f"{stats.voice_last_sim:.2f}")
                if args.voice_lock
                else ""
            ),
        ),
        Text.assemble(
            ("Tokens & cost\n", "bold"),
            f"calls {stats.llm_calls} · in {stats.prompt_tokens:,} (cached {stats.cached_tokens:,})\n",
            f"out {stats.completion_tokens:,} (reasoning {stats.reasoning_tokens:,})\n",
            f"session ${stats.cost(args.price_in, args.price_out):.5f} · tts chars {stats.tts_chars:,}",
        ),
        Text.assemble(
            ("GPU (RTX 5090) · CPU\n", "bold"),
            f"util {g.get('util', 0):>3}% · avg {avg_util:.0f}% · avg while active {avg_active:.0f}%\n",
            f"VRAM {g.get('vram_mb', 0) / 1024:.1f}/{g.get('vram_total_mb', 0) / 1024:.0f} GB · "
            f"spike {g.get('proc_mb', 0) / 1024:.2f} GB (peak {stats.proc_vram_peak_mb / 1024:.2f})\n",
            f"{g.get('power_w', 0):.0f} W · {g.get('temp_c', 0)}°C · spike CPU {stats.cpu_pct:.0f}%",
        ),
    )

    tl = Text()
    for c in stats.tool_calls[-6:]:
        args_s = ", ".join(f"{k}={v!r}" for k, v in c["args"].items() if k != "text")[:40]
        if c["end"] is None:
            tl.append("⏳ ", style="yellow")
            tl.append(f"{c['name']}({args_s})", style="bold")
            tl.append(f"  {time.monotonic() - c['start']:.1f}s…\n", style="yellow")
        else:
            tl.append("✓ ", style="green")
            tl.append(f"{c['name']}({args_s})", style="bold")
            tl.append(f"  {c['end'] - c['start']:.1f}s  ", style="dim")
            tl.append(f"{c['result']}\n")
    draft = Text(stats.draft or "(no draft yet)", style="" if stats.draft else "dim")
    draft_title = f"prompt to send · v{stats.draft_version}" + (" · SENT (dry run)" if stats.draft_sent else "")
    tools_row = Table.grid(expand=True, padding=(0, 1))
    tools_row.add_column(ratio=1)
    tools_row.add_column(ratio=1)
    tools_row.add_row(
        Panel(tl or Text("(no tool calls yet)", style="dim"), title="tools", border_style="dim", height=8),
        Panel(draft, title=draft_title, border_style="green" if stats.draft_sent else "dim", height=8),
    )

    parts = [
        Panel(head, border_style=style),
        Panel(convo, title="conversation", border_style="dim", height=13),
        tools_row,
        Panel(t, title="per turn (ms)", border_style="dim"),
        Panel(summary, border_style="dim"),
    ]
    if stats.warnings:
        parts.append(Text("\n".join(stats.warnings), style="yellow"))
    parts.append(Text(" Ctrl+C to quit · nothing is saved", style="dim"))
    return Group(*parts)


def make_input_filter(args: argparse.Namespace):
    """Denoise (RNNoise) then digital gain, applied to mic audio before VAD and STT."""
    import numpy as np
    from pipecat.audio.filters.base_audio_filter import BaseAudioFilter

    class GainFilter(BaseAudioFilter):
        def __init__(self, db: float):
            self._g = 10 ** (db / 20)

        async def start(self, sample_rate: int):
            pass

        async def stop(self):
            pass

        async def process_frame(self, frame):
            pass

        async def filter(self, audio: bytes) -> bytes:
            x = np.frombuffer(audio, np.int16).astype(np.float32) * self._g
            return np.clip(x, -32768, 32767).astype(np.int16).tobytes()

    class ChainFilter(BaseAudioFilter):
        def __init__(self, filters):
            self._filters = filters

        async def start(self, sample_rate: int):
            for f in self._filters:
                await f.start(sample_rate)

        async def stop(self):
            for f in self._filters:
                await f.stop()

        async def process_frame(self, frame):
            for f in self._filters:
                await f.process_frame(frame)

        async def filter(self, audio: bytes) -> bytes:
            for f in self._filters:
                audio = await f.filter(audio)
            return audio

    chain = []
    if args.denoise == "rnnoise":
        from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter

        chain.append(RNNoiseFilter())
    if args.gain_db:
        chain.append(GainFilter(args.gain_db))
    return ChainFilter(chain) if chain else None


def setup_tools(llm, stats: Stats):
    """Dummy tools with realistic latency, spoken lead-ins and a live tool/draft view. Nothing touches the system."""
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
    from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.transcriptions.language import Language

    voices = {"american": ("af_heart", Language.EN_US), "british": ("bf_emma", Language.EN_GB)}

    lead_ins = {
        "set_voice": "Okay.",
        "stop_screen_recording": "Stopping the recording and saving it.",
        "start_screen_recording": "Starting the recording.",
        "git_status": "Checking git.",
        "search_session": "Let me look through the session.",
        "run_tests": "Starting the tests.",
    }

    def track(params) -> dict:
        call = {"name": params.function_name, "args": dict(params.arguments or {}), "start": time.monotonic(),
                "end": None, "result": None}
        stats.tool_calls.append(call)
        return call

    async def done(params, call: dict, result: dict, summary: str):
        call["end"], call["result"] = time.monotonic(), summary
        await params.result_callback(result)

    async def git_status(params):
        call = track(params)
        await asyncio.sleep(0.4)
        await done(params, call, {"branch": "feature/uploader-retry", "modified": ["upload.py", "retry.py",
                   "tests/test_upload.py"], "ahead_of_main": 2}, "feature/uploader-retry · 3 modified · 2 ahead")

    async def search_session(params):
        call = track(params)
        await asyncio.sleep(1.5)
        await done(params, call, {"matches": [
            "Agent chose exponential backoff with jitter, base 0.5 s, max 5 attempts.",
            "Decided to only retry on network errors and 5xx, never on 4xx.",
            "test_upload_timeout has been flaky since the timeout was lowered to 2 s."]},
            "3 matches: backoff policy · retry scope · flaky timeout test")

    async def run_tests(params):
        call = track(params)
        await asyncio.sleep(6.0)
        await done(params, call, {"passed": 38, "failed": 2, "failures": [
            "test_upload_retry_backoff: expected 5 attempts, got 4",
            "test_upload_timeout: TimeoutError after 2.0 s"]}, "38 passed · 2 failed")

    async def set_volume(params):
        call = track(params)
        await done(params, call, {"ok": True, "volume_percent": params.arguments.get("percent"),
                   "note": "dummy, nothing changed"}, f"volume → {params.arguments.get('percent')}% (dummy)")

    async def recording_active() -> bool:
        proc = await asyncio.create_subprocess_exec("pgrep", "-f", "^gpu-screen-recorder",
                                                    stdout=asyncio.subprocess.DEVNULL)
        return await proc.wait() == 0

    async def start_screen_recording(params):
        call = track(params)
        if await recording_active():
            await done(params, call, {"ok": False, "error": "a recording is already running"}, "already recording")
            return
        args = ["omarchy", "capture", "screenrecording", "--fullscreen"]
        if params.arguments.get("with_audio", True):
            args += ["--with-desktop-audio", "--with-microphone-audio"]
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
        await proc.wait()
        if await recording_active():
            stats.recording_since = time.monotonic()
            stats.recording_file = Path("/tmp/omarchy-screenrecord-filename").read_text().strip() \
                if Path("/tmp/omarchy-screenrecord-filename").exists() else None
            await done(params, call, {"ok": True, "recording": "fullscreen", "audio": bool(len(args) > 4)},
                       "recording (fullscreen" + (" + audio)" if len(args) > 4 else ")"))
        else:
            await done(params, call, {"ok": False, "error": "recorder did not start"}, "failed to start")

    async def stop_screen_recording(params):
        call = track(params)
        if not await recording_active():
            await done(params, call, {"ok": False, "error": "no recording is running"}, "nothing to stop")
            return
        proc = await asyncio.create_subprocess_exec("omarchy", "capture", "screenrecording", "--stop-recording",
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
        out, _ = await proc.communicate()
        secs = time.monotonic() - stats.recording_since if stats.recording_since else None
        stats.recording_since = None
        path = (out.decode().strip().splitlines() or [stats.recording_file or ""])[-1]
        size = Path(path).stat().st_size / 2**20 if path and Path(path).exists() else None
        await done(params, call, {"ok": True, "file": Path(path).name if path else None,
                   "duration_secs": round(secs) if secs else None, "size_mb": round(size, 1) if size else None},
                   f"saved {Path(path).name if path else '?'}" + (f" · {size:.1f} MB" if size else ""))

    async def set_voice(params):
        call = track(params)
        accent = str(params.arguments.get("accent") or "").lower()
        if not accent:  # "change voices": swap to the other one
            accent = "british" if (stats.voice or "af_").startswith("af_") else "american"
        if accent not in voices:
            await done(params, call, {"ok": False, "error": f"unknown accent {accent!r}"}, f"unknown: {accent}")
            return
        voice, lang = voices[accent]
        # Goes downstream ahead of the confirmation, so the reply is already in the new voice.
        await params.llm.push_frame(TTSUpdateSettingsFrame(delta=KokoroTTSService.Settings(voice=voice, language=lang)))
        stats.voice = voice
        await done(params, call, {"ok": True, "voice": f"{accent} female"}, f"{accent} ({voice})")

    async def update_draft(params):
        call = track(params)
        stats.draft = params.arguments.get("text", "")
        stats.draft_version += 1
        stats.draft_sent = False
        await done(params, call, {"version": stats.draft_version}, f"draft v{stats.draft_version}")

    async def finalize_prompt(params):
        call = track(params)
        stats.draft_sent = True
        await done(params, call, {"delivered": "dry run: would paste into Claude Code (no Enter)",
                   "version": stats.draft_version}, f"sent v{stats.draft_version} (dry run)")

    schemas = [
        FunctionSchema("git_status", "Current branch and changed files of the user's repo.", {}, []),
        FunctionSchema("search_session", "Search the coding agent's session history for earlier decisions or facts.",
                       {"query": {"type": "string"}}, ["query"]),
        FunctionSchema("run_tests", "Run the project's test suite (takes a while; runs in the background).",
                       {"target": {"type": "string", "description": "test path or 'all'"}}, ["target"]),
        FunctionSchema("set_volume", "Set the system output volume.",
                       {"percent": {"type": "integer", "minimum": 0, "maximum": 100}}, ["percent"]),
        FunctionSchema("start_screen_recording",
                       "Start recording the full screen (with desktop + microphone audio by default). "
                       "Only when the user explicitly asks to record.",
                       {"with_audio": {"type": "boolean"}}, []),
        FunctionSchema("stop_screen_recording", "Stop the screen recording and save it to ~/Videos.", {}, []),
        FunctionSchema("set_voice", "Change your speaking voice (both female). Omit accent to swap to the other one.",
                       {"accent": {"type": "string", "enum": ["british", "american"]}}, []),
        FunctionSchema("update_draft", "Replace the prompt draft being prepared for the coding agent.",
                       {"text": {"type": "string", "description": "the full new draft"}}, ["text"]),
        FunctionSchema("finalize_prompt", "Deliver the current draft to the coding agent. Only after user approval.",
                       {}, []),
    ]
    handlers = {"git_status": git_status, "search_session": search_session, "run_tests": run_tests,
                "set_volume": set_volume, "update_draft": update_draft, "finalize_prompt": finalize_prompt,
                "start_screen_recording": start_screen_recording, "stop_screen_recording": stop_screen_recording,
                "set_voice": set_voice}
    for name, h in handlers.items():
        llm.register_function(name, h, cancel_on_interruption=(name != "run_tests"))

    @llm.event_handler("on_function_calls_started")
    async def _lead_in(service, function_calls):
        # The model often calls tools without saying anything first; fill the silence ourselves.
        if stats.response_full.strip():
            return
        for fc in function_calls:
            if fc.function_name in lead_ins:
                await service.push_frame(TTSSpeakFrame(lead_ins[fc.function_name]))
                break

    return ToolsSchema(standard_tools=schemas)


SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "wespeaker_en_voxceleb_resnet34.onnx"
)
SPEAKER_MODEL = Path("~/.cache/passe-spike/wespeaker_en_voxceleb_resnet34.onnx").expanduser()


class VoicePrint:
    """Speaker embeddings (WeSpeaker ResNet34 via sherpa-onnx, CPU, ~30 ms per utterance)."""

    def __init__(self):
        import sherpa_onnx

        if not SPEAKER_MODEL.exists():
            import urllib.request

            SPEAKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading speaker model (26 MB)…", flush=True)
            urllib.request.urlretrieve(SPEAKER_MODEL_URL, SPEAKER_MODEL)
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(SPEAKER_MODEL), num_threads=2)
        )

    def embed(self, pcm: bytes, sample_rate: int = 16000):
        import numpy as np

        x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        st = self._ex.create_stream()
        st.accept_waveform(sample_rate, x)
        st.input_finished()
        e = np.array(self._ex.compute(st))
        return e / (np.linalg.norm(e) + 1e-9)


async def capture(args: argparse.Namespace, seconds: float):
    """Yield (filtered_pcm, filtered_volume, voice_confidence) per 32 ms frame from the mic, via the input chain."""
    import numpy as np
    import pyaudio
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.volume import AudioVolumeTracker

    sr = 16000
    probe = SileroVADAnalyzer()
    probe.set_sample_rate(sr)
    n = probe.num_frames_required()
    chain = make_input_filter(args)
    if chain:
        await chain.start(sr)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=n,
                     input_device_index=args.input_device)
    tracker = AudioVolumeTracker()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            raw = stream.read(n, False)
            out = await chain.filter(raw) if chain else raw
            out = (out + b"\x00" * len(raw))[: len(raw)]
            tracker.update(out, sr)
            conf = float(np.asarray(probe.voice_confidence(out)).max())
            yield out, float(tracker.volume), conf
            await asyncio.sleep(0)
    finally:
        stream.close()
        pa.terminate()
        if chain:
            await chain.stop()


async def record_step(args: argparse.Namespace, title: str, seconds: float) -> list[tuple[bytes, float, float]]:
    print(f"\n▶ {title}")
    for i in (3, 2, 1):
        print(f"  starting in {i}…", end="\r", flush=True)
        await asyncio.sleep(1)
    frames, t0, last = [], time.monotonic(), -1
    async for f in capture(args, seconds):
        frames.append(f)
        left = int(seconds - (time.monotonic() - t0))
        if left != last:
            voiced = sum(1 for _, v, c in frames if c >= args.vad_confidence) * 0.032
            print(f"  ● recording… {left:2}s left · speech heard {voiced:4.1f}s   ", end="\r", flush=True)
            last = left
    print()
    return frames


def voiced_audio(args: argparse.Namespace, frames) -> tuple[bytes, list[float]]:
    sel = [(pcm, v) for pcm, v, c in frames if c >= args.vad_confidence and v > 0.05]
    return b"".join(p for p, _ in sel), [v for _, v in sel]


def chunk_sims(vp: VoicePrint, ref, pcm: bytes, secs: float = 2.5) -> list[float]:
    step = int(secs * 16000) * 2
    return [float(ref @ vp.embed(pcm[i : i + step])) for i in range(0, len(pcm) - step // 2, step)]


async def enroll_voice(args: argparse.Namespace, vp: VoicePrint):
    """Record your voice once at startup for the voice lock. Kept in memory only."""
    while True:
        frames = await record_step(
            args, f"Voice lock enrollment: talk normally for {args.enroll_secs:.0f}s (anything you like)",
            args.enroll_secs,
        )
        pcm, _ = voiced_audio(args, frames)
        if len(pcm) / 32000 >= 3:
            print(f"  enrolled from {len(pcm) / 32000:.1f}s of speech")
            return vp.embed(pcm)
        print(f"  only {len(pcm) / 32000:.1f}s of speech heard; need 3s. Again.")


def calibrate(args: argparse.Namespace) -> None:
    asyncio.run(_calibrate(args))


async def _calibrate(args: argparse.Namespace) -> None:
    import numpy as np

    vp = VoicePrint()
    print(f"Calibration · input chain {args.denoise}+{args.gain_db:g}dB · nothing is saved")
    quiet = await record_step(args, "1/3  Stay quiet (room noise)", 5)
    you = await record_step(args, "2/3  Talk normally, as you would to Passe", 15)
    bg = await record_step(
        args,
        "3/3  Stay quiet yourself. Let background speech happen: your partner talking in the other room,\n"
        "     or a podcast/video playing there. (Nothing playing is fine too.)",
        15,
    )

    noise = float(np.percentile([v for _, v, _ in quiet], 99))
    you_pcm, you_vols = voiced_audio(args, you)
    bg_pcm, bg_vols = voiced_audio(args, bg)
    if len(you_pcm) / 32000 < 4:
        print(f"\nOnly {len(you_pcm) / 32000:.1f}s of your speech was heard. Check the mic (--mic-test) and retry.")
        return
    split = int(len(you_pcm) * 0.6) // 2 * 2
    ref = vp.embed(you_pcm[:split])
    you_sims = chunk_sims(vp, ref, you_pcm[split:])
    bg_sims = chunk_sims(vp, ref, bg_pcm) if len(bg_pcm) / 32000 >= 1.5 else []
    you_p10, you_med = float(np.percentile(you_vols, 10)), float(np.median(you_vols))

    print("\nResults")
    print(f"  room noise        volume p99 {noise:.2f}")
    print(f"  you               volume p10 {you_p10:.2f} · median {you_med:.2f} · {len(you_pcm) / 32000:.1f}s speech")
    print(f"  you vs your voice match {min(you_sims):.2f}–{max(you_sims):.2f}" if you_sims else "  (not enough speech to test your match)")
    if bg_vols:
        bg_p95 = float(np.percentile(bg_vols, 95))
        print(f"  background speech volume p95 {bg_p95:.2f} · {len(bg_pcm) / 32000:.1f}s heard"
              + (f" · voice match {min(bg_sims):.2f}–{max(bg_sims):.2f}" if bg_sims else ""))
    else:
        bg_p95 = None
        print("  background speech none got through the input chain")

    notes = []
    if bg_p95 is not None and bg_p95 < you_p10:
        min_vol = (bg_p95 + you_p10) / 2
        notes.append("volume alone separates you from the background")
    else:
        min_vol = max(noise + 0.05, you_p10 - 0.1)
        if bg_p95 is not None:
            notes.append("background speech is as loud as you at the mic: volume can't separate it, use the voice lock")
    min_vol = round(min(0.9, max(0.3, min_vol)), 2)

    lo_you = min(you_sims) if you_sims else 0.7
    if bg_sims and max(bg_sims) < lo_you:
        thr = (max(bg_sims) + lo_you) / 2
        notes.append("voice lock cleanly separates you from the background")
    elif bg_sims:
        thr = lo_you - 0.05
        notes.append("some background speech sounds like you to the model; expect occasional misses")
    else:
        thr = lo_you - 0.15
    thr = round(min(0.8, max(0.4, thr)), 2)

    print("\nSuggested")
    for n in notes:
        print(f"  · {n}")
    print(f"\n  uv run spike.py --min-volume {min_vol} --voice-lock --voice-threshold {thr}\n")


def meter(v: float, width: int = 16) -> str:
    n = max(0, min(width, round(float(v) * width)))
    return "█" * n + "░" * (width - n)


def mic_test(args: argparse.Namespace) -> None:
    """Live mic level and VAD state through the same input chain as the chat, to tune the flags."""
    asyncio.run(_mic_test(args))


async def _mic_test(args: argparse.Namespace) -> None:
    import numpy as np
    import pyaudio
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams, VADState
    from pipecat.audio.volume import AudioVolumeTracker
    from rich.live import Live
    from rich.text import Text

    sr = 16000
    vad = SileroVADAnalyzer(
        params=VADParams(stop_secs=args.vad_stop, min_volume=args.min_volume, confidence=args.vad_confidence)
    )
    vad.set_sample_rate(sr)
    probe = SileroVADAnalyzer()  # separate instance: Silero is stateful, so don't query the gate's model twice
    probe.set_sample_rate(sr)
    n = vad.num_frames_required()
    chain = make_input_filter(args)
    if chain:
        await chain.start(sr)
    pa = pyaudio.PyAudio()
    idx = args.input_device if args.input_device is not None else pa.get_default_input_device_info()["index"]
    name = pa.get_device_info_by_index(idx)["name"]
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=n,
                     input_device_index=args.input_device)
    raw_t, out_t = AudioVolumeTracker(), AudioVolumeTracker()
    peak_raw = peak_out = peak_conf = 0.0
    speaking_secs = 0.0
    print(f"Mic test on [{idx}] {name} · chain {args.denoise}+{args.gain_db:g}dB. Talk; Ctrl+C to stop.")
    t0 = time.monotonic()
    sec: dict = {}
    timed = args.seconds > 0
    try:
        with (contextlib.nullcontext() if timed else Live(auto_refresh=False)) as live:
            while not timed or time.monotonic() - t0 < args.seconds:
                raw = stream.read(n, False)
                await asyncio.sleep(0)
                out = await chain.filter(raw) if chain else raw
                if len(out) < len(raw):  # filters may buffer; keep frames aligned for the VAD
                    out = out + b"\x00" * (len(raw) - len(out))
                raw_t.update(raw, sr)
                out_t.update(out, sr)
                conf = float(np.asarray(probe.voice_confidence(out[: n * 2])).max())
                state = await vad.analyze_audio(out[: n * 2])
                if state == VADState.SPEAKING:
                    speaking_secs += n / sr
                peak_raw = max(peak_raw, float(raw_t.volume))
                peak_out = max(peak_out, float(out_t.volume))
                peak_conf = max(peak_conf, conf)
                talking = state in (VADState.STARTING, VADState.SPEAKING)
                if timed:
                    k = int(time.monotonic() - t0)
                    if k not in sec:
                        if sec:
                            print(sec[max(sec)], flush=True)
                        sec[k] = ""
                    prev = sec[k]
                    pr, po, pc, ps = (map(float, prev.split()[2::2]) if prev else (0, 0, 0, 0))
                    sec[k] = (f"{k:2}s rawvol {max(pr, float(raw_t.volume)):.2f} filtvol "
                              f"{max(po, float(out_t.volume)):.2f} voice {max(pc, conf):.2f} "
                              f"speech {ps + (n / sr if state == VADState.SPEAKING else 0):.2f}")
                    continue
                live.update(Text.assemble(
                    (" SPEECH " if talking else " silence ", "reverse green" if talking else "dim"),
                    f"  raw {meter(raw_t.volume)} {float(raw_t.volume):.2f}",
                    f"  → filtered {meter(out_t.volume)} {float(out_t.volume):.2f} (gate {args.min_volume:.2f})",
                    f"  voice {conf:.2f} (gate {args.vad_confidence:.2f})",
                    f"  detected {speaking_secs:4.1f}s",
                ), refresh=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stream.close()
        pa.terminate()
        if chain:
            await chain.stop()
    print(f"peak raw {peak_raw:.2f} · peak filtered {peak_out:.2f} · peak voice {peak_conf:.2f} · "
          f"speech detected {speaking_secs:.1f}s")


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def print_summary(stats: Stats, args: argparse.Namespace) -> None:
    from rich.console import Console

    c = Console()
    c.print(render(stats, args))
    c.print("[dim]Session ended. Nothing was written to disk.[/dim]")


# ---------------------------------------------------------------------------------------------------------
# Pipeline


async def main(args: argparse.Namespace) -> None:
    from loguru import logger

    stats = Stats()
    if not args.no_dashboard:
        logger.remove()
        logger.add(lambda m: stats.warnings.append(m.record["message"][:160]), level="WARNING")

    if args.tts_device == "cuda":
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"  # kokoro-onnx; skips the TensorRT probe
        import onnxruntime

        onnxruntime.preload_dlls()
    else:
        os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"

    import numpy as np
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        InputAudioRawFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        MetricsFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
    )
    from pipecat.metrics.metrics import (
        LLMUsageMetricsData,
        TTFBMetricsData,
        TTSUsageMetricsData,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed
    from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.baseten.llm import BasetenLLMService
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
    from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.workers.runner import WorkerRunner

    stats.state = "starting"
    print(f"Loading models (Whisper on GPU, Kokoro on {args.tts_device}, Silero, smart-turn)…", flush=True)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=args.input_device,
            output_device_index=args.output_device,
            audio_in_filter=make_input_filter(args),
        )
    )
    stt = WhisperSTTService(
        device="cuda",
        compute_type="float16",
        settings=WhisperSTTService.Settings(model=args.stt_model, language=Language.EN),
    )
    # First CUDA inference takes seconds (kernel loading); do it now, not on your first sentence.
    if stt._model is not None:
        segs, _ = stt._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        list(segs)

    extra = {} if args.reasoning == "none" else {"reasoning_effort": args.reasoning}
    llm = BasetenLLMService(
        api_key=load_api_key(),
        settings=BasetenLLMService.Settings(model=args.model, max_completion_tokens=600, extra=extra),
    )
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=args.voice, speed=args.speed), text_filters=[MarkdownTextFilter()]
    )
    cal = "Sure, that sounds reasonable. What language is the uploader written in, and which errors should retry?"
    samples, sr = tts._kokoro.create(cal, voice=args.voice, speed=args.speed, lang="en-us")
    stats.tts_cps = len(cal) / (len(samples) / sr)

    voice_lock = None
    if args.voice_lock:
        from pipecat.frames.frames import TranscriptionFrame, VADUserStoppedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        vp = VoicePrint()
        enrolled = await enroll_voice(args, vp)

        class VoiceLock(FrameProcessor):
            """Drops transcriptions whose audio doesn't match the enrolled voice."""

            def __init__(self):
                super().__init__()
                self._audio: deque = deque()
                self._segments: deque = deque()
                self._seg_start: float | None = None

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                now = time.monotonic()
                if isinstance(frame, InputAudioRawFrame):
                    self._audio.append((now, frame.audio))
                    while self._audio and now - self._audio[0][0] > 30:
                        self._audio.popleft()
                elif isinstance(frame, VADUserStartedSpeakingFrame):
                    self._seg_start = now - 0.5  # VAD confirms speech ~0.2 s after it starts
                elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._seg_start is not None:
                    self._segments.append((self._seg_start, now))
                    self._seg_start = None
                elif isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
                    start, end = self._segments.popleft() if self._segments else (now - 3, now)
                    pcm = b"".join(a for t, a in self._audio if start <= t <= end)
                    secs = len(pcm) / 32000
                    sim = float(enrolled @ vp.embed(pcm)) if secs >= 0.3 else None
                    threshold = args.voice_threshold - (0.1 if secs < 1.0 else 0)  # short clips match worse
                    stats.voice_last_sim = sim
                    if sim is not None and sim < threshold:
                        stats.voice_rejected += 1
                        stats.transcript.append(("other", f"{frame.text}  [not you: {sim:.2f}]"))
                        return
                await self.push_frame(frame, direction)

        voice_lock = VoiceLock()

    tools = setup_tools(llm, stats)
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}], tools=tools)
    turn_strategies = None
    if args.interrupt == "words":
        turn_strategies = UserTurnStrategies(start=[MinWordsUserTurnStartStrategy(min_words=3)])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=args.vad_stop, min_volume=args.min_volume, confidence=args.vad_confidence
                )
            ),
            user_turn_strategies=turn_strategies,
            **(
                {
                    "empty_user_turn": EmptyUserTurnConfig(
                        interrupted_prompt="Someone in the background (not the user) spoke over you. Ignore it "
                        "and briefly continue where you left off."
                    )
                }
                if voice_lock
                else {}
            ),
        ),
    )
    user_agg, assistant_agg = aggregators.user(), aggregators.assistant()

    @user_agg.event_handler("on_user_turn_stopped")
    async def _user_turn(_agg, _strategy, message):
        if message.content:
            stats.turn().user_text = message.content
            sim = f"  [{stats.voice_last_sim:.2f}]" if voice_lock and stats.voice_last_sim is not None else ""
            stats.transcript.append(("you", message.content + sim))

    @user_agg.event_handler("on_user_backchannel")
    async def _backchannel(_agg, _message):
        stats.backchannels += 1

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def _assistant_turn(_agg, message):
        stats.live_bot = ""
        tr = stats.turn()
        tr.bot_text = message.content or ""
        if message.interrupted:
            tr.interrupted = True
            stats.interruptions += 1
            full = (stats.response_full or message.content or "").strip()
            if full and stats.last_barge and stats.response_speech_start:
                # What you'd heard when you started talking: VAD confirms ~0.2 s late, playback reaches you late.
                heard = stats.last_barge - 0.2 - args.output_latency - stats.response_speech_start
                cut = max(0, int(heard * stats.tts_cps))
                if cut < len(full):
                    nxt = full.find(" ", cut)
                    cut = len(full) if nxt == -1 else nxt
                    if full[cut:].strip():
                        stats.transcript.append(("passe", full[:cut].strip(), full[cut:].strip()))
                        stats.last_barge = None
                        return
            if message.content:
                stats.transcript.append(("passe", message.content, "…"))
            return
        if message.content:
            stats.transcript.append(("passe", message.content))

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def _breakdown(_obs, b):
        tr = stats.turn()
        tr.total_s = b.total_secs
        tr.end_of_turn_s = b.user_turn_secs

    from pipecat.audio.volume import AudioVolumeTracker

    mic_tracker = AudioVolumeTracker()

    class Monitor(BaseObserver):
        async def on_push_frame(self, data: FramePushed):
            if not data.first_push:
                return
            f = data.frame
            if isinstance(f, InputAudioRawFrame):
                x = np.frombuffer(f.audio, np.int16).astype(np.float32)
                rms = float(np.sqrt((x**2).mean())) if len(x) else 0.0
                stats.mic_dbfs = 20 * np.log10(max(rms, 1.0) / 32768)
                mic_tracker.update(f.audio, f.sample_rate)
                stats.mic_volume = mic_tracker.volume
                return
            now = time.monotonic()
            if isinstance(f, VADUserStartedSpeakingFrame) and stats._bot_speaking and stats._barge_t is None:
                stats._barge_t = now
                stats.last_barge = now
            elif isinstance(f, UserStartedSpeakingFrame):
                stats.state = "you"
                cur = stats.turns[-1] if stats.turns else None
                if cur is None or cur.llm_s is not None or cur.total_s is not None:
                    stats.new_turn()
            elif isinstance(f, UserStoppedSpeakingFrame):
                stats.state = "thinking"
            elif isinstance(f, LLMFullResponseStartFrame):
                stats.live_bot = ""
                stats.response_full = ""
                stats.response_speech_start = None
            elif isinstance(f, LLMTextFrame):
                stats.live_bot = (stats.live_bot + f.text)[-300:]
                stats.response_full += f.text
            elif isinstance(f, BotStartedSpeakingFrame):
                stats._bot_speaking = True
                if stats.response_speech_start is None:
                    stats.response_speech_start = now
                stats.state = "speaking"
            elif isinstance(f, BotStoppedSpeakingFrame):
                stats._bot_speaking = False
                if stats._barge_t is not None:
                    stats.interrupt_reaction_ms.append((now - stats._barge_t) * 1000)
                    stats._barge_t = None
                if stats.state == "speaking":
                    stats.state = "listening"
            elif isinstance(f, MetricsFrame):
                for d in f.data:
                    self._metric(d)

        def _metric(self, d):
            if not stats.turns:  # startup metrics, before anyone spoke
                return
            tr = stats.turn()
            proc = d.processor.lower()
            if isinstance(d, TTFBMetricsData) and d.value > 0:
                if "stt" in proc or "whisper" in proc:
                    tr.stt_s = d.value
                elif "llm" in proc or "baseten" in proc:
                    tr.llm_s = d.value
                elif "tts" in proc or "kokoro" in proc:
                    tr.tts_s = tr.tts_s or d.value
            elif isinstance(d, LLMUsageMetricsData):
                u = d.value
                stats.llm_calls += 1
                stats.prompt_tokens += u.prompt_tokens
                stats.completion_tokens += u.completion_tokens
                stats.reasoning_tokens += u.reasoning_tokens or 0
                stats.cached_tokens += u.cache_read_input_tokens or 0
                tr.prompt_tokens += u.prompt_tokens
                tr.completion_tokens += u.completion_tokens
                tr.reasoning_tokens += u.reasoning_tokens or 0
            elif isinstance(d, TTSUsageMetricsData):
                stats.tts_chars += d.value

    pipeline = Pipeline(
        [transport.input(), stt, *([voice_lock] if voice_lock else []), user_agg, llm, tts, transport.output(),
         assistant_agg]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency, Monitor()],
        idle_timeout_secs=None,
        enable_rtvi=False,
    )

    hw = asyncio.create_task(sample_hardware(stats))
    stats.state = "listening"
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    try:
        if args.no_dashboard:
            await runner.run()
        else:
            from rich.live import Live

            with Live(render(stats, args), screen=True, auto_refresh=False) as live:

                async def refresh():
                    while True:
                        live.update(render(stats, args), refresh=True)
                        await asyncio.sleep(0.2)

                ui = asyncio.create_task(refresh())
                try:
                    await runner.run()
                finally:
                    ui.cancel()
    finally:
        hw.cancel()
        if stats.recording_since:
            import subprocess

            subprocess.run(["omarchy", "capture", "screenrecording", "--stop-recording"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Stopped the screen recording that was still running (saved to ~/Videos).")
        print_summary(stats, args)


if __name__ == "__main__":
    _ensure_cuda12_libs()
    a = parse_args()
    if a.list_devices:
        list_devices()
        sys.exit(0)
    if a.mic_test:
        mic_test(a)
        sys.exit(0)
    if a.calibrate:
        calibrate(a)
        sys.exit(0)
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
