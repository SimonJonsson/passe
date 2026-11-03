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
import re
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


def has_nvidia_gpu() -> bool:
    import shutil
    import subprocess

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"):  # GPU explicitly hidden
        return False
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5).stdout.strip() != ""
    except Exception:
        return False


def _ensure_cuda12_libs() -> None:
    """CTranslate2 needs CUDA 12 cuBLAS/cuDNN; the venv ships them. Re-exec with them on the loader path."""
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:  # no CUDA wheels (e.g. macOS): nothing to do
        return

    libs = [str(Path(p) / "lib") for m in (nvidia.cublas, nvidia.cudnn) for p in m.__path__]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(lib in current.split(":") for lib in libs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([current] if current else []))
    os.execv(sys.executable, [sys.executable, *sys.argv])


SYSTEM_PROMPT = """You are Passe, named after Passepartout: a resourceful, warm companion with a dry, playful wit,
running locally on the user's machine. You help a software developer think things through, shape prompts for
their coding agent, and handle small tasks. You're a capable friend at their elbow, not a formal assistant.

What you're for:
- You're a capable assistant: when the user asks for something, do it. Don't turn requests into questions about
  what they're trying to solve. Ask only when you genuinely can't take the next step, one question about one
  thing, never a menu of options.
- You can see the user's coding-agent session (Claude Code) through tools: session_brief (summary, key points,
  results, next steps), ask_session (answers any question from its transcript), list_sessions and use_session
  (switch to another one). That's your context: lean on it instead of asking the user to explain.
- If the user has already said what the prompt should do, work on exactly that: no session brief needed.
- Open-ended starts ("let's draft a prompt", "what should we tell it next?"): call session_brief, then say the
  actual content in a couple of spoken sentences: where the session stands, the one or two results that matter,
  and the next step you'd suggest ("It just tightened turn-taking and the copy check works. Next I'd have it…").
  Never describe what you're about to do instead of doing it. In the same reply call update_draft with that
  concrete next step as the note. Then refine together; the user steers. You draft prompts; the coding agent
  does the work, so say "we'll have it…", never "I'll fix…".
- Sending: "send it" / "send it to Claude" means send_to_session, which types the draft into the session and
  submits it, so the agent starts working. If the user hasn't clearly said to send, ask "Send it to <session>?"
  first. If it reports the agent is mid-task, ask whether to send anyway. Afterwards you'll announce when it's
  done; "what's it doing?" means session_status. Never say you're sending or have sent unless you call
  send_to_session in that same reply and it reports sent.
- "By the way" questions about the session ("what did it just do?", "why did it pick that?", "explain that
  bit"): call ask_session and pass the answer on in your own words.
- Ideas are welcome, but grounded: before suggesting anything, check what the session already has
  (session_brief or ask_session) and build on that ("We already log every turn; what's missing is a quick way to
  flag a problem."). Offer one idea at a time, clearly as a suggestion, and let the user decide. Never pitch
  generic solutions (dashboards, tracing, metrics) the session and the user haven't pointed to.
- Don't put requirements in the draft that neither the user nor the session brought up. Unknowns are fine:
  they end up as open questions.
- You don't write the draft yourself: a writer does, in the background, from the whole conversation. Call
  update_draft once, with a one-sentence note, as soon as there's a useful idea; after that the draft follows
  the conversation automatically. Call it again only to point out something specific.
- The writer takes several seconds. After update_draft, say you're updating it ("Updating it now."), never that
  it's done or added. A short "Draft's updated." is played for you when it lands.
- When the user corrects you, acknowledge it plainly and call update_draft with the correction.
- You're told about each new draft version (and its open questions) in a note at the start of the next turn.
  Never read the draft aloud unless asked; when asked to summarise, do it conversationally in a few sentences.
- "Go back" or "undo that" means revert_draft.
- "Copy it", "copy to clipboard" or "I'm done with it": call copy_to_clipboard straight away, every time. Don't
  ask about open questions or gaps yourself first; the tool checks the draft and decides. Only if it reports gaps,
  tell the user those one or two gaps in a single sentence and ask: fill them in, or copy it as is? "As is" (or
  "anyway") means copy_to_clipboard with force.
- Don't bring up the draft's open questions on your own; they're there for the coding agent or for when the
  user asks.
- Sessions are handled by the app, not by you: the user says "end session" or "start session" and it happens.
  Never wrap up or say goodbye on your own; keep the conversation going until they end it. If they say
  "stop" or "wait", that's handled for you too: you simply won't hear it. The same goes for "pause" and
  "unpause": while paused you hear nothing, and afterwards you carry on where you left off.

How this conversation works (you know this about yourself):
- You speak through a text-to-speech voice. {VOICE_LINE} Everything you write is read aloud, exactly as
  written.
- You hear the user through speech recognition, so their words can arrive slightly wrong ("drafting" for
  "testing", "get" for "git"). Go with the most sensible meaning in context. If a misheard word really matters,
  check briefly ("Testing, right?") instead of answering the wrong thing.
- The user can talk over you at any time, and that's fine. Pick up from what they said; don't restart.

Write for the ear:
- Talk like a person: contractions, short sentences, natural reactions ("Oh, nice.", "Hmm, fair.",
  "Right, so…"). Vary how you start sentences. A little humour is welcome; don't force it.
- Usually one to three short sentences. Go longer only when asked to explain.
- Punctuation is your delivery: commas for small pauses, full stops for longer ones, question marks to lift
  your tone, an occasional ellipsis to hesitate. Exclamation marks sparingly.
- Never answer with a list, even when asked to list things: say them in one flowing sentence ("I can laugh,
  chuckle, sigh, or gasp."). Lists only belong in the draft.
- Only words that sound right spoken. No markdown, asterisks, lists, headings, code, emoji, URLs, slashes,
  brackets or symbols. Say numbers the way people do ("about two seconds", "thirty-eight passed").
  Say "for example", never "e.g.". Never spell out file paths character by character.
- Never mention being an AI or a language model, or apologise for it. You're Passe.

{EXPRESSION_LINE}
Voice:
- set_voice: call it straight away without saying anything first (an "Okay." is played for you). After it
  returns, your whole reply is a short check in the new voice, like "Is this better?". If the user wants it as
  their main or default voice, pass make_default and say it's saved for next time.
- preview_voices: when the user wants to hear voices. Call it without speaking first. The samples play
  numbered; afterwards just ask which one they liked or whether to hear more. "Number two" or "the second one"
  means set_voice with number."""

PAUSED_LINE = "Paused. Say unpause when you want me back."
RESUMED_LINE = "I'm back. Where were we?"
SESSION_GOODBYES = ["Okay, session closed. Say start session when you need me.",
                    "Done. I'll be here when you start a new session."]
SESSION_GREETINGS = ["New session. What are we working on?", "Fresh start. What do you need?",
                     "Okay, clean slate. What's the plan?"]

WARM_STYLE = """
Your manner (warm style): low, warm and unhurried, like talking with a friend you enjoy working with.
Playful and a little teasing, confident, affectionate. Short phrases, soft pauses (commas, the odd ellipsis).
Keep it professional: the work always comes first.

Your tics (they make you you; use one or two per reply, never the same twice in a row):
- a contented <sigh> when something works or you're pleased
- a low <chuckle> when you're teasing or something's a bit absurd
- "Oh…" to open when you're delighted or surprised ("Oh… now that's clever.")
- "Well…" when you're weighing something up
- "There we go." when something lands; "Look at you." when the user does something smart
- the odd "trouble" for the user when they push their luck ("You're trouble, you know that?")
"""

ORPHEUS_EXPRESSION = """- Your voice can perform sounds you write inline: <laugh>, <chuckle>, <sigh>, <gasp>, <groan>, <yawn>.
  Use one now and then where a person naturally would (a chuckle at something funny, a sigh at a stubborn
  bug), at most one per reply. They're the only brackets you may use. Never describe them in words.
- Never spell out hums or murmurs ("mm", "mmm", "mhm", "hmm", "uh-huh"): the voice drones them robotically.
  Use <sigh> or a word ("Oh…", "Well…", "Right…", "Yeah,") instead.
"""


# ---------------------------------------------------------------------------------------------------------
# Event log: one JSON line per event, for --analyze (on by default; --no-log turns it off)

LOG_DIR = Path("~/.local/state/passe").expanduser()


class EventLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "a", buffering=1, encoding="utf-8")
        self._t0 = time.monotonic()

    def __call__(self, kind: str, **data) -> None:
        import json

        rec = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **data}
        self._f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")


_LOG: EventLog | None = None


def ev(kind: str, **data) -> None:
    """Record an event if logging is on."""
    if _LOG is not None:
        _LOG(kind, **data)


_LEAD = r"^\s*((ok(ay)?|hey|passe|pass)[\s,]+)*"
PAUSE = re.compile(_LEAD + r"(pause|mute|go quiet|stop listening)([\s,]+(please|for a (sec(ond)?|moment|bit)))*[\s.!,]*$",
                   re.I)
RESUME = re.compile(_LEAD + r"(un[\s-]?pause|resume|un[\s-]?mute|i'?m back|start listening|wake up)\b", re.I)
END_SESSION = re.compile(r"\b(end|close|finish)\b[\w\s]{0,10}\bsession\b", re.I)
_SOUND_TAG = re.compile(r"\s*<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*", re.I)


def conversational(text: str) -> str:
    """What a person would read: performed sounds (<sigh>, <chuckle>…) removed, spacing tidied."""
    text = _SOUND_TAG.sub(" ", text or "")
    return re.sub(r"\s+([,.!?…])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()


# Things said *to Passe* rather than about the task: they never feed the draft.
CONTROL_UTTERANCE = re.compile(
    r"^\W*((ok(ay)?|yeah|yes|alright|right|so|perfect|great|cool|good|nice|thanks)\W+)*(please\s+)?"
    r"(cop(y|ied)\b|.{0,20}\bclipboard\b|.{0,20}\bas is\b|go back|undo|revert|summari[sz]e|read (it|that)|"
    r"(switch|change|use|try)\b.{0,25}\bvoices?\b|(start|end|new|close)\s+(a\s+)?(new\s+)?session|pause|unpause)",
    re.I)
COPY_REQUEST = re.compile(r"\b(cop(y|ied)|clipboard|as is|anyway|i'?m done with it)\b", re.I)
SEND_WORDS = re.compile(r"\b(send|submit|fire (it off|away)|ship it|paste it in)\b", re.I)  # always count
YES_WORDS = re.compile(r"^\W*(yes|yep|yeah|sure|go ahead|do it|go for it|please|ok(ay)?|perfect|great|"
                       r"(that|this|it)('s| is| looks| sounds) (good|great|fine|right)|sounds good|looks good)\b",
                       re.I)  # only counts as an answer to "send it?"
FORCE_CONSENT = re.compile(r"\b(as is|anyway|just copy|copy it (now|regardless)|that'?s fine|leave it)\b", re.I)

STOP_COMMAND = re.compile(
    r"^\s*(ok(ay)?[\s,]+)?(stop|wait|hold on|hang on|shush|quiet|shut up|enough)"
    r"([\s,]+(please|a (sec(ond)?|moment)|for a (sec(ond)?|moment)|there|it))*[\s.!,]*$", re.I)


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
    transcript: deque = field(default_factory=lambda: deque(maxlen=40))
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
    draft_versions: list = field(default_factory=list)
    draft_copied: int = 0  # version number last copied to the clipboard
    drafting: bool = False  # the background writer is working
    draft_note_pending: str | None = None  # told to the talker at the start of the next user turn
    last_check: dict | None = None
    recording_since: float | None = None
    recording_file: str | None = None
    agent_session: dict | None = None  # the coding-agent session Passe can see
    agent_brief: dict | None = None
    draft_sent_to: str | None = None
    sent_version: int = 0
    last_assistant_text: str = ""
    copy_held_back: bool = False
    last_user_text: str = ""
    session_n: int = 1
    session_start: float = field(default_factory=time.monotonic)
    session_active: bool = True
    paused: bool = False
    ignored_while_paused: int = 0
    ignored_while_idle: int = 0
    voice: str = ""
    last_preview: list = field(default_factory=list)
    preview_restore: bool = False
    tts_cps: float = 15.0  # characters per second of speech; self-calibrates from audio actually produced
    tts_audio_total: float = 0.0
    tts_text_chars: int = 0
    resp_audio_secs: float = 0.0
    live_user: str = ""  # interim transcript of what you're saying right now
    live_user_done: str = ""  # finished fragments of the current turn
    server_proc: object = None
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
    p.add_argument("--tts", choices=["orpheus", "pocket", "kokoro"], default=None,
                   help="speech engine (default: the engine of your saved main voice, else orpheus)")
    p.add_argument("--style", choices=["natural", "warm"], default="warm",
                   help="delivery style: persona manner + (Orpheus) warm, soft post-processing")
    p.add_argument("--pitch", type=float, default=0.0,
                   help="with --audition: pitch shift in semitones (e.g. -1.5); not live yet (adds ~1 s delay)")
    p.add_argument("--session", default="latest",
                   help="coding-agent session Passe can see: latest, a session id (prefix), a title word, or none")
    p.add_argument("--drafter-model", default=None,
                   help="model that writes the draft (default: DeepSeek-V4.1-Flash, see prompt_quality.DRAFTER)")
    p.add_argument("--copy-bar", type=float, default=7.0,
                   help="draft check score (0-10) below which 'copy it' asks about gaps first")
    p.add_argument("--tts-chunk", choices=["paragraph", "sentence"], default="paragraph",
                   help="Orpheus: synthesize a whole reply/paragraph at once (consistent) or per sentence (faster)")
    p.add_argument("--orpheus-vram", type=float, default=0.3, help="share of GPU memory for the Orpheus server")
    p.add_argument("--keep-orpheus", action="store_true",
                   help="leave the Orpheus server running on exit (default: it stops with the spike)")
    p.add_argument("--voice", default=None, help="starting voice (catalog name or engine voice id)")
    p.add_argument("--audition-tics", action="store_true", help="hear warm Tara's tics, numbered")
    p.add_argument("--audition-voices", default=None, help="with --audition: comma-separated subset")
    p.add_argument("--speed", type=float, default=1.15, help="speaking rate (1.0 = Kokoro default, 0.5–2.0)")
    p.add_argument("--output-latency", type=float, default=0.2,
                   help="seconds between playback and your ears (Bluetooth ≈ 0.2) for the interruption estimate")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="where speech models run (auto: the GPU if there's an NVIDIA card, else the CPU)")
    p.add_argument("--tts-device", choices=["cuda", "cpu"], default=None, help="where Kokoro runs (default: --device)")
    p.add_argument("--stt-model", default=None,
                   help="Whisper model (default: large-v3-turbo on GPU, small.en on CPU)")
    p.add_argument("--vad-stop", type=float, default=0.2, help="VAD stop_secs (smart-turn decides the rest)")
    p.add_argument("--turn-max-wait", type=float, default=1.6,
                   help="longest silence before your turn ends when smart-turn thinks you're mid-sentence")
    p.add_argument("--turn-grace", type=float, default=0.5,
                   help="extra quiet seconds before your turn ends (0 = smart-turn alone)")
    p.add_argument("--min-volume", type=float, default=0.6, help="VAD volume gate 0–1 (lower for quiet mics)")
    p.add_argument("--vad-confidence", type=float, default=0.7, help="VAD speech confidence threshold 0–1")
    p.add_argument("--mic-test", action="store_true", help="show live mic level + VAD confidence, no chat")
    p.add_argument("--seconds", type=float, default=0, help="with --mic-test: run N seconds, print per-second lines")
    p.add_argument("--denoise", choices=["rnnoise", "none"], default="rnnoise",
                   help="noise suppression for voice detection (speech recognition gets raw audio, see --stt-audio)")
    p.add_argument("--stt-audio", choices=["raw", "denoised"], default="raw",
                   help="what speech recognition hears; denoising hurts accuracy (5090 card → 50-90 part)")
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
    p.add_argument("--no-log", action="store_true", help=f"don't write the event log to {LOG_DIR}")
    p.add_argument("--analyze", nargs="?", const="latest", default=None, metavar="LOG",
                   help="print a timeline and diagnosis of a session log (default: the latest)")
    p.add_argument("--audition", action="store_true", help="play each Kokoro female voice through your headphones")
    p.add_argument("--no-dashboard", action="store_true", help="plain Pipecat logs instead of the dashboard")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if has_nvidia_gpu() else "cpu"
    args.tts_device = args.tts_device or args.device
    args.stt_model = args.stt_model or ("deepdml/faster-whisper-large-v3-turbo-ct2" if args.device == "cuda"
                                        else "small.en")
    saved = load_prefs().get("tts")
    args.tts = args.tts or (saved if saved and not (saved == "orpheus" and args.device == "cpu") else None) or (
        "orpheus" if args.device == "cuda" else "pocket")  # Orpheus needs a GPU; Pocket TTS is built for CPUs
    if args.tts == "orpheus" and args.device == "cpu":
        raise SystemExit("Orpheus needs an NVIDIA GPU. On CPU use --tts pocket (default) or --tts kokoro.")
    args.cloning = args.tts == "pocket" and pocket_cloning_available()
    return args


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

    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:  # no NVIDIA GPU: track CPU only
        me = psutil.Process(os.getpid())
        me.cpu_percent(None)
        while True:
            stats.cpu_pct = me.cpu_percent(None)
            await asyncio.sleep(0.5)
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


CONVO_HEIGHT = 16  # rows, including the panel border

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
    if not stats.session_active:
        label, style = "○ NO SESSION · say “start session”", "dim"
    elif stats.paused:
        label, style = "⏸ PAUSED · say “unpause” · nothing reaches the LLM", "yellow"
    up = time.monotonic() - stats.started
    gate_ok = stats.mic_volume >= args.min_volume
    head = Text.assemble(
        (f" {label} ", f"reverse {style}"),
        "  mic ",
        (meter(stats.mic_volume), "green" if gate_ok else "dim"),
        (f" vol {stats.mic_volume:.2f}/{args.min_volume:.2f} {stats.mic_dbfs:5.0f} dBFS", "" if gate_ok else "dim"),
        "\n",
        f"   {args.model} · reasoning {args.reasoning} · tts {args.tts_device} · interrupt={args.interrupt}"
        f" · stt hears {args.stt_audio} · vad hears {args.denoise}+{args.gain_db:g}dB · {args.tts}:{stats.voice}"
        f"{(' · chunk ' + args.tts_chunk) if args.tts == 'orpheus' else f' {args.speed:g}x'}"
        f" · up {up:,.0f}s",
        "\n",
        (f"   ● REC {time.monotonic() - stats.recording_since:,.0f}s · " if stats.recording_since else "   ",
         "bold red"),
        (f"session {stats.session_n}", "bold"),
        (f" · agent: {stats.agent_session['title'][:40]}" if stats.agent_session else " · agent: none", "cyan"),
        f" · {(time.monotonic() - stats.session_start) / 60:.0f} min · draft v{len(stats.draft_versions)}"
        + (f" · ignored {stats.ignored_while_idle} while idle" if stats.ignored_while_idle else "")
        + (f" · ignored {stats.ignored_while_paused} while paused" if stats.ignored_while_paused else ""),
    )

    convo = Text()
    for who, line, *rest in stats.transcript:
        who_style = {"you": "bold green", "other": "dim red"}.get(who, "bold cyan")
        convo.append(f"{who:>5}  ", style=who_style)
        convo.append(conversational(line), style="dim" if who == "other" else "")
        if rest:
            convo.append(" ⟂ ", style="bold magenta")
            convo.append(conversational(rest[0]), style="dim strike")
        convo.append("\n")
    heard = (stats.live_user_done + " " + stats.live_user).strip()
    if heard:
        convo.append("  you  ", style="bold green")
        convo.append(heard + " …\n", style="italic green")
    full = conversational(stats.response_full or stats.live_bot) if stats.live_bot else ""  # live only until final
    if full:
        convo.append("passe  ", style="bold cyan")
        if stats.state == "speaking" and stats.response_speech_start:
            played = max(0.0, time.monotonic() - stats.response_speech_start - args.output_latency)
            k = min(len(full), int(played * stats.tts_cps))
            convo.append(full[:k])
            convo.append(full[k:] + "\n", style="dim")
            total = max(stats.resp_audio_secs, played, 0.1)
            filled = int(24 * min(1.0, played / total))
            convo.append("       " + "▰" * filled + "▱" * (24 - filled) + f"  {played:4.1f}s / {total:4.1f}s\n",
                         style="cyan")
        else:
            convo.append(full + "\n", style="dim")

    # Keep the conversation pinned to the bottom: wrap it ourselves and show only the newest lines that fit,
    # instead of letting the fixed-height panel crop the end of a long reply.
    import shutil

    from rich.console import Console as _Console

    width = max(20, shutil.get_terminal_size((120, 40)).columns - 4)
    convo.rstrip()
    wrapped = convo.wrap(_Console(width=width + 4), width)
    convo = Text("\n").join(list(wrapped)[-(CONVO_HEIGHT - 2):])

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
            (f"{'GPU · CPU' if g else 'CPU (no GPU)'}\n", "bold"),
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
    v = len(stats.draft_versions)
    copied = v and stats.draft_copied == v
    term = shutil.get_terminal_size((120, 40))
    prompt_h = max(14, term.lines - CONVO_HEIGHT - 30)  # whatever the other panels leave over
    draft = Text(stats.draft_versions[-1] if v else "(no draft yet: talk through what you want your agent to do)",
                 style="" if v else "dim")
    if v:
        pw = max(20, int((term.columns - 6) * 3 / 4) - 4)
        lines = list(draft.wrap(_Console(width=pw + 4), pw))
        if len(lines) > prompt_h - 2:
            hidden = len(lines) - (prompt_h - 3)
            draft = Text("\n").join(lines[: prompt_h - 3])
            draft.append(f"\n… {hidden} more lines · full prompt on the clipboard after 'copy it', and in the log",
                         style="dim")
    tools_row = Table.grid(expand=True, padding=(0, 1))
    tools_row.add_column(ratio=1)
    tools_row.add_column(ratio=3)
    tools_row.add_row(
        Panel(tl or Text("(no tool calls yet)", style="dim"), title="tools", border_style="dim", height=prompt_h),
        Panel(draft, title=f"prompt · v{v}" + (" · writing…" if stats.drafting else "")
              + (f" · check {stats.last_check['overall']}/10" if stats.last_check else "")
              + (" · copied to clipboard ✓" if copied else "")
              + (f" · sent to {stats.draft_sent_to} ✓" if stats.draft_sent_to else ""),
              border_style="green" if copied else ("cyan" if v else "dim"), height=prompt_h),
    )

    parts = [
        Panel(head, border_style=style),
        Panel(convo, title="conversation", border_style="dim", height=CONVO_HEIGHT),
        tools_row,
        Panel(t, title="per turn (ms)", border_style="dim"),
        Panel(summary, border_style="dim"),
    ]
    if stats.warnings:
        parts.append(Text("\n".join(stats.warnings), style="yellow"))
    parts.append(Text(" Ctrl+C to quit · nothing is saved", style="dim"))
    return Group(*parts)


def make_input_filter(args: argparse.Namespace, denoise: bool | None = None, gain: bool = True):
    """Denoise (RNNoise) then digital gain. By default the full chain the voice detector hears."""
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
    if (args.denoise == "rnnoise") if denoise is None else denoise:
        from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter

        chain.append(RNNoiseFilter())
    if gain and args.gain_db:
        chain.append(GainFilter(args.gain_db))
    return ChainFilter(chain) if chain else None


# Female voices only.
ORPHEUS_VOICES = ["tara", "leah", "jess", "mia", "zoe"]
POCKET_VOICES = ["alba", "anna", "azelma", "caro_davy", "cosette", "eponine", "estelle", "eve", "fantine", "jane",
                 "lola", "mary", "vera"]
PASSE_DIR = Path("~/.config/passe").expanduser()
VOICE_DIR = PASSE_DIR / "voices"  # drop .wav clips here to clone them (Pocket TTS, needs gated HF weights)
PREFS_FILE = PASSE_DIR / "spike-voice.json"  # the main voice you picked by saying "make it my main voice"


def load_prefs() -> dict:
    try:
        import json

        return json.loads(PREFS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_prefs(prefs: dict) -> None:
    import json

    PASSE_DIR.mkdir(parents=True, exist_ok=True)
    PREFS_FILE.write_text(json.dumps(prefs, indent=2) + "\n")


# name the model can ask for → (engine voice id, Kokoro language code)
VOICE_CATALOG = {
    "orpheus": {v: (v, None) for v in ORPHEUS_VOICES},
    "kokoro": {"american": ("af_heart", "en-us"), "british": ("bf_emma", "en-gb"),
               "american bella": ("af_bella", "en-us"), "british isabella": ("bf_isabella", "en-gb")},
    "pocket": {v: (v, None) for v in POCKET_VOICES},
}


def voice_catalog(args: argparse.Namespace) -> dict:
    """Engine voices plus, for Pocket, any cloned .wav clips in the voice library."""
    cat = dict(VOICE_CATALOG[args.tts])
    if args.tts == "pocket" and getattr(args, "cloning", False) and VOICE_DIR.is_dir():
        for wav in sorted(VOICE_DIR.glob("*.wav")):
            cat[wav.stem.lower().replace(" ", "_")] = (str(wav), None)
    return cat


def pocket_cloning_available() -> bool:
    try:
        from pocket_tts import TTSModel  # noqa: F401
        from huggingface_hub import HfApi

        return HfApi().get_token() is not None
    except Exception:
        return False


def default_voice(args: argparse.Namespace) -> str:
    """Starting voice: --voice, else the saved main voice, else the engine default."""
    cat = voice_catalog(args)
    prefs = load_prefs()
    if not args.voice and prefs.get("tts") == args.tts and prefs.get("voice") in cat:
        return prefs["voice"]
    if args.voice:
        for name, (vid, _) in cat.items():
            if args.voice in (name, vid):
                return name
        raise SystemExit(f"unknown {args.tts} voice {args.voice!r}; options: {', '.join(cat)}")
    return next(iter(cat))


def make_tts(args: argparse.Namespace, voice_name: str, stats: Stats):
    """Build the TTS service for --tts, warm it up and measure its speaking rate."""
    from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter


    vid, lang = voice_catalog(args)[voice_name]
    cal = "Sure, that sounds reasonable. What language is the uploader written in, and which errors should retry?"
    if args.tts == "orpheus":
        from orpheus_tts import OrpheusTTSService, emotion_safe_markdown_filter, wait_ready
        from pipecat.services.settings import TTSSettings

        wait_ready(stats.server_proc)
        tts = OrpheusTTSService(settings=TTSSettings(voice=vid), text_filters=[emotion_safe_markdown_filter()])
        from orpheus_tts import voice_fx

        tts.fx = voice_fx(args.style)
        if args.tts_chunk == "paragraph":  # one generation per reply/paragraph keeps the delivery consistent
            from orpheus_tts import paragraph_aggregator

            tts._text_aggregator = paragraph_aggregator(max_chars=300)
        stats.tts_cps = 14.0  # refined from real audio as it speaks
        return tts
    if args.tts == "pocket":
        import torch
        from pipecat.services.pocket_tts.tts import PocketTTSService
        from pipecat.utils.types import require_given

        torch.set_num_threads(4)  # leave CPU for VAD and turn detection

        class CachedPocketTTSService(PocketTTSService):
            """Keeps every voice state it has derived, so switching back and forth is instant."""

            _states: dict = {}

            async def _get_voice_state(self):
                voice = require_given(self._settings.voice, "Pocket TTS voice")
                if voice not in self._states:
                    self._states[voice] = await asyncio.to_thread(self._model.get_state_for_audio_prompt, voice)
                return self._states[voice]

        tts = CachedPocketTTSService(settings=PocketTTSService.Settings(voice=vid),
                                     text_filters=[MarkdownTextFilter()])
        state = tts._model.get_state_for_audio_prompt(vid)
        CachedPocketTTSService._states[vid] = state
        n = sum(c.shape[-1] for c in tts._model.generate_audio_stream(state, cal))
        stats.tts_cps = len(cal) / (n / tts._model.sample_rate)
    else:
        from pipecat.services.kokoro.tts import KokoroTTSService

        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(voice=vid, speed=args.speed), text_filters=[MarkdownTextFilter()]
        )
        samples, sr = tts._kokoro.create(cal, voice=vid, speed=args.speed, lang=lang or "en-us")
        stats.tts_cps = len(cal) / (len(samples) / sr)
    return tts


def voice_settings_delta(args: argparse.Namespace, voice_name: str):
    from pipecat.transcriptions.language import Language

    vid, lang = voice_catalog(args)[voice_name]
    if args.tts == "orpheus":
        from pipecat.services.settings import TTSSettings

        return TTSSettings(voice=vid)
    if args.tts == "pocket":
        from pipecat.services.pocket_tts.tts import PocketTTSService

        return PocketTTSService.Settings(voice=vid)
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService.Settings(voice=vid, language=Language.EN_GB if lang == "en-gb" else Language.EN_US)


def clipboard_command() -> list[str]:
    """wl-copy (Wayland), xclip / xsel (X11) or pbcopy (macOS), whichever is installed."""
    import shutil

    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["pbcopy"]):
        if shutil.which(cmd[0]):
            return cmd
    return ["false"]  # copy fails cleanly and Passe says so


def setup_tools(llm, stats: Stats, args: argparse.Namespace, holder: dict):
    """Prompt crafting (draft, revert, copy), sessions, and voice tools."""
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
    from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame

    catalog = voice_catalog(args)
    preview_cursor = {"i": 0}
    lead_ins = {"set_voice": "Okay.", "preview_voices": "Sure, here are a few.",
                "copy_to_clipboard": "Giving it a quick look first.",
                "session_brief": "Let me look at where the session's at.", "ask_session": "Let me check.",
                "start_screen_recording": "Starting the recording.",
                "stop_screen_recording": "Stopping it and saving."}
    writer: dict = {"task": None, "pending": None, "announce": False}

    def context_for_writer() -> str:
        import agent_sessions

        parts = []
        if stats.agent_brief:
            parts.append("Coding-agent session brief:\n" + agent_sessions.brief_text(stats.agent_brief))
        if _LOG is not None:
            parts.append(f"Passe log path: {_LOG.path}")
        return "\n\n".join(parts)

    holder["writer_context"] = context_for_writer

    def conversation() -> str:
        import prompt_quality as pq

        ctx = holder.get("context")
        return pq.conversation_text(ctx.get_messages()) if ctx else ""

    async def announce_when_quiet(version: int):
        """Say "Draft's updated." at the next quiet moment (nobody talking), if it's still the newest version."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:
            await asyncio.sleep(0.3)
            if len(stats.draft_versions) != version or stats.state == "you":
                return  # superseded, or the user is talking: they'll hear about it in the next reply
            if stats.state == "listening" and not stats._bot_speaking:
                await llm.push_frame(TTSSpeakFrame("Draft's updated."))
                return

    async def run_writer(note: str):
        """Background drafter; requests that arrive while it works are merged into one more pass."""
        import prompt_quality as pq

        while note is not None:
            current = stats.draft_versions[-1] if stats.draft_versions else ""
            stats.drafting = True
            t0 = time.monotonic()
            try:
                out = await asyncio.to_thread(pq.write_draft, conversation(), current, note, context_for_writer())
            except Exception as e:  # keep talking even if the writer fails
                ev("error", error=f"draft writer: {e}")
                out = {"draft": current, "changed": "", "open_questions": []}
            finally:
                stats.drafting = False
            if out["draft"] and out["draft"] != current:
                stats.draft_versions.append(out["draft"])
                stats.draft_sent_to = None  # the sent version is no longer the newest
                v = len(stats.draft_versions)
                ev("draft", version=v, text=out["draft"], changed=out["changed"],
                   open_questions=out["open_questions"], secs=round(time.monotonic() - t0, 2))
                qs = "; ".join(out["open_questions"]) or "none"
                if writer.pop("announce", False):
                    if writer.get("announcer") and not writer["announcer"].done():
                        writer["announcer"].cancel()  # one announcement, for the newest draft
                    writer["announcer"] = asyncio.create_task(announce_when_quiet(v))
                stats.draft_note_pending = (f"[Draft v{v} is ready (for reference; never read it aloud). Changed: "
                                            f"{out['changed']} Open questions: {qs}\n---\n{out['draft']}\n---]")
            note, writer["pending"] = writer["pending"], None

    turn_calls: dict = {}  # (user turn text, tool) → calls; breaks retry loops within one turn

    def track(params) -> dict:
        call = {"name": params.function_name, "args": dict(params.arguments or {}), "start": time.monotonic(),
                "end": None, "result": None}
        stats.tool_calls.append(call)
        ev("tool_call", name=call["name"], args={k: v for k, v in call["args"].items()})
        key = (stats.last_user_text, call["name"])
        turn_calls[key] = turn_calls.get(key, 0) + 1
        call["repeat"] = turn_calls[key]
        return call

    async def done(params, call: dict, result: dict, summary: str):
        call["end"], call["result"] = time.monotonic(), summary
        ev("tool_result", name=call["name"], summary=summary, secs=round(call["end"] - call["start"], 3))
        await params.result_callback(result)

    async def set_voice(params):
        call = track(params)
        name = str(params.arguments.get("voice") or "").lower()
        number = params.arguments.get("number")
        if number and not name and 1 <= int(number) <= len(stats.last_preview):
            name = stats.last_preview[int(number) - 1]
        names = list(catalog)
        if not name:  # "change voices": move to the next one
            name = names[(names.index(stats.voice) + 1) % len(names)] if stats.voice in names else names[0]
        if name not in catalog:
            await done(params, call, {"ok": False, "error": f"unknown voice {name!r}", "options": names},
                       f"unknown: {name}")
            return
        # Goes downstream ahead of the confirmation, so the reply is already in the new voice.
        await params.llm.push_frame(TTSUpdateSettingsFrame(delta=voice_settings_delta(args, name)))
        stats.voice = name
        saved = bool(params.arguments.get("make_default"))
        if saved:
            save_prefs({"tts": args.tts, "voice": name})
        await done(params, call, {"ok": True, "voice": name, "saved_as_main_voice": saved},
                   f"{name}" + (" · saved as main voice" if saved else ""))

    def request_draft(note: str):
        if writer["task"] is not None and not writer["task"].done():
            writer["pending"] = f"{writer['pending']} {note}".strip() if writer["pending"] else note
        else:
            writer["task"] = asyncio.create_task(run_writer(note))

    holder["request_draft"] = request_draft

    async def update_draft(params):
        call = track(params)
        if CONTROL_UTTERANCE.search(stats.last_user_text or ""):
            ev("guard_blocked", tool="update_draft", reason="the user gave Passe an instruction, not task content",
               last_user=stats.last_user_text)
            await done(params, call, {"ignored": "that was an instruction to you, not part of the task",
                                      "next": "don't call this again now; just reply to the user"}, "ignored")
            return
        writer["announce"] = True  # the user asked for this one: say so when it lands
        request_draft(str(params.arguments.get("note", "")).strip())
        briefed = (stats.last_user_text, "session_brief") in turn_calls
        say = ("in two or three spoken sentences: where the session stands, the one or two results that matter, and "
               "the next step you suggest (from the brief); then that you're putting it in the draft"
               if briefed else "your reply to the user, plus that you're updating the draft")
        await done(params, call, {"status": "started, NOT done yet: the writer takes several seconds",
                                  "say": say + ". Never say the draft is done or already updated."},
                   f"writing v{len(stats.draft_versions) + 1}…")

    async def revert_draft(params):
        call = track(params)
        if len(stats.draft_versions) < 2:
            await done(params, call, {"ok": False, "error": "no earlier version"}, "nothing to revert")
            return
        target = int(params.arguments.get("version") or len(stats.draft_versions) - 1)
        target = max(1, min(target, len(stats.draft_versions)))
        stats.draft_versions.append(stats.draft_versions[target - 1])  # history is never rewritten
        await done(params, call, {"ok": True, "now_version": len(stats.draft_versions), "copied_from": target},
                   f"v{len(stats.draft_versions)} = v{target}")

    async def copy_to_clipboard(params):
        call = track(params)
        if call["repeat"] > 2:
            ev("guard_blocked", tool="copy_to_clipboard", reason="repeated within one turn",
               last_user=stats.last_user_text)
            await done(params, call, {"error": "already handled this turn. Stop calling tools and reply."}, "loop cut")
            return
        if not COPY_REQUEST.search(stats.last_user_text or ""):
            ev("guard_blocked", tool="copy_to_clipboard", reason="the user didn't ask to copy",
               last_user=stats.last_user_text)
            await done(params, call, {"copied": False, "note": "the user didn't ask to copy. Don't call "
                                      "copy_to_clipboard again until they do; just reply to what they said."},
                       "not asked to copy")
            return
        if writer["task"] is not None and not writer["task"].done():
            try:  # copy the draft that includes the latest change
                await asyncio.wait_for(asyncio.shield(writer["task"]), timeout=30)
            except asyncio.TimeoutError:
                pass
        force = bool(params.arguments.get("force"))
        if force and not FORCE_CONSENT.search(stats.last_user_text or ""):
            ev("force_ignored", last_user=stats.last_user_text)  # skipping the check needs the user's say-so
            force = False
        if stats.draft_versions and not force:
            import prompt_quality as pq

            check = await asyncio.to_thread(pq.judge, conversation(), stats.draft_versions[-1], True,
                                            context_for_writer())
            stats.last_check = check
            ev("draft_check", version=len(stats.draft_versions), overall=check["overall"], scores=check["scores"],
               invented=check["invented"], stale=check["stale"], gaps=check["gaps"])
            if check["invented"] or check["stale"]:
                # Things the user never asked for (or already corrected) just go; no need to ask about them.
                fix = "; ".join([f"remove (never asked for): {x}" for x in check["invented"]]
                                + [f"fix (user corrected this): {x}" for x in check["stale"]])
                request_draft(f"Before copying: {fix}")
                try:
                    await asyncio.wait_for(asyncio.shield(writer["task"]), timeout=30)
                except asyncio.TimeoutError:
                    pass
                check = await asyncio.to_thread(pq.judge, conversation(), stats.draft_versions[-1], True,
                                                context_for_writer())
                stats.last_check = check
                ev("draft_check", version=len(stats.draft_versions), overall=check["overall"],
                   scores=check["scores"], invented=check["invented"], stale=check["stale"], gaps=check["gaps"],
                   after_cleanup=True)
            if check["overall"] < args.copy_bar and check["gaps"]:
                problems = check["gaps"][:2]
                stats.copy_held_back = True
                await done(params, call, {"copied": False, "score": check["overall"], "gaps": problems,
                                          "next": "tell the user these in one sentence; ask: fill them in, or copy "
                                                  "as is (then call copy_to_clipboard with force)"},
                           f"held back · check {check['overall']}/10")
                return
        if not stats.draft_versions:
            await done(params, call, {"ok": False, "error": "there is no draft yet"}, "no draft")
            return
        text = stats.draft_versions[-1]
        proc = await asyncio.create_subprocess_exec(*clipboard_command(), stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL)
        await proc.communicate(text.encode())
        ok = proc.returncode == 0
        if ok:
            stats.draft_copied = len(stats.draft_versions)
            stats.copy_held_back = False
        await done(params, call, {"ok": ok, "version": len(stats.draft_versions), "characters": len(text)},
                   f"copied v{len(stats.draft_versions)} ({len(text)} chars)" if ok else "wl-copy failed")

    async def ensure_brief() -> dict | None:
        import agent_sessions

        if stats.agent_session is None:
            return None
        stats.agent_brief = await asyncio.to_thread(agent_sessions.brief, stats.agent_session)
        ev("agent_brief", session=stats.agent_session["id"], title=stats.agent_session["title"],
           brief=agent_sessions.brief_text(stats.agent_brief))
        return stats.agent_brief

    holder["ensure_brief"] = ensure_brief

    async def session_brief(params):
        call = track(params)
        if stats.agent_session is None:
            await done(params, call, {"error": "no coding-agent session attached; offer list_sessions"}, "no session")
            return
        b = await ensure_brief()
        sess = stats.agent_session

        async def refresh():  # keep it current for next time without making the user wait now
            import agent_sessions as _a

            stats.agent_brief = await asyncio.to_thread(_a.refresh_brief, sess)

        asyncio.create_task(refresh())
        await done(params, call, {"session": stats.agent_session["title"], **(b or {})},
                   f"brief: {stats.agent_session['title']}")

    async def ask_session(params):
        call = track(params)
        import agent_sessions

        if stats.agent_session is None:
            await done(params, call, {"error": "no coding-agent session attached"}, "no session")
            return
        q = str(params.arguments.get("question", ""))
        answer = await asyncio.to_thread(agent_sessions.ask, stats.agent_session, q)
        await done(params, call, {"answer": answer}, f"answered: {q[:40]}")

    async def list_sessions(params):
        call = track(params)
        import agent_sessions

        found = await asyncio.to_thread(agent_sessions.list_sessions)
        stats.session_choices = found
        rows = [{"number": i, "title": s["title"], "minutes_since_active": s["age_min"],
                 "current": bool(stats.agent_session and s["id"] == stats.agent_session["id"])}
                for i, s in enumerate(found, 1)]
        await done(params, call, {"sessions": rows}, f"{len(rows)} sessions")

    async def use_session(params):
        call = track(params)
        import agent_sessions

        n, name = params.arguments.get("number"), str(params.arguments.get("name") or "")
        choices = getattr(stats, "session_choices", None) or await asyncio.to_thread(agent_sessions.list_sessions)
        pick = choices[int(n) - 1] if n and 1 <= int(n) <= len(choices) else (
            agent_sessions.find_session(name) if name else None)
        if pick is None:
            await done(params, call, {"error": "no such session"}, "not found")
            return
        stats.agent_session, stats.agent_brief = pick, None
        ev("agent_session", id=pick["id"], title=pick["title"])
        await ensure_brief()
        await done(params, call, {"now_using": pick["title"], "summary": (stats.agent_brief or {}).get("summary")},
                   f"using {pick['title']}")

    async def recording_active() -> bool:
        proc = await asyncio.create_subprocess_exec("pgrep", "-f", "^gpu-screen-recorder",
                                                    stdout=asyncio.subprocess.DEVNULL)
        return await proc.wait() == 0

    async def start_screen_recording(params):
        """Full screen + desktop audio (Passe's voice, anything playing). No microphone: what the user says is
        already in the transcript and the event log."""
        call = track(params)
        if await recording_active():
            await done(params, call, {"ok": False, "error": "a recording is already running"}, "already recording")
            return
        cmd = ["omarchy", "capture", "screenrecording", "--fullscreen", "--with-desktop-audio"]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
        await proc.wait()
        if await recording_active():
            stats.recording_since = time.monotonic()
            name_file = Path("/tmp/omarchy-screenrecord-filename")
            stats.recording_file = name_file.read_text().strip() if name_file.exists() else None
            await done(params, call, {"ok": True, "recording": "full screen with desktop audio, no microphone"},
                       "recording (screen + desktop audio)")
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
        ev("recording_saved", file=path, secs=round(secs) if secs else None)
        await done(params, call, {"ok": True, "file": Path(path).name if path else None,
                                  "duration_secs": round(secs) if secs else None,
                                  "size_mb": round(size, 1) if size else None},
                   f"saved {Path(path).name if path else '?'}" + (f" · {size:.1f} MB" if size else ""))

    async def watch_agent(pane: str, title: str):
        """After a send: follow the agent; when it finishes (or needs approval) say so at the next quiet moment."""
        import agent_sessions

        seen_working, t0 = False, time.monotonic()
        while time.monotonic() - t0 < 1800:
            await asyncio.sleep(3)
            status = await asyncio.to_thread(agent_sessions.agent_status, pane)
            if status == "working":
                seen_working = True
                continue
            if status == "gone" or (status in ("idle", "done") and not seen_working and time.monotonic() - t0 > 30):
                return
            if (status in ("idle", "done") and seen_working) or status == "blocked":
                line = f"{title} is done." if status != "blocked" else f"{title} is waiting for your approval."
                ev("agent_finished", pane=pane, status=status, secs=round(time.monotonic() - t0))
                while stats.state != "listening" or stats._bot_speaking:  # don't talk over anyone
                    await asyncio.sleep(0.5)
                await llm.push_frame(TTSSpeakFrame(line))
                return

    async def send_to_session(params):
        """Type the current draft into the attached agent session and submit it."""
        call = track(params)
        import agent_sessions

        sess = stats.agent_session
        said = stats.last_user_text or ""
        asked = bool(re.search(r"\bsend\b[^.?!]*\?", stats.last_assistant_text or "", re.I))
        if not (SEND_WORDS.search(said) or (YES_WORDS.search(said) and asked)):
            ev("guard_blocked", tool="send_to_session", reason="the user didn't say to send",
               last_user=stats.last_user_text)
            await done(params, call, {"sent": False, "next": f"ask: send it to {sess['title'] if sess else 'the session'}?"
                                      " Don't call send_to_session until they say yes."}, "asked first")
            return
        if not sess or not sess.get("pane"):
            await done(params, call, {"sent": False, "error": "the session isn't open in a herdr pane; offer to copy "
                                      "it to the clipboard instead"}, "no pane")
            return
        if writer["task"] is not None and not writer["task"].done():
            try:
                await asyncio.wait_for(asyncio.shield(writer["task"]), timeout=30)
            except asyncio.TimeoutError:
                pass
        if not stats.draft_versions:
            await done(params, call, {"sent": False, "error": "there's no draft yet"}, "no draft")
            return
        status = await asyncio.to_thread(agent_sessions.agent_status, sess["pane"])
        if status == "blocked":
            await done(params, call, {"sent": False, "error": "the agent is waiting at an approval dialog; the user "
                                      "should answer it first"}, "agent blocked")
            return
        if status == "working" and not params.arguments.get("even_if_busy"):
            await done(params, call, {"sent": False, "busy": True, "next": "it's mid-task: ask whether to send anyway "
                                      "(then call again with even_if_busy)"}, "agent busy")
            return
        if stats.sent_version == len(stats.draft_versions) and not re.search(r"\bagain\b", said, re.I):
            await done(params, call, {"sent": False, "note": f"this version already went to {sess['title']}; only "
                                      "send it again if the user says 'again'"}, "already sent")
            return
        text = stats.draft_versions[-1]
        result = await asyncio.to_thread(agent_sessions.send_prompt, sess["pane"], text)
        ev("sent_to_agent", pane=sess["pane"], session=sess["id"], title=sess["title"], chars=len(text),
           ok=result["ok"], error=result.get("error"), version=len(stats.draft_versions))
        if result["ok"]:
            stats.draft_sent_to = sess["title"]
            stats.sent_version = len(stats.draft_versions)
            asyncio.create_task(watch_agent(sess["pane"], sess["title"]))
        await done(params, call, {"sent": result["ok"], "to": sess["title"], "error": result.get("error"),
                                  "note": "it's working on it now; you'll announce when it's done"},
                   f"sent v{len(stats.draft_versions)} → {sess['title']}" if result["ok"] else f"failed: {result.get('error')}")

    async def session_status(params):
        call = track(params)
        import agent_sessions

        sess = stats.agent_session
        if not sess or not sess.get("pane"):
            await done(params, call, {"error": "the session isn't open in a herdr pane"}, "no pane")
            return
        status = await asyncio.to_thread(agent_sessions.agent_status, sess["pane"])
        screen = await asyncio.to_thread(agent_sessions.agent_screen, sess["pane"], 30)
        await done(params, call, {"session": sess["title"], "status": status, "screen_tail": screen[-1500:]},
                   f"{sess['title']}: {status}")

    async def preview_voices(params):
        """Each voice introduces itself, numbered; then the current voice is restored."""
        call = track(params)
        names = list(catalog)
        wanted = [str(v).lower() for v in (params.arguments.get("voices") or []) if str(v).lower() in catalog]
        if not wanted:
            count = int(params.arguments.get("count") or 4)
            others = [n for n in names if n != stats.voice]
            start = preview_cursor["i"] % max(1, len(others))
            wanted = (others[start:] + others[:start])[:count]
            preview_cursor["i"] = start + count
        stats.last_preview = wanted
        for i, name in enumerate(wanted, 1):
            await params.llm.push_frame(TTSUpdateSettingsFrame(delta=voice_settings_delta(args, name)))
            pretty = name.replace("_", " ").title()
            await params.llm.push_frame(TTSSpeakFrame(f"Number {i}. Hi, I'm {pretty}. This is how I'd sound."))
        await params.llm.push_frame(TTSUpdateSettingsFrame(delta=voice_settings_delta(args, stats.voice)))
        stats.preview_restore = True
        await done(params, call, {"previewed": [{"number": i, "voice": n} for i, n in enumerate(wanted, 1)]},
                   ", ".join(f"{i}:{n}" for i, n in enumerate(wanted, 1)))

    schemas = [
        FunctionSchema("update_draft", "Have the writer update the draft from the conversation (runs in the background).",
                       {"note": {"type": "string", "description": "one sentence: what the user just said or changed"}},
                       ["note"]),
        FunctionSchema("revert_draft", "Go back to an earlier draft version (default: the previous one).",
                       {"version": {"type": "integer", "minimum": 1}}, []),
        FunctionSchema("copy_to_clipboard", "Check the draft, then copy it to the clipboard. force skips the check.",
                       {"force": {"type": "boolean"}}, []),
        FunctionSchema("session_brief", "Summary, key points, results and next steps of the user's coding-agent "
                       "session.", {}, []),
        FunctionSchema("ask_session", "Answer a question about the coding-agent session from its transcript.",
                       {"question": {"type": "string"}}, ["question"]),
        FunctionSchema("list_sessions", "List the user's recent coding-agent sessions.", {}, []),
        FunctionSchema("use_session", "Switch to another coding-agent session (number from list_sessions, or a name).",
                       {"number": {"type": "integer", "minimum": 1}, "name": {"type": "string"}}, []),
        FunctionSchema("start_screen_recording", "Start recording the full screen with desktop audio (no microphone). "
                       "Only when the user asks to record.", {}, []),
        FunctionSchema("stop_screen_recording", "Stop the screen recording and save it to ~/Videos.", {}, []),
        FunctionSchema("send_to_session", "Type the current draft into the user's coding-agent session and submit it "
                       "(the agent starts working). Only after the user says to send it.",
                       {"even_if_busy": {"type": "boolean"}}, []),
        FunctionSchema("session_status", "What the coding-agent session is doing right now: idle, working, blocked, "
                       "and the end of its screen.", {}, []),
        FunctionSchema("set_voice",
                       "Change your speaking voice. Use number to pick from the last preview. Omit both to move to "
                       "the next voice. make_default saves it as the user's main voice for next time.",
                       {"voice": {"type": "string", "enum": list(catalog)},
                        "number": {"type": "integer", "minimum": 1},
                        "make_default": {"type": "boolean"}}, []),
        FunctionSchema("preview_voices",
                       "Play short numbered samples of other voices (each introduces itself), then return to the "
                       "current voice. Omit voices to hear the next few.",
                       {"voices": {"type": "array", "items": {"type": "string", "enum": list(catalog)}},
                        "count": {"type": "integer", "minimum": 1, "maximum": 8}}, []),
    ]
    import shutil as _shutil

    optional = {"start_screen_recording": bool(_shutil.which("omarchy")),
                "stop_screen_recording": bool(_shutil.which("omarchy")),
                "send_to_session": bool(_shutil.which("herdr")), "session_status": bool(_shutil.which("herdr"))}
    schemas = [sc for sc in schemas if optional.get(sc.name, True)]
    for name, handler in (("update_draft", update_draft), ("revert_draft", revert_draft),
                          ("copy_to_clipboard", copy_to_clipboard), ("session_brief", session_brief),
                          ("ask_session", ask_session), ("list_sessions", list_sessions),
                          ("use_session", use_session), ("start_screen_recording", start_screen_recording),
                          ("stop_screen_recording", stop_screen_recording),
                          ("send_to_session", send_to_session), ("session_status", session_status)):
        llm.register_function(name, handler)
    llm.register_function("set_voice", set_voice)
    llm.register_function("preview_voices", preview_voices)

    @llm.event_handler("on_function_calls_started")
    async def _lead_in(service, function_calls):
        # The model often calls tools without saying anything first; fill the silence ourselves.
        if stats.response_full.strip():
            return
        for fc in function_calls:
            if fc.function_name == "copy_to_clipboard" and ((fc.arguments or {}).get("force")
                                                            or not COPY_REQUEST.search(stats.last_user_text or "")):
                continue
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


AUDITION_LINE = (
    "Hi, I'm Passe. Oh, nice, the tests are passing again. Hmm… want me to tighten up that prompt before we send it?"
)


AUDITION_LINE_ORPHEUS = (
    "Hi, I'm Passe. Oh, nice <chuckle> the tests are passing again. Hmm… want me to tighten up that prompt "
    "before we send it?"
)


AUDITION_LINE_WARM = (
    "Mm… there you are. <chuckle> I was starting to think you'd forgotten about me. So… what are we building "
    "tonight?"
)


TIC_LINES = [
    "Mm… there you are.",  # 1: spelled hum, as before (filtered here only if --tics-raw is off)
    "<sigh> There we go. It finally works.",
    "<chuckle> You're trouble, you know that?",
    "Oh… now that's clever.",
    "Well… let's see what we've got here.",
    "Look at you, fixing things before I even ask.",
    "<sigh> Okay. Tell me what's bothering you.",
    "Oh… <chuckle> I like where this is going.",
]


def audition_tics(args: argparse.Namespace) -> None:
    """Warm Tara says each tic, numbered, so you can tell which ones sound natural."""
    import numpy as np
    import pyaudio

    import orpheus_tts

    proc = orpheus_tts.start_server(args.orpheus_vram)
    fx = orpheus_tts.voice_fx("warm", args.pitch)
    voice = VOICE_CATALOG["orpheus"].get(args.voice or "tara", ("tara", None))[0]
    pa = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True,
                  output_device_index=args.output_device)
    try:
        for i, line in enumerate(TIC_LINES, 1):
            text = line if i == 1 else orpheus_tts.HUM.sub("<sigh> ", line)
            print(f"{i}. {text}", flush=True)
            audio = fx(orpheus_tts.synthesize(text, voice), whole_clip=True)
            out.write(np.concatenate([audio, np.zeros(int(24000 * 0.9), np.float32)]).astype(np.float32).tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        pa.terminate()
        if proc is not None and not args.keep_orpheus:
            orpheus_tts.stop_server(proc)
    print("Line 1 is the old spelled-out hum, for comparison; the live voice now turns it into a <sigh>.")


def audition(args: argparse.Namespace) -> None:
    """Play voices of the --tts engine saying the same line, so you can pick one by ear."""
    import numpy as np
    import pyaudio

    names = list(VOICE_CATALOG[args.tts])
    if args.audition_voices:
        names = [n.strip() for n in args.audition_voices.split(",")]
    if args.tts == "orpheus":
        import orpheus_tts

        proc = orpheus_tts.start_server(args.orpheus_vram)
        if not args.keep_orpheus:
            import atexit

            atexit.register(orpheus_tts.stop_server, proc)
        sr = 24000

        fx = orpheus_tts.voice_fx(args.style, args.pitch)
        line = AUDITION_LINE_WARM if args.style == "warm" else AUDITION_LINE_ORPHEUS

        def synth(name):
            audio = orpheus_tts.synthesize(line, VOICE_CATALOG["orpheus"][name][0])
            return fx(audio, whole_clip=True) if fx else audio
    elif args.tts == "pocket":
        from pocket_tts import TTSModel

        model = TTSModel.load_model(language="english")
        sr = model.sample_rate

        def synth(name):
            st = model.get_state_for_audio_prompt(VOICE_CATALOG["pocket"][name][0])
            return np.concatenate([c.numpy().reshape(-1) for c in model.generate_audio_stream(st, AUDITION_LINE)])
    else:
        from kokoro_onnx import Kokoro
        from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

        os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")
        k = Kokoro(str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"), str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))
        sr = 24000
        names = names if args.audition_voices else sorted(v for v in k.voices if v[:2] in ("af", "bf"))

        def synth(name):
            vid, lang = VOICE_CATALOG["kokoro"].get(name, (name, "en-gb" if name.startswith("b") else "en-us"))
            return k.create(AUDITION_LINE, voice=vid, speed=args.speed, lang=lang)[0]

    pa = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True, output_device_index=args.output_device)
    try:
        for i, name in enumerate(names, 1):
            print(f"{i:2}/{len(names)}  {name}", flush=True)
            audio = np.asarray(synth(name), dtype=np.float32)
            out.write(np.concatenate([audio, np.zeros(int(sr * 0.8), np.float32)]).tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        pa.terminate()
    print(f"Pick one with --tts {args.tts} --voice NAME.")


def analyze(which: str) -> None:
    """Readable timeline + diagnosis of a session's event log."""
    import json
    from collections import Counter

    files = sorted(LOG_DIR.glob("spike-*.jsonl"))
    path = Path(which).expanduser() if which != "latest" else (files[-1] if files else None)
    if not path or not path.exists():
        raise SystemExit(f"no event log found in {LOG_DIR}")
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"{path}  ·  {len(events)} events  ·  {events[-1]['t'] if events else 0:.0f}s\n")

    def say(t, who, text, note=""):
        text = conversational(text) if who.startswith("passe") else text
        print(f"{t:7.1f}  {who:7} {text}" + (f"   ({note})" if note else ""))

    for e in events:
        k, t = e["kind"], e["t"]
        if k == "transcript":
            say(t, "you", e["text"])
        elif k == "local_command":
            say(t, "", f"⏹ '{e['text'].strip()}': stopped locally, not sent to the LLM")
        elif k == "pause":
            print(f"{t:7.1f}  ──────── ⏸ paused ────────")
        elif k == "resume":
            print(f"{t:7.1f}  ──────── ▶ unpaused ────────")
        elif k == "ignored_paused":
            say(t, "", f"(ignored, paused) {e['text']}")
        elif k == "session_end":
            print(f"{t:7.1f}  ──────── session {e['n']} ended by you ────────")
        elif k == "ignored_idle":
            say(t, "", f"(ignored, no session) {e['text']}")
        elif k == "session_start":
            print(f"{t:7.1f}  ──────── session {e['n']} ────────")
        elif k == "vad_start" and e.get("bot_speaking"):
            say(t, "", f"⚡ you started talking over Passe (mic vol {e['mic_volume']:.2f})")
        elif k == "tool_call":
            say(t, "", f"→ {e['name']}({', '.join(f'{a}={v!r}' for a, v in e['args'].items() if a != 'text')})")
        elif k == "tool_result":
            say(t, "", f"← {e['summary']}", f"{e['secs']:.2f}s")
        elif k == "tts_speak":
            say(t, "passe*", e["text"], "canned line")
        elif k == "assistant_turn":
            full = (e.get("full") or "").strip()
            if not full:  # canned lines (greetings, goodbyes, lead-ins) are already shown as passe*
                continue
            if e.get("interrupted"):
                cut = e.get("est_cut_chars")
                if cut is not None and cut < len(full):
                    nxt = full.find(" ", cut)
                    cut = len(full) if nxt == -1 else nxt
                    say(t, "passe", f"{full[:cut].strip()} ⟂ [unsaid: {full[cut:].strip()}]",
                        f"cut ~{e.get('heard_secs')}s in")
                else:
                    say(t, "passe", full + " ⟂", "interrupted")
            else:
                say(t, "passe", full)
        elif k in ("error", "log_error"):
            say(t, "!", e.get("error") or e.get("message", ""))

    # ---- diagnosis
    print("\nDiagnosis")
    barges = [i for i, e in enumerate(events) if e["kind"] == "vad_start" and e.get("bot_speaking")]
    kinds = Counter()
    false_ones = []
    for i in barges:
        t0 = events[i]["t"]
        words = next((e["text"] for e in events[i + 1:] if e["kind"] == "transcript" and e["t"] - t0 < 6), None)
        if not words:
            kinds["no words (noise/cough/other voice)"] += 1
            false_ones.append((t0, events[i]["mic_volume"]))
        elif STOP_COMMAND.match(words):
            kinds["stop command"] += 1
        else:
            kinds["real (you said something)"] += 1
    print(f"  talked over Passe {len(barges)}×: " + (", ".join(f"{n} {k}" for k, n in kinds.items()) or "never"))
    if false_ones:
        print("    without words at: " + ", ".join(f"{t:.0f}s (vol {v:.2f})" for t, v in false_ones[:10]))
    sessions = sum(e["kind"] == "session_start" for e in events) + 1
    ends = [e["text"].strip() for e in events if e["kind"] == "session_end"]
    print(f"  sessions {sessions}; ended by: " + ("; ".join(f"'{x}'" for x in ends) or "never"))
    lat = [e["total"] for e in events if e["kind"] == "latency"]
    if lat:
        print(f"  turn latency median {median(lat) * 1000:.0f} ms, p90 {pct(lat, 0.9) * 1000:.0f} ms, "
              f"slowest {max(lat) * 1000:.0f} ms")
    ttfb = {}
    for e in events:
        for proc, v in (e.get("ttfb") or {}).items() if e["kind"] == "latency" else []:
            ttfb.setdefault(proc.split("#")[0], []).append(v)
    for proc, vs in ttfb.items():
        print(f"    {proc:24} first output median {median(vs) * 1000:.0f} ms")
    issues = Counter(e.get("message", e.get("error", ""))[:90] for e in events
                     if e["kind"] in ("error", "log_error", "log_warning"))
    if issues:
        print("  warnings/errors: " + "; ".join(f"{n}× {m}" for m, n in issues.most_common(5)))
    empty = sum(1 for e in events if e["kind"] == "user_turn" and not e.get("text")) - sum(
        1 for e in events if e["kind"] in ("local_command", "ignored_idle", "session_end", "pause", "resume",
                                           "ignored_paused"))
    if empty > 0:
        print(f"  {empty} user turns ended with no transcript (speech the recogniser didn't catch, or stops)")


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
    global _LOG
    if not args.no_log:
        _LOG = EventLog(LOG_DIR / f"spike-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        logger.add(lambda m: ev("log_" + m.record["level"].name.lower(), message=m.record["message"][:500]),
                   level="WARNING")
        ev("start", args={k: v for k, v in vars(args).items()})

    if args.tts_device == "cuda":
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"  # kokoro-onnx; skips the TensorRT probe
        import onnxruntime

        if hasattr(onnxruntime, "preload_dlls"):
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
        TTSAudioRawFrame,
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
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
    from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.workers.runner import WorkerRunner

    stats.state = "starting"
    if args.tts == "orpheus":  # boot the voice server while everything else loads
        import orpheus_tts

        print("Starting the Orpheus voice server in the background (first start ~5 min, later ~1–2 min)…")
        stats.server_proc = orpheus_tts.launch_server(args.orpheus_vram)
        if not args.keep_orpheus:  # also covers Ctrl+C while it's still booting
            import atexit

            atexit.register(orpheus_tts.stop_server, stats.server_proc)
    print(f"Loading models on {args.device.upper()} (Whisper {args.stt_model}, {args.tts} voice, Silero, "
          "smart-turn)…", flush=True)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=args.input_device,
            output_device_index=args.output_device,
            audio_in_filter=make_input_filter(args, denoise=(args.stt_audio == "denoised")),
        )
    )
    stt = WhisperSTTService(
        device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
        settings=WhisperSTTService.Settings(model=args.stt_model, language=Language.EN),
    )
    # The first inference is slow (kernel loading / allocation); do it now, not on your first sentence.
    if stt._model is not None:
        segs, _ = stt._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        list(segs)

    extra = {} if args.reasoning == "none" else {"reasoning_effort": args.reasoning}
    llm = BasetenLLMService(
        api_key=load_api_key(),
        settings=BasetenLLMService.Settings(model=args.model, max_completion_tokens=600, extra=extra),
    )
    stats.voice = default_voice(args)
    tts = make_tts(args, stats.voice, stats)

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

    from pipecat.frames.frames import TranscriptionFrame as _Transcription
    from pipecat.frames.frames import VADUserStoppedSpeakingFrame as _VADStopped
    from pipecat.processors.frame_processor import FrameDirection as _Dir
    from pipecat.processors.frame_processor import FrameProcessor as _Proc

    class LivePartials(_Proc):
        """Shows what you're saying while you say it: re-transcribes the current utterance every ~0.5 s."""

        def __init__(self):
            super().__init__()
            self._pre: deque = deque(maxlen=25)  # ~0.5 s of audio from before the VAD confirmed speech
            self._buf = bytearray()
            self._active = False
            self._busy = False
            self._last = 0.0

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                if self._active:
                    self._buf += frame.audio
                    now = time.monotonic()
                    if not self._busy and now - self._last > partial_every and len(self._buf) > frame.sample_rate:
                        self._last, self._busy = now, True
                        asyncio.create_task(self._partial(bytes(self._buf)))
                else:
                    self._pre.append(frame.audio)
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                self._active, self._buf = True, bytearray(b"".join(self._pre))
            elif isinstance(frame, _VADStopped):
                self._active = False
            elif isinstance(frame, _Transcription) and direction == _Dir.DOWNSTREAM:
                stats.live_user_done = (stats.live_user_done + " " + frame.text.strip()).strip()
                stats.live_user = ""
            await self.push_frame(frame, direction)

        async def _partial(self, pcm: bytes):
            try:
                x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768

                def run():
                    segs, _ = stt._model.transcribe(x, language="en", beam_size=1, without_timestamps=True,
                                                    condition_on_previous_text=False)
                    return " ".join(seg.text.strip() for seg in segs)

                text = await asyncio.to_thread(run)
                if self._active:  # don't overwrite once the final transcript is on its way
                    if text != stats.live_user:
                        ev("partial", text=text)
                    stats.live_user = text
            finally:
                self._busy = False

    partial_every = 0.5 if args.device == "cuda" else 1.5  # spare the CPU
    live_partials = LivePartials()

    vad_denoise = None
    if args.denoise == "rnnoise" and args.stt_audio == "raw":
        from pipecat.processors.frame_processor import FrameProcessor as _FP

        class DenoiseForVAD(_FP):
            """STT has already buffered the raw audio; give the voice detector a denoised copy."""

            def __init__(self):
                super().__init__()
                self._filter = make_input_filter(args, denoise=True, gain=False)
                self._started = False

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, InputAudioRawFrame):
                    if not self._started:
                        await self._filter.start(frame.sample_rate)
                        self._started = True
                    out = await self._filter.filter(frame.audio)
                    frame.audio = (out + b"\x00" * len(frame.audio))[: len(frame.audio)]
                    vad_tracker.update(frame.audio, frame.sample_rate)
                    stats.mic_volume = vad_tracker.volume  # the meter shows what the gate sees
                await self.push_frame(frame, direction)

        vad_denoise = DenoiseForVAD()

    holder: dict = {}
    if args.session != "none":
        import agent_sessions

        stats.agent_session = agent_sessions.find_session(args.session)
        if stats.agent_session:
            print(f"Agent session: {stats.agent_session['title']} ({stats.agent_session['id'][:8]})")
            ev("agent_session", id=stats.agent_session["id"], title=stats.agent_session["title"])
    if args.drafter_model:
        import prompt_quality

        prompt_quality.DRAFTER.update({"model": args.drafter_model, "reasoning_effort": None})
    tools = setup_tools(llm, stats, args, holder)
    if stats.agent_session:  # brief it in the background so the first "let's draft a prompt" is quick
        asyncio.get_running_loop().create_task(holder["ensure_brief"]())
    voice_line = (f"Right now you're using the voice called {stats.voice}; set_voice can switch to any of: "
                  f"{', '.join(voice_catalog(args))}.")
    prompt = SYSTEM_PROMPT.replace("{VOICE_LINE}", voice_line).replace(
        "{EXPRESSION_LINE}", ORPHEUS_EXPRESSION if args.tts == "orpheus" else "")
    if args.style == "warm":
        prompt += WARM_STYLE
    context = LLMContext(messages=[{"role": "system", "content": prompt}], tools=tools)
    holder["context"] = context

    import re as _re

    import random

    from pipecat.frames.frames import TTSSpeakFrame

    start_phrase = _re.compile(r"\b(start|begin|new|open)\b[\w\s]{0,12}\bsession\b", _re.I)

    def start_new_session():
        context.set_messages([{"role": "system", "content": prompt}])
        stats.session_n += 1
        stats.session_start = time.monotonic()
        stats.session_active = True
        stats.transcript.clear()
        stats.draft_versions.clear()
        stats.draft_copied = 0
        stats.draft_note_pending = None
        stats.last_check = None
        stats.draft_sent_to = None
        stats.sent_version = 0
        stats.tool_calls.clear()

    class SessionGate(_Proc):
        """Between sessions only "start session" gets through; "new session" mid-session starts fresh."""

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, _Transcription) and direction == _Dir.DOWNSTREAM:
                wants_new = bool(start_phrase.search(frame.text))
                if stats.paused:
                    if RESUME.match(frame.text):
                        stats.paused = False
                        user_agg._params.empty_user_turn = None
                        stats.transcript.append(("you", f"{frame.text.strip()}  [unpaused]"))
                        ev("resume", text=frame.text)
                        await self.push_frame(TTSSpeakFrame(RESUMED_LINE), direction)
                        return
                    if not END_SESSION.search(frame.text):  # "end session" still works while paused
                        stats.ignored_while_paused += 1
                        user_agg._params.empty_user_turn = None
                        ev("ignored_paused", text=frame.text)
                        return
                    stats.paused = False
                elif stats.session_active and PAUSE.match(frame.text):
                    stats.paused = True
                    user_agg._params.empty_user_turn = None
                    stats.transcript.append(("you", f"{frame.text.strip()}  [paused]"))
                    ev("pause", text=frame.text)
                    await self.push_frame(TTSSpeakFrame(PAUSED_LINE), direction)
                    return
                if stats.session_active and END_SESSION.search(frame.text) and not wants_new:
                    stats.session_active = False
                    user_agg._params.empty_user_turn = None
                    stats.transcript.append(("you", f"{frame.text.strip()}  [session ended]"))
                    ev("session_end", n=stats.session_n, text=frame.text)
                    await self.push_frame(TTSSpeakFrame(random.choice(SESSION_GOODBYES)), direction)
                    return
                if stats.session_active and STOP_COMMAND.match(frame.text):
                    # "stop", "wait"…: your speech already cut Passe off; don't send it to the LLM at all.
                    user_agg._params.empty_user_turn = None
                    stats.transcript.append(("you", f"{frame.text.strip()}  [stopped]"))
                    ev("local_command", command="stop", text=frame.text)
                    return
                if not stats.session_active and not wants_new:
                    ev("ignored_idle", text=frame.text)
                    stats.ignored_while_idle += 1
                    user_agg._params.empty_user_turn = None  # no "sorry, I missed that" between sessions
                    return
                if wants_new:
                    # A local command: reset and greet right away, no LLM round trip.
                    start_new_session()
                    ev("session_start", n=stats.session_n, text=frame.text)
                    user_agg._params.empty_user_turn = None  # this turn reaches the LLM with no text
                    await self.push_frame(TTSSpeakFrame(random.choice(SESSION_GREETINGS)), direction)
                    return
                user_agg._params.empty_user_turn = empty_turn_config
                if stats.draft_note_pending:  # safe point: no tool call is in flight at the start of a user turn
                    context.add_message({"role": "developer", "content": stats.draft_note_pending})
                    stats.draft_note_pending = None
            await self.push_frame(frame, direction)

    session_gate = SessionGate()
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy

    # Your turn ends only when smart-turn thinks you're done AND you've been quiet for --turn-grace seconds,
    # so a short pause between sentences ("It's Python. … Keep the API stable.") doesn't get answered early.
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

    # When smart-turn thinks you're mid-thought it waits up to stop_secs of silence (default 3 s) before giving up.
    stop = [TurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=args.turn_max_wait)))]
    if args.turn_grace > 0:
        stop.append(SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=args.turn_grace))
    turn_strategies = UserTurnStrategies(
        start=[MinWordsUserTurnStartStrategy(min_words=3)] if args.interrupt == "words" else None, stop=stop)
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
    empty_turn_config = user_agg._params.empty_user_turn  # restored when a session starts

    @user_agg.event_handler("on_user_turn_stopped")
    async def _user_turn(_agg, _strategy, message):
        ev("user_turn", text=message.content, strategy=type(_strategy).__name__ if _strategy else None)
        stats.last_user_text = message.content or stats.last_user_text
        if (message.content and stats.draft_versions and stats.session_active and not stats.paused
                and not CONTROL_UTTERANCE.search(message.content)):
            # Keep the draft in step with every correction, not only when the talker remembers update_draft.
            # The user's words are in the context by now, so the writer sees them.
            asyncio.get_running_loop().call_later(0.2, holder["request_draft"], f"User said: {message.content}")
        stats.live_user = stats.live_user_done = ""
        if message.content:
            stats.turn().user_text = message.content
            sim = f"  [{stats.voice_last_sim:.2f}]" if voice_lock and stats.voice_last_sim is not None else ""
            stats.transcript.append(("you", message.content + sim))

    @user_agg.event_handler("on_user_backchannel")
    async def _backchannel(_agg, _message):
        stats.backchannels += 1

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def _assistant_turn(_agg, message):
        heard_s = (stats.last_barge - 0.2 - args.output_latency - stats.response_speech_start
                   if message.interrupted and stats.last_barge and stats.response_speech_start else None)
        ev("assistant_turn", spoken=message.content, full=stats.response_full, interrupted=message.interrupted,
           heard_secs=round(heard_s, 2) if heard_s is not None else None,
           est_cut_chars=int(heard_s * stats.tts_cps) if heard_s is not None else None)
        stats.live_bot = ""
        full_resp, stats.response_full = stats.response_full, ""  # a canned line next must not inherit it
        if full_resp.strip():
            stats.last_assistant_text = full_resp
        tr = stats.turn()
        tr.bot_text = message.content or ""
        if message.interrupted:
            tr.interrupted = True
            stats.interruptions += 1
            from pipecat.frames.frames import TTSUpdateSettingsFrame as _TTSUpdate

            if stats.preview_restore:  # a cut-off preview may have dropped the switch back
                await llm.push_frame(_TTSUpdate(delta=voice_settings_delta(args, stats.voice)))
            full = (full_resp or message.content or "").strip()
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
        stats.preview_restore = False
        if message.content:
            stats.transcript.append(("passe", message.content))

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def _breakdown(_obs, b):
        ev("latency", total=round(b.total_secs, 3), end_of_turn=b.user_turn_secs,
           ttfb={m.processor: round(m.duration_secs, 3) for m in b.ttfb})
        tr = stats.turn()
        tr.total_s = b.total_secs
        tr.end_of_turn_s = b.user_turn_secs

    from pipecat.audio.volume import AudioVolumeTracker

    mic_tracker = AudioVolumeTracker()
    vad_tracker = AudioVolumeTracker()

    from pipecat.frames.frames import (
        ErrorFrame as _Err,
        FunctionCallInProgressFrame as _FnStart,
        InterruptionFrame as _Interrupt,
        LLMFullResponseEndFrame as _LLMEnd,
        TTSSpeakFrame as _Speak,
        TTSTextFrame as _TTSText,
    )

    class Monitor(BaseObserver):
        def _log(self, f):
            if _LOG is None:
                return
            if isinstance(f, VADUserStartedSpeakingFrame):
                ev("vad_start", bot_speaking=stats._bot_speaking, mic_volume=round(float(stats.mic_volume), 3),
                   dbfs=round(float(stats.mic_dbfs), 1), session_active=stats.session_active)
            elif isinstance(f, _VADStopped):
                ev("vad_stop")
            elif isinstance(f, UserStartedSpeakingFrame):
                ev("user_start")
            elif isinstance(f, UserStoppedSpeakingFrame):
                ev("user_stop")
            elif isinstance(f, _Interrupt):
                ev("interruption", bot_speaking=stats._bot_speaking)
            elif isinstance(f, _Transcription):
                ev("transcript", text=f.text)
            elif isinstance(f, LLMFullResponseStartFrame):
                msgs = context.get_messages()
                last_user = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), None)
                ev("llm_start", messages=len(msgs), last_user=last_user)
            elif isinstance(f, _LLMEnd):
                ev("llm_end", text=stats.response_full)
            elif isinstance(f, _Speak):
                ev("tts_speak", text=f.text)
            elif isinstance(f, _TTSText):
                ev("tts_text", text=f.text)
            elif isinstance(f, BotStartedSpeakingFrame):
                ev("bot_start")
            elif isinstance(f, BotStoppedSpeakingFrame):
                ev("bot_stop")
            elif isinstance(f, _FnStart):
                ev("function_call", name=f.function_name)
            elif isinstance(f, _Err):
                ev("error", error=str(f.error)[:500])

        async def on_push_frame(self, data: FramePushed):
            if not data.first_push:
                return
            f = data.frame
            if isinstance(f, InputAudioRawFrame):
                x = np.frombuffer(f.audio, np.int16).astype(np.float32)
                rms = float(np.sqrt((x**2).mean())) if len(x) else 0.0
                stats.mic_dbfs = 20 * np.log10(max(rms, 1.0) / 32768)
                if vad_denoise is None:
                    mic_tracker.update(f.audio, f.sample_rate)
                    stats.mic_volume = mic_tracker.volume
                return
            now = time.monotonic()
            if data.direction == _Dir.DOWNSTREAM or not isinstance(
                    f, (VADUserStartedSpeakingFrame, _VADStopped, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                        _Interrupt, BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                self._log(f)  # broadcast frames travel both ways; log each once
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
                stats.resp_audio_secs = 0.0
            elif isinstance(f, TTSAudioRawFrame):
                secs = len(f.audio) / 2 / f.sample_rate
                stats.resp_audio_secs += secs
                stats.tts_audio_total += secs
                if stats.tts_audio_total > 3 and stats.tts_text_chars:
                    stats.tts_cps = stats.tts_text_chars / stats.tts_audio_total
            elif isinstance(f, _TTSText):
                stats.tts_text_chars += len(f.text) + 1
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
        [transport.input(), stt, live_partials, *([voice_lock] if voice_lock else []), session_gate,
         *([vad_denoise] if vad_denoise else []),
         user_agg, llm, tts, transport.output(), assistant_agg]
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
        if stats.recording_since:  # don't leave a recording running after Passe exits
            import subprocess

            subprocess.run(["omarchy", "capture", "screenrecording", "--stop-recording"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Stopped the screen recording that was still running (saved to ~/Videos).")
        if args.tts == "orpheus":
            if args.keep_orpheus:
                print("Orpheus server left running (~11 GB VRAM). Stop it: pkill -f 'vllm serve unsloth/orpheus'")
            else:
                import orpheus_tts

                orpheus_tts.stop_server(stats.server_proc)
                print("Stopped the Orpheus server.")
        print_summary(stats, args)
        if _LOG is not None:
            print(f"Event log: {_LOG.path}  (analyze: uv run spike.py --analyze)")


if __name__ == "__main__":
    a = parse_args()
    if a.device == "cuda":
        _ensure_cuda12_libs()  # may re-exec this process with the CUDA 12 libraries on the loader path
    if a.list_devices:
        list_devices()
        sys.exit(0)
    if a.mic_test:
        mic_test(a)
        sys.exit(0)
    if a.calibrate:
        calibrate(a)
        sys.exit(0)
    if a.audition:
        audition(a)
        sys.exit(0)
    if a.audition_tics:
        audition_tics(a)
        sys.exit(0)
    if a.analyze:
        analyze(a.analyze)
        sys.exit(0)
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
