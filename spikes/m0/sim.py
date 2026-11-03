"""Scripted end-to-end test: feeds synthesized speech into spike.py instead of the mic; output goes nowhere.

    uv run sim.py stop            # run a scenario (see SCENARIOS), then analyze its event log
    uv run sim.py prompt --tts kokoro

Everything stays in .sim/ (logs, fake clipboard, prefs): your real clipboard, main voice and logs are untouched.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path

HERE = Path(__file__).parent
SIM = HERE / ".sim"
SIM.mkdir(exist_ok=True)
(SIM / "bin").mkdir(exist_ok=True)
fake = SIM / "bin" / "wl-copy"
fake.write_text(f"#!/bin/sh\ncat > {SIM / 'clipboard.txt'}\n")
fake.chmod(0o755)
os.environ["PATH"] = f"{SIM / 'bin'}:{os.environ['PATH']}"

sys.path.insert(0, str(HERE))
import spike  # noqa: E402

spike._ensure_cuda12_libs()
spike.LOG_DIR = SIM / "logs"
spike.PREFS_FILE = SIM / "prefs.json"

import numpy as np  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402

os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")  # the user's synthetic voice; spike resets it
from pipecat.frames.frames import InputAudioRawFrame  # noqa: E402
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR  # noqa: E402
from pipecat.transports.base_input import BaseInputTransport  # noqa: E402
from pipecat.transports.base_output import BaseOutputTransport  # noqa: E402
from pipecat.transports.local import audio as la  # noqa: E402

SR = 16000
SCENARIOS = {
    "stop": [
        ("silence", 1.0),
        ("say", "Help me write a prompt for Claude Code to refactor the uploader module."),
        ("silence", 2.6),  # Passe should be mid-reply now
        ("say", "Stop."),
        ("silence", 4.0),
        ("say", "It's Python. Keep the public API stable and add tests."),
        ("silence", 10.0),
        ("say", "Wait."),
        ("silence", 3.0),
        ("say", "End session."),
        ("silence", 5.0),
        ("say", "Start session."),
        ("silence", 5.0),
    ],
    "pause": [
        ("silence", 1.0),
        ("say", "I need a prompt for Claude Code to add logging to the uploader."),
        ("silence", 8.0),
        ("say", "Pause."),
        ("silence", 4.0),
        ("say", "Honey, did you see my keys? They were on the table."),
        ("silence", 5.0),
        ("say", "Unpause."),
        ("silence", 5.0),
        ("say", "Use structured JSON logs and include the request ID."),
        ("silence", 10.0),
    ],
    "voice": [  # replays the user's 17:10 voice-consistency session
        ("silence", 1.0),
        ("say", "Let's draft a prompt."),
        ("silence", 7.0),
        ("say", "Currently the voice is inconsistent between segments. It changes accents and so on."),
        ("silence", 12.0),
        ("say", "Yeah, it changes tone as well, in a very unnatural way."),
        ("silence", 12.0),
        ("say", "It is using a single voice. It's just that the voice itself is quite inconsistent."),
        ("silence", 12.0),
        ("say", "No, it's just that it's switching personality, and it's kind of weird."),
        ("silence", 14.0),
        ("say", "Copy it to the clipboard."),
        ("silence", 16.0),
        ("say", "Copy it as is."),
        ("silence", 8.0),
    ],
    "observability": [  # replays the user's 17:10 observability session
        ("silence", 1.0),
        ("say", "Let's work a bit on how we can evaluate and have observability for this platform."),
        ("silence", 10.0),
        ("say", "We want to see how well the subjective performance is. We want to evaluate prompts and results, "
                "as well as conversational flow and conversational quality."),
        ("silence", 12.0),
        ("say", "This is quite hard, because everything in the conversation is subjective. My voice is subjective, "
                "so the comprehension of the model is subjective."),
        ("silence", 12.0),
        ("say", "What we need is a way to understand problems. I need to be able to report problems as well, "
                "without writing huge prompts."),
        ("silence", 14.0),
        ("say", "No, that's fine."),
        ("silence", 8.0),
        ("say", "Copy it to the clipboard."),
        ("silence", 16.0),
        ("say", "Copy it as is."),
        ("silence", 8.0),
    ],
    "fromsession": [  # "let's draft a prompt" with a coding-agent session attached; a by-the-way question
        ("silence", 1.0),
        ("say", "Let's draft a prompt."),
        ("silence", 16.0),
        ("say", "Yeah, go with that."),
        ("silence", 12.0),
        ("say", "By the way, what did it change about the turn-taking?"),
        ("silence", 14.0),
        ("say", "Okay. Copy it."),
        ("silence", 16.0),
        ("say", "Copy it as is."),
        ("silence", 8.0),
    ],
    "send": [  # draft from the session, a decoy that must NOT send, then an explicit send
        ("silence", 1.0),
        ("say", "Let's draft a prompt."),
        ("silence", 16.0),
        ("say", "Yeah, go with that."),
        ("silence", 12.0),
        ("say", "That looks good."),
        ("silence", 10.0),
        ("say", "Send it to the session."),
        ("silence", 18.0),
    ],
    "prompt": [
        ("silence", 1.0),
        ("say", "Hey, I want a prompt for Claude Code to add retry logic to the uploader."),
        ("silence", 9.0),
        ("say", "It's Python. Use exponential backoff, five attempts max, and only retry on network errors."),
        ("silence", 10.0),
        ("say", "Perfect. Copy it to the clipboard."),
        ("silence", 7.0),
        ("say", "End session."),
        ("silence", 6.0),
        ("say", "What's the weather like today?"),
        ("silence", 5.0),
        ("say", "Start session."),
        ("silence", 7.0),
    ],
}

name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "stop"
script = SCENARIOS[name]
kokoro = Kokoro(str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"), str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))


def speech(text: str) -> np.ndarray:
    samples, sr = kokoro.create(text, voice="am_michael", lang="en-us")
    x = np.interp(np.arange(0, len(samples), sr / SR), np.arange(len(samples)), samples)
    return (np.clip(x, -1, 1) * 20000).astype(np.int16)


async def feed(transport):
    chunk = SR // 50
    for kind, arg in script:
        audio = np.zeros(int(arg * SR), np.int16) if kind == "silence" else speech(arg)
        if kind == "say":
            print(f"[sim] you: {arg}", file=sys.stderr, flush=True)
        for i in range(0, len(audio), chunk):
            await transport.push_audio_frame(
                InputAudioRawFrame(audio=audio[i : i + chunk].tobytes(), sample_rate=SR, num_channels=1))
            await asyncio.sleep(0.02)
    os.kill(os.getpid(), signal.SIGINT)


async def in_setup(self, setup):
    await BaseInputTransport.setup(self, setup)


async def in_start(self, frame):
    await BaseInputTransport.start(self, frame)
    await self.set_transport_ready(frame)
    self._sim = asyncio.create_task(feed(self))


async def out_setup(self, setup):
    await BaseOutputTransport.setup(self, setup)


async def out_start(self, frame):
    await BaseOutputTransport.start(self, frame)
    await self.set_transport_ready(frame)


async def write(self, frame):
    await asyncio.sleep(len(frame.audio) / 2 / frame.sample_rate)  # "play" in real time, silently
    return True


la.LocalAudioInputTransport.setup = in_setup
la.LocalAudioInputTransport.start = in_start
la.LocalAudioOutputTransport.setup = out_setup
la.LocalAudioOutputTransport.start = out_start
la.LocalAudioOutputTransport.write_audio_frame = write

# Fake herdr: never send into a real session. Sends land in .sim/sent.txt; the "agent" works ~6 s, then idles.
import agent_sessions  # noqa: E402
import time as _time  # noqa: E402

_sent_at: list = []


def fake_send(pane, text):
    (SIM / "sent.txt").write_text(f"pane {pane}\n\n{text}\n")
    _sent_at.append(_time.monotonic())
    return {"ok": True}


def fake_status(pane):
    return "working" if _sent_at and _time.monotonic() - _sent_at[-1] < 6 else "idle"


agent_sessions.send_prompt = fake_send
agent_sessions.agent_status = fake_status
agent_sessions.agent_screen = lambda pane, lines=40: "❯ (fake agent screen)"
(SIM / "sent.txt").unlink(missing_ok=True)

sys.argv = ["spike.py", "--no-dashboard", "--output-latency", "0", *[a for a in sys.argv[2:]]]
try:
    asyncio.run(spike.main(spike.parse_args()))
except KeyboardInterrupt:
    pass
print("\n" + "=" * 100)
spike.analyze("latest")
