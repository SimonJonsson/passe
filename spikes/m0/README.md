# Passe: a voice companion for your coding agent

Talk to Passe like a colleague. It helps you shape the next prompt for your coding agent (Claude Code, Codex…),
explains what your agent session has been doing, copies the prompt to your clipboard, and can even send it straight
into a Claude Code session. Speech recognition and the voice run on your own machine. The language model runs on
[Baseten](https://www.baseten.co), which needs an API key (see below).

> This is a prototype (the "M0 spike"). It works well enough to use every day, but expect rough edges. Design notes
> and measurements live in [NOTES.md](NOTES.md).

## What you need

| | |
|---|---|
| **OS** | Linux (built and tested on Arch / Omarchy, Wayland). macOS on Apple Silicon works (set up on an M3, macOS 15): speech recognition runs on its GPU via MLX, the voice on the CPU; mic noise suppression (RNNoise) is Linux-only, so it's off there. Windows: untested. |
| **Hardware** | A microphone and **headphones** (there's no echo cancellation, so speakers make Passe hear itself). Any modern CPU works. An NVIDIA GPU is optional and gives a more expressive voice (Orpheus). |
| **Software** | [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12+ for you), PortAudio, and a clipboard tool (`wl-copy`, `xclip`, `xsel` or `pbcopy`) |
| **Disk** | ~6 GB for the Python environment and models, plus ~15 GB more if you set up the GPU voice |
| **Account** | A **Baseten API key**, the one paid part (see below) |

## 1. Get a Baseten API key

Passe's "brain" runs on Baseten's hosted models; nothing else in Passe is paid.

1. Sign up at [baseten.co](https://www.baseten.co) and add a payment method.
2. Create an API key in your Baseten account settings (API keys). Copy it; it's usually shown only once.
3. Give it to Passe in **one** of these ways:
   ```sh
   # a) a file in your home directory (simplest; Passe reads ~/baseten_apikey by default)
   printf '%s' 'YOUR_KEY_HERE' > ~/baseten_apikey && chmod 600 ~/baseten_apikey

   # b) an environment variable, e.g. in ~/.bashrc / ~/.zshrc / fish config
   export BASETEN_API_KEY='YOUR_KEY_HERE'

   # c) a key file somewhere else
   export BASETEN_API_KEY_FILE=~/secrets/baseten
   ```
   Never commit the key, and don't paste it into chats or issues.

**Models it uses** (Baseten "Model APIs", nothing to deploy):

| Model | What for | Price (late 2026, per 1M tokens) |
|---|---|---|
| `openai/gpt-oss-120b` | the conversation itself, the quick check before copying | $0.10 in / $0.50 out |
| `deepseek-ai/DeepSeek-V4.1-Flash` | writing the prompt draft, summarising agent sessions | $0.30 in / $1.20 out |

**What it costs:** a conversation is typically **1–5 cents**. Building and testing every feature of this spike over
three days came to $0.84 in total. Check your usage on the Baseten dashboard.

**What gets sent to Baseten:** the *text* of what you say (your audio never leaves your machine), Passe's replies,
the prompt draft, and, if session awareness is on (it is by default when you use Claude Code), excerpts of
your Claude Code session transcript. Turn that last part off with `--session none`.

## 2. Install

```sh
# system packages: PortAudio (for the microphone) and a clipboard tool
sudo pacman -S portaudio wl-clipboard                    # Arch / Omarchy
sudo apt install portaudio19-dev python3-dev wl-clipboard  # Debian / Ubuntu (use xclip on X11)
brew install portaudio                                   # macOS (pbcopy is built in)

# uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Passe
git clone https://github.com/SimonJonsson/passe.git
cd passe/spikes/m0
uv sync          # creates .venv and installs everything (a few GB: PyTorch, Whisper, voices…)
```

## 3. Check your microphone

```sh
uv run spike.py --mic-test --seconds 10
```

Talk normally while it runs. You should see `speech` rise above 0.5 in the seconds you're talking. If not, see
[Troubleshooting](#troubleshooting). `uv run spike.py --list-devices` shows your audio devices; pick them with
`--input-device N` / `--output-device N`.

## 4. Run it

```sh
uv run spike.py
```

It picks the best mode for your machine automatically:

| | CPU (no NVIDIA GPU) | NVIDIA GPU |
|---|---|---|
| Speech recognition | Whisper `small.en` (Apple Silicon: `large-v3-turbo` on the GPU via MLX) | Whisper `large-v3-turbo` |
| Voice | **Pocket TTS**: natural female voices, made for CPUs | **Orpheus**: very expressive, can laugh and sigh (needs the extra setup below) |
| Typical reply delay | ~1.5–3 s on a fast CPU, more on older ones; ~1.4 s on an M3 | ~1–1.7 s |

The first start downloads models (a few minutes). A dashboard opens with the conversation, the prompt draft,
tool activity and latency. **Ctrl+C quits.**

## Things to say

| Say | What happens |
|---|---|
| "I want a prompt for Claude Code to …" | Passe asks what's missing, one question at a time, and writes the draft in the **prompt** panel |
| "Let's draft a prompt." / "What should we tell it next?" | Passe reads your Claude Code session, says where it stands, suggests a next step and drafts it |
| "What did it just do?" / "Why did it pick that?" | answered from your Claude Code session (by-the-way questions) |
| "Go back" / "undo that" | previous draft version |
| "Copy it" | quick quality check, then the prompt is on your clipboard. If it finds gaps it asks; "copy it as is" skips them |
| "Send it to the session" | types the prompt into your Claude Code session and submits it (needs herdr, see below) |
| "Send 'run the tests' to scalp" | sends exactly what you dictated, to the session you named |
| "Find the scalping conversation" / "use the trading one" | switches to another session; names are matched loosely |
| "Stop" / "wait" | Passe stops talking; nothing is sent to the model |
| "Pause" … "unpause" | Passe ignores everything until you unpause (no tokens spent) |
| "End session" (or "stop session") / "start session" | close the conversation / start fresh with an empty draft |
| "Let me hear some voices" / "number two, make it my main voice" | voice samples; pick one and it's remembered |

Talk naturally; you can interrupt Passe at any time.

## Useful options

| Option | What it does |
|---|---|
| `--device cpu` / `--device cuda` | force CPU or GPU mode (default: automatic) |
| `--tts pocket \| kokoro \| orpheus` | voice engine (Orpheus needs a GPU) |
| `--voice NAME` | starting voice, e.g. `--voice eve` (try `--audition` to hear them all) |
| `--style natural` | plain persona instead of the default warm, playful style |
| `--session none` | don't read any Claude Code session (nothing from it is sent to Baseten) |
| `--session NAME` | attach a specific session by title word or id (default: the most recently active) |
| `--turn-grace 0.8` | wait longer before answering if Passe keeps cutting you off (default 0.5 s) |
| `--min-volume 0.5` | for quiet microphones |
| `--voice-lock` | learns your voice at start and ignores other people in the room (`--calibrate` helps tune it) |
| `--no-log` | don't write the event log |

## Optional extras

**Agent session awareness:** automatic if you use [Claude Code](https://claude.com/claude-code) or
[opencode](https://opencode.ai): Passe reads your most recently active session (Claude Code transcripts in
`~/.claude/projects/`, opencode's local database, opened read-only). Name a session by voice, e.g. "use the
opencode one" or "switch to the rules engine session"; titles and project folders are matched loosely.

**Sending prompts into Claude Code:** install [herdr](https://github.com/omacom-io/herdr) and run your Claude Code or opencode
sessions in herdr panes. Then "send it" types the prompt into the session and Passe tells you when the agent is done. Without herdr,
use "copy it" and paste.

**Screen recording:** on [Omarchy](https://github.com/basecamp/omarchy), "record my screen" / "stop recording" records the screen
with desktop audio (no microphone) to `~/Videos`.

**GPU voice (Orpheus):** with an NVIDIA GPU (~12 GB free VRAM):
```sh
uv venv --python 3.12 .venv-vllm && VIRTUAL_ENV=.venv-vllm uv pip install vllm   # ~8 GB, separate on purpose
uv run spike.py --tts orpheus     # first start downloads ~6.6 GB and takes ~5 min; later ~30 s
```
The Orpheus server starts and stops with Passe.

## Troubleshooting

| Problem | Try |
|---|---|
| It doesn't react to me | `uv run spike.py --mic-test`. If "voice" goes up but "filtered" stays low, use `--min-volume 0.5`, or raise your mic level (`wpctl set-volume @DEFAULT_SOURCE@ 100%` on PipeWire) |
| It mishears words | a closer microphone helps most (webcam mics pick up a lot of room echo) |
| It answers before I've finished | `--turn-grace 0.8` |
| It hears other people / the TV | `--voice-lock` (run `--calibrate` first) |
| Replies feel slow on CPU | expected on older CPUs. `--tts kokoro` is lighter but sounds flatter |
| `pyaudio` fails to install | install PortAudio first (see Install) |
| "copy it" fails | install a clipboard tool: `wl-clipboard` (Wayland), `xclip` (X11) |
| "Orpheus needs an NVIDIA GPU" | you're on CPU: just use the default (`--tts pocket`) |
| "Orpheus server exited: Address already in use" | another program uses Orpheus's port (18100): `PASSE_ORPHEUS_PORT=18200 uv run spike.py` |
| Bluetooth headphones feel laggy | Bluetooth adds ~0.2 s; wired is snappier |

## Logs and privacy

- **Event log:** every run writes a text log (what was heard, replies, drafts, tool calls, timings) to
  `~/.local/state/passe/spike-<time>.jsonl`. No audio is ever saved. `uv run spike.py --analyze` turns the
  latest log into a readable timeline. `--no-log` turns it off.
- **Settings:** your chosen main voice is saved in `~/.config/passe/spike-voice.json`.
- **Sent to Baseten:** see [the API key section](#1-get-a-baseten-api-key).

## For developers

| Tool | What it does |
|---|---|
| `uv run spike.py --analyze [LOG]` | timeline and diagnosis of a session |
| `uv run prompt_eval.py latest` | scores a session: prompt quality, conversation quality, tool-calling accuracy |
| `uv run sim.py prompt\|voice\|observability\|fromsession\|send\|stop\|pause` | replays a scripted conversation with synthesized speech (no mic, fake clipboard and herdr) |
| `uv run stt_eval.py` | compares speech-recognition setups on your own voice |
| `uv run consistency_eval.py` | measures how consistent the GPU voice is across a reply |
| `uv run tone_shift.py latest` | finds tone shifts in Passe's replies |
| `uv run agent_sessions.py` | lists the Claude Code sessions Passe can see |

How it's built, and why: [NOTES.md](NOTES.md).
