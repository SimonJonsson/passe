# Passe spike: development notes

Design decisions, measurements and history, in roughly the order they were built. For setup and usage, see
[README.md](README.md).


Throwaway proof of concept for turn-taking, interruptions, token use and GPU load (see `docs/roadmap.md` → M0).

```
mic (webcam) → Silero VAD → smart-turn v3 → faster-whisper large-v3-turbo (GPU)
            → Baseten gpt-oss-120b → Kokoro (GPU) → headphones
```

## Run

```sh
cd spikes/m0
uv run spike.py                      # chat; Ctrl+C quits and prints a summary
uv run spike.py --list-devices       # then --input-device N / --output-device N
uv run spike.py --interrupt words    # 'mhm' no longer interrupts (needs 3+ words; acts after you pause)
uv run spike.py --reasoning medium   # or --model zai-org/GLM-5.3-Flash --price-in … --price-out …
uv run spike.py --tts-device cpu     # compare Kokoro on CPU
uv run spike.py --mic-test           # live mic level + VAD state through the input chain (--seconds N for a log)
uv run spike.py --denoise none --gain-db 0   # raw mic, no input chain
uv run spike.py --speed 1.3          # faster voice (default 1.15)
uv run spike.py --tts orpheus        # default: Orpheus 3B on the GPU, expressive (<laugh>, <sigh>…)
uv run spike.py --tts pocket         # Kyutai Pocket TTS: CPU, ~50 ms to first audio
uv run spike.py --audition --tts pocket --audition-voices alba,anna,eve   # hear voices without chatting
uv run spike.py --calibrate          # quiet / you / background speech → suggested flags
uv run spike.py --voice-lock         # enroll your voice at start; turns that aren't you are dropped
```

The API key is read from `$BASETEN_API_KEY`, or the file `~/baseten_apikey` (override with `$BASETEN_API_KEY_FILE`).

**Nothing is saved at runtime**: no logs, recordings or transcripts. Disk use is only the venv and model weights in
`~/.cache/huggingface` (Whisper) and `~/.cache/pipecat/kokoro-onnx`.

## Mic input chain

The webcam mic is quiet and reverberant: speech peaked around volume 0.70–0.76 against the VAD's 0.60 gate, and the
pipeline never detected a turn. Mic audio now goes through **RNNoise → +12 dB gain** before VAD and STT. Measured
on 2026-11-01: room noise 0.44 → **0.00**, speech 0.74 → **0.85**, speech detected ~90% of each talking second.

## Voice lock (other people in the room)

RNNoise removes noise but keeps *any* speech. `--voice-lock` records ~10 s of your voice at startup (in memory
only) and checks every transcribed turn against it with a WeSpeaker speaker-embedding model (sherpa-onnx, CPU,
~30 ms). Turns that aren't you are dropped and shown dimmed as `other`; if they cut Passe off, it resumes.
Scripted test with two synthetic voices: you matched 0.82–0.88, the other voice 0.41–0.48, so all four
of their lines were rejected at the 0.60 threshold. `--calibrate` suggests the floor and threshold for your room.

## Prompt workflow and sessions

Passe's main job: talk through what you want your coding agent to do, and it shapes a written prompt with
you (one clarifying question at a time). The **prompt** panel shows the current draft and its version.

| Say | What happens |
|---|---|
| "I want a prompt for Claude Code to …" | questions, then `update_draft` (goal, context, constraints, done criteria) |
| "go back" / "undo that" | `revert_draft` (history is never rewritten) |
| "copy it" / "copy to clipboard" | `copy_to_clipboard` via `wl-copy` (real) |
| "stop" / "wait" / "hold on" | handled locally: Passe stays quiet, nothing reaches the LLM, the session continues |
| "end session" / "close session" | handled locally (the model can't end sessions); canned goodbye, then idle |
| "start session" / "new session" | instant local greeting; conversation, transcript and draft start fresh |

Turn-taking: your turn ends only when smart-turn thinks you're done **and** you've been quiet for
`--turn-grace` seconds (default 0.7), so a pause between sentences doesn't get answered early.

## Prompt quality

- **Talker vs writer.** The fast talker (gpt-oss-120b) only converses: one diagnostic question at a time, no
  invented requirements. A background writer (DeepSeek-V4.1-Flash, `prompt_quality.DRAFTER`, `--drafter-model`)
  rewrites the draft from the whole conversation after every user turn: only what you said or confirmed,
  corrections applied, unknowns as *Open questions*. Instructions to Passe ("copy it", "go back", "switch
  voices"…) never feed the draft.
- **Check before copying.** "Copy it" runs a quick rubric check: invented or stale content is removed
  automatically. Only real gaps are put to you ("fill them in, or copy it as is?"); `--copy-bar` sets the score
  below which it asks.
- **Guards in code.** `update_draft` is ignored after a command, `copy_to_clipboard` only runs if you asked to
  copy, and skipping the check needs an explicit "as is" / "anyway".

`uv run prompt_eval.py latest|all|FILE…` scores sessions on **prompt** (faithful, goal, context, constraints,
verifiable, open questions), **conversation** (one question, diagnostic, concise, responsive, no invention,
flow) and **tools** (requests acted on; errors; attempts caught by guards). `uv run sim.py voice|observability`
replays two of your real sessions for before/after comparisons.

| 2026-11-01 | prompt | conversation | tools |
|---|---|---|---|
| your sessions, old setup | 6.8 | 7.4 | 7.7 |
| same conversations replayed, new setup | **8.6** | **9.4** | **10.0** |

Writer model comparison (same conversations, same judge): gpt-oss medium 7.0 overall / 3.8 faithful,
gpt-oss high 8.1 / 5.8, GLM-5.3 8.9 / 9.5 (11.5 s), DeepSeek-V4.1-Flash 8.9 / 9.5 (4 s, shortest drafts).

## Session awareness (/btw)

Passe sees your coding-agent session: by default the most recently active Claude Code session
(`--session latest|<id prefix>|<title word>|none`). `agent_sessions.py` lists sessions (herdr pane titles where
available; `uv run agent_sessions.py`), reads Claude Code transcripts and has a background model brief them
(summary, key points, results, open problems, next steps; cached until the session moves on).

| Say | What happens |
|---|---|
| "let's draft a prompt" | `session_brief` → Passe says where the session stands and suggests the next step → drafting starts |
| "what did it just do?", "why did it pick that?" | `ask_session`, answered from the transcript |
| "which sessions are there?" / "use the trading one" | `list_sessions` / `use_session` |

The writer grounds drafts in the brief, and every draft opens with a **Read first** line pointing the agent at
this Passe conversation's log. Ideas are grounded: Passe checks what the session already has before suggesting
anything, one suggestion at a time. Replay: `uv run sim.py fromsession`, which scored prompt 9.2 ·
conversation 10 · tools 9.5 on 2026-11-01.

### Sending into the session (herdr)

| Say | What happens |
|---|---|
| "send it" / "send it to Claude" | `send_to_session`: the draft is typed into the session's herdr pane and submitted (`herdr agent prompt`; multi-line arrives as one message) |
| "what's it doing?" | `session_status`: idle / working / blocked plus the end of its screen |
| (after a send) | Passe follows the agent and says "*<session>* is done" (or "…is waiting for your approval") at the next quiet moment |

Guards in code: it only sends when you said "send"/"submit", or answered yes ("yeah", "sounds good") right after
Passe asked "Send it?". It won't send to an agent waiting at a dialog, asks first if the agent is mid-task, and
won't re-send the same version unless you say "again". Sessions map to panes even after Claude Code compacts a
conversation into a new transcript (matched by title). Verified against a real Claude Code pane, 2026-11-02.
`sim.py send` uses a fake herdr, so scripted runs never type into your sessions.

### Honest drafting

The writer takes several seconds (6–27 s in practice), so after `update_draft` Passe says it's *updating*, never
that it's done. When a draft you asked for lands, it says "Draft's updated." at the next quiet moment, or not at
all if you're already talking. The scorecard flags "said the draft was updated before it was".

### Screen recording

"Record my screen" / "stop recording": the full screen plus **desktop audio** (Passe's voice, anything playing),
no microphone: what you say is already in the transcript and the event log. Saved to `~/Videos`; ● REC shows
in the header, and it stops cleanly if Passe exits.

## Observability

Every run writes an event log to `~/.local/state/passe/spike-<time>.jsonl` (`--no-log` to disable): speech
starts/stops with mic level, partial and final transcripts, barge-ins, LLM requests and replies, tool calls,
TTS lines, sessions, latency and warnings/errors.

    uv run spike.py --analyze           # timeline + diagnosis of the latest run
    uv run spike.py --analyze FILE

The diagnosis splits barge-ins into real ones, stop commands and ones with no words (noise, coughs, other
voices), lists what ended each session, and summarises latency per stage.

`uv run sim.py stop|prompt` replays a scripted conversation with synthesized speech (no mic, silent output,
fake clipboard, logs in `.sim/`) and prints the analysis.

## Orpheus (default voice engine)

Orpheus 3B speaks in SNAC audio codes; a local vLLM server (separate `.venv-vllm`) generates them and
`orpheus_tts.py` decodes them on the GPU. Female voices only: tara, leah, jess, mia, zoe. Passe may
use one emotion tag per reply (`<laugh>`, `<chuckle>`, `<sigh>`, `<gasp>`, `<groan>`, `<yawn>`).

Measured 2026-11-01: ~155 ms to first audio warm, 2.3× realtime, ~11.5 GB VRAM total. The first server start
takes ~5 min (6.6 GB download + compile). The server is left running between spike runs; stop it with
`pkill -f 'vllm serve unsloth/orpheus'` or run with `--stop-orpheus`.

Setup: `uv venv --python 3.12 .venv-vllm && VIRTUAL_ENV=.venv-vllm uv pip install vllm`

The dashboard shows what you're saying as you say it (Whisper re-transcribes the utterance every ~0.5 s)
and, while Passe talks, how far it has got through its reply, with a progress bar.

Other tools are voice-only (`preview_voices`, `set_voice`). The earlier dummy tools (git, tests, screen recording,
draft) are in git history at d8364bc.

### Style: warm (default)

`--style warm` (the default; `--style natural` for the plain persona) makes Passe low, warm, unhurried and a
little teasing, with signature tics (`<sigh>`, `<chuckle>`, "Oh…", "Well…", "There we go."). It also adds warm,
zero-latency post-processing (low shelf, softer highs, gentle compression). Spelled-out hums ("mm", "mhm", "hmm")
sound robotic in Orpheus, so the model is told not to write them, and a filter maps any that slip through to
`<sigh>`, "Yeah," or "Well…". Hear the tics with `uv run spike.py --audition-tics`. Pitch shifting (`--pitch`)
is audition-only for now, because streaming it adds ~1 s of delay.

### Consistent delivery

Orpheus samples every request independently, so synthesizing sentence by sentence made Tara drift in pitch,
energy and pace within one reply. By default the spike now sends Orpheus a whole reply (split into ~400-char
paragraphs for long ones). `uv run consistency_eval.py` measures it, here with 3 renderings of a five-sentence reply:

| Configuration | Voice identity ↑ | Pitch drift ↓ | Loudness drift ↓ |
|---|---|---|---|
| per sentence (before) | 0.561 | 33.3 Hz | 3.3 dB |
| per sentence, lower temperature | 0.566 | 28.6 Hz | 3.1 dB |
| **whole reply (now)** | **0.600** | **18.3 Hz** | **1.9 dB** |

Orpheus gets an audio-token budget proportional to the text (a fixed 1600 cut long replies off at 19.5 s), and the server runs with an 8k context. Cost: Passe waits for the LLM to finish the reply before speaking, about +250–350 ms per turn in the scripted
runs. `--tts-chunk sentence` switches back.

## Voice library

Say "let me hear some voices": each voice introduces itself, numbered, then Passe switches back. Say
"number two, make it my main voice" and it's saved to `~/.config/passe/spike-voice.json` (the only
setting the spike persists), so the next `uv run spike.py` starts with that engine and voice.

Your own clips: put a few seconds of clean speech as `~/.config/passe/voices/<name>.wav` and Pocket TTS clones it
into a selectable voice. Cloning needs Kyutai's gated weights: accept the terms on the `kyutai/pocket-tts`
Hugging Face page and run `uvx hf auth login` once. Until then, the clips are skipped.

## Reading the dashboard

| Column | Meaning |
|---|---|
| total | You stop talking → Passe starts speaking (pipeline side) |
| end-of-turn | VAD silence + smart-turn decision + STT finalisation |
| stt / llm / tts | Time to first result of each stage |
| tok in / out / reason | Tokens for that turn (reasoning = hidden thinking tokens) |
| interrupted | You talked over the reply |

Bluetooth headphones add roughly 150–250 ms of output latency on top of everything shown.

## Scripted run (synthesized speech in, silent output), 2026-11-01

| Metric | Kokoro on CPU | Kokoro on GPU |
|---|---|---|
| Median turn (stop talking → reply starts) | 935 ms | **686 ms** |
| p90 turn | 3,386 ms | **959 ms** |
| End of turn / STT / LLM first token | ~260 / ~260 / ~210–270 ms | same |
| TTS first audio | 290–2,330 ms | **44–144 ms** |
| Spike VRAM | 2.5 GB | 5.0 GB |
| GPU utilisation (avg) | ~0% | ~2% |
| Cost of a 4-turn conversation | $0.0004 | $0.0004 |

Setup notes: CTranslate2 needs CUDA 12 cuBLAS/cuDNN (pip wheels in the venv; the script re-execs with
`LD_LIBRARY_PATH` set). Kokoro on GPU uses `onnxruntime-gpu`, installed alongside `onnxruntime`.

## Conversational drafts (2026-11-03)

Drafts used to be structured markdown (Goal / Context / Requirements / Done when / Open questions). On request,
the writer now produces a conversational summary: a *Read first* line, two to four short paragraphs of prose (the
situation with concrete details, what we want, how we'll know it's done), and a short open-questions line.
The fromsession replay scored prompt 9.8 · conversation 9.3 · tools 10 in this format.

Send-to-session fixes from a live session: "Confirm." and "sent to…" are accepted as consent; sends can carry dictated
text; sessions named by voice are matched loosely ("scalping" finds "scalp") and `send_to_session(session=…)`
switches first; refused sends no longer retry in a loop; "stop session" ends the session. Orpheus moved to port
18100 (`PASSE_ORPHEUS_PORT` overrides) because another project held 8100.

## opencode sessions (2026-11-04)

`agent_sessions.py` reads opencode sessions too: herdr reports each opencode pane's `ses_…` id, and the session
lives in opencode's SQLite store (`~/.local/share/opencode/opencode.db`). It is opened read-only and only the
`session`, `message` and `part` tables are queried (the same file holds opencode's account tokens). Parts
become the same plain transcript as Claude Code's: text by role, tool calls as one-liners. Briefs, by-the-way
questions and herdr sends work the same for both kinds. Spoken session names now match on title words (weighted
double) and the project folder name, ranked by how many spoken words match; saying "opencode" or "claude" filters
by kind. Claude-only title de-duplication (for compacted conversations) no longer merges distinct opencode
sessions that share a title.

## macOS / Apple Silicon (2026-11-03)

Setup on an M3 (macOS 15): `pyrnnoise` is Linux-only (no macOS 15 wheel), so `--denoise` falls back to `none`;
Python is pinned to 3.12; `mlx-whisper` is installed on Apple Silicon (Pipecat's Whisper module requires it there).

Turn-taking was much worse than on the GPU box. The event log showed why: end of turn ~1.8 s, nearly all of it
speech recognition (faster-whisper `small.en` on the CPU, ~1.6 s for a 6.7 s utterance). Live partials made it
worse: they used the same model, and a final transcript that started while a partial ran took twice as long (3.3 s).

`--stt-engine auto` now picks **MLX on the Apple GPU** with `large-v3-turbo` (0.72 s for the same utterance), and
live partials run `base.en` on the CPU (0.36 s), so they no longer slow the final transcript (0.73 s alongside one;
partials on the GPU doubled it to ~1.5 s). Scripted `sim.py prompt` runs on the M3:

| STT | STT first output (median) | End of turn | Turn (median) |
|---|---|---|---|
| MLX `large-v3-turbo` (default) | 926 ms | 0.93 s | 1.42 s |
| MLX `small.en` | 385 ms | 0.75 s | 1.26 s |

End of turn can't go below VAD stop (0.2 s) + `--turn-grace` (0.5 s): `small.en` hits that floor, turbo is ~0.2 s
over it. Turbo stays the default for accuracy; `--stt-model mlx-community/whisper-small.en-mlx` trades it for speed.

herdr on macOS reports no `agent_session` for panes, so `agent_sessions.py` also maps a pane to its transcript by
working directory (and terminal title when several panes share one).
