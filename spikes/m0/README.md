# M0 spike: local voice loop

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

## Tools

Spoken lead-ins ("Checking git."), one-sentence results, and a live tool + draft panel in the dashboard.

| Tool | What it does | Real? |
|---|---|---|
| `git_status`, `search_session` | canned answers after 0.4 s / 1.5 s | dummy |
| `run_tests` | 6 s, **runs in the background**: keep talking, Passe reports when done | dummy |
| `set_volume` | reports success, changes nothing | dummy |
| `update_draft`, `finalize_prompt` | drive the "prompt to send" panel; finalize is a dry run | display only |
| `set_voice` | "change voices" → "Okay." → switch → "Is this better?" (British `bf_emma` / American `af_heart`) | **real** |
| `start_screen_recording`, `stop_screen_recording` | `omarchy capture screenrecording --fullscreen` with desktop + mic audio → `~/Videos` | **real** |

Screen recording is the one thing that writes a file. If the spike exits mid-recording it stops the recording cleanly.

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
