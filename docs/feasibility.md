# Feasibility

Assessment as of 2026-11-01. **Verdict: go.** Technically feasible on the local machine, cheap to run with
the cascade backend, and the main unknown is whether talking prompts through actually beats typing them, which
M2 is designed to answer early.

## Constraints

- One local machine: Arch Linux + Omarchy (Hyprland), RTX 5090 (32 GB), headphones as the normal audio setup.
- Limited budget. LLMs are hosted (Baseten subscription); local LLMs are ruled out as too resource-hungry.
- Solo development, part-time, with Claude Code assisting.

## Resource use

| Resource | Use |
|---|---|
| VRAM | ~4–5 GB (Whisper large-v3-turbo, Kokoro, smart-turn). Unloaded after idle. |
| CPU | Silero VAD + audio I/O, negligible. |
| Network | One talker request per turn; briefer requests in the background. |

## Cost model

Assumed usage: **10 conversations/day × ~3 min, 22 workdays/month** → ~100 talker turns/day, ~10 briefs/day.

Baseten list prices (late 2026, check before relying on them):

| Model | Input $/1M | Output $/1M |
|---|---|---|
| gpt-oss-120b | 0.10 | 0.50 |
| GLM-5.3 | 1.40 | 4.40 |
| Kimi K3 | 3.00 | 15.00 |

### Cascade backend (default)

| Item | Per unit | Per month |
|---|---|---|
| STT / TTS / VAD / turn | local | $0 |
| Talker: gpt-oss-120b, ~4k in / 200 out per turn | ~$0.0005/turn | ~$1 |
| Briefer: gpt-oss-120b, ~30k in / 1k out per brief | ~$0.004/brief | ~$1 |
| **Total** | | **~$2** (~$10 at 5× usage) |

For comparison, GLM-5.3 for both roles is about $25/month before prompt caching. Kimi K3 as the briefer
alone is about $30–45/month. Both are kept as optional upgrades, e.g. only for `polish`.

### Grok Voice realtime backend (optional)

| Item | Per month |
|---|---|
| $0.08/min × 30 min/day × 22 days | **~$55** |

Per-minute billing charges for silence too, and composing a prompt involves a lot of thinking aloud and pausing. If
used: push-to-talk only, and close the session when idle. It still needs the briefer for session context.

**Guardrail:** monthly cap per backend in config; paid backends turn off when the cap is reached
(`budget_exceeded`).

## Technical risks

| Area | Confidence | Notes / mitigation |
|---|---|---|
| Local STT/TTS, Pipecat, Baseten integration | High | Off-the-shelf; Pipecat has a Baseten LLM service. |
| ~1 s from end of speech to first audio | High | One network hop. Measured in M0; filler acknowledgment as a fallback. |
| Interruption feel on headphones | Medium-high | No echo problem; tune duck/confirm thresholds. |
| End of turn while thinking aloud | Medium | smart-turn, "hold on" command, silence backstop, push-to-talk fallback. |
| Draft quality from gpt-oss (reasoning low) | Medium | Measured in M0; `polish` with a stronger model as an escape hatch. |
| Tool choice accuracy (OS actions) | Medium | Keep ~10–15 tools loaded; specific tools over shell; eval set of 20+ commands. |
| Context adapters | Medium | Undocumented formats; fixture tests; `pane_capture` fallback. |
| `wtype` delivery | Medium | Refocus before typing; prefer herdr/tmux. |
| Grok backend | Medium-low | Context limits, drafting through function calls, billing details all untested. |
| **Product fit: do I actually use it?** | **Unknown** | The real risk. Get a crude version into daily use by end of M2. |

## Effort (rough, part-time)

| Milestone | Estimate |
|---|---|
| M0 spike + benchmarks | 1–2 days |
| M1 core, daemon, CLI | 2–3 days |
| M2 voice in the daemon | 3–5 days → **daily use ~1.5–2 weeks in** |
| M3 TUI + Claude Code context | 3–4 days |
| M4 Omarchy widget + pack | 2–3 days |
| M5 adapters, protocol v1 | ongoing |
| Grok backend (optional) | 2–3 days |

## Go/no-go checkpoints

| When | Kill / change course if |
|---|---|
| **After M0** | Cascade can't do ≤ 1.5 s turns or reliable interruption → consider Grok as the default (with push-to-talk), or stop. |
| **After M2 + one week of use** | You still type most prompts → pause the project; the idea doesn't fit your workflow. |
| **Ongoing** | Monthly spend exceeds the cap regularly → revisit model choices. |
