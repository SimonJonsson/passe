# Passe

**A spoken `/btw` for your coding-agent session.** Press a key and start a side conversation about what
your agent is doing, without interrupting it. Passe has read the session, so you can ask what just
happened ("simplify that last answer for me"), think out loud about what to do next, and talk the next
prompt into shape. When you say *"okay, sounds good, let's use that"*, Passe hands the prompt to the agent.
It can also drive the operating system or just chat.

Named after Passepartout, the valet who handles the arrangements so the traveller doesn't have to.

> **Status:** planning. No code yet. Start with [`docs/roadmap.md`](docs/roadmap.md) → M0.

## At a glance

```
  you ──voice──▶  Passe  ──prompt──▶  Claude Code / Codex / aider / any terminal agent
       ◀─voice──    │
                    ├── reads the agent's session for context (like /btw)
                    ├── explains the agent's last answer in plain language
                    ├── edits a versioned draft you can see
                    └── runs OS actions through gated tool packs (Omarchy, tmux, …)
```

- **Agent-agnostic.** Context sources and delivery targets are adapters; pane capture works for any agent.
- **Several entrypoints, one engine.** The Omarchy (Hyprland) integration, the terminal TUI and the CLI are
  all clients of one daemon, talking a versioned protocol.
- **Fluid conversation.** Open mic, interrupt at any time, smart end-of-turn detection, nothing read aloud at
  length.
- **Cheap.** Speech-to-text and text-to-speech run locally on the GPU; one hosted LLM call per turn
  (gpt-oss-120b on Baseten, about $2/month at expected use).

## Documents

| Doc | What's in it |
|---|---|
| [`docs/features.md`](docs/features.md) | Feature list and the Omarchy widget (mockup + lifecycle) |
| [`docs/architecture.md`](docs/architecture.md) | System diagrams, every component explained, key flows, layout |
| [`docs/protocol.md`](docs/protocol.md) | Daemon ⇄ client protocol v0 (the integration surface) |
| [`docs/feasibility.md`](docs/feasibility.md) | Cost model, risks, effort, go/no-go checkpoints |
| [`docs/roadmap.md`](docs/roadmap.md) | Milestones M0–M5 with exit criteria |
| [`docs/decisions.md`](docs/decisions.md) | Decision log and open questions |

## Stack (planned)

Python 3.12+ · uv workspace · [Pipecat](https://docs.pipecat.ai) voice pipeline · Silero VAD ·
smart-turn · faster-whisper large-v3-turbo · Kokoro TTS · Baseten Model APIs (gpt-oss-120b) ·
Textual TUI · Hyprland + Waybar integration · optional Grok Voice realtime backend.
