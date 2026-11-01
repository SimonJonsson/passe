# Decisions

A log of what was decided and why. Newest at the bottom. Change a decision by adding a new entry that
supersedes the old one, rather than editing history.

| # | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-11-01 | Name: **Passe** (from Passepartout). | |
| D2 | 2026-11-01 | **Agent-agnostic**: context sources, delivery and resolvers are adapters; `pane_capture` as the universal fallback. | Must work with Claude Code, Codex and others. |
| D3 | 2026-11-01 | **Python 3.12+**, uv workspace with `passe-core`, `passe-tui`, `passe-omarchy`. | Python ecosystem for voice/ML; packages keep the separation enforceable. |
| D4 | 2026-11-01 | **Engine + daemon + protocol; entrypoints are clients.** There is always a daemon, started on demand. | Several entrypoints (Omarchy, terminal, CLI) over one engine; models load once; one code path. |
| D5 | 2026-11-01 | **Cascade voice pipeline by default** (separate STT → LLM → TTS), Grok speech-to-speech as an optional backend. | The output is a precise text draft; cascade is inspectable, model-swappable and ~25× cheaper. |
| D6 | 2026-11-01 | **Pipecat** for the voice pipeline, wrapped behind Passe interfaces. | Interruptions, VAD, smart-turn and streaming are what it's built for; weeks saved. |
| D7 | 2026-11-01 | **Headphones** are the primary setup; no echo cancellation in v1. | Simpler, more reliable interruptions. Speakers profile later. |
| D8 | 2026-11-01 | **Local STT/TTS** on the RTX 5090: faster-whisper large-v3-turbo, Kokoro (Piper fallback), Silero, smart-turn. | Free, private, fast; ~4–5 GB VRAM. |
| D9 | 2026-11-01 | **Hosted LLMs via Baseten**; no local LLMs. | Local LLMs too resource-hungry; Baseten subscription available. |
| D10 | 2026-11-01 | **gpt-oss-120b** for the talker (reasoning low) and briefer (reasoning high); stronger model only for on-demand `polish`. | Tasks are simple (prompt tuning, OS actions, chat); ~$2/month. GLM-5.3 thinking-off remains a candidate in M0. |
| D11 | 2026-11-01 | **Talker / briefer split**: the talker sees a compact brief, not the raw agent session. | Keeps per-turn prompts small (3–5k tokens) and fast regardless of session length. |
| D12 | 2026-11-01 | Drafts are edited through **tool calls** (`update_draft`, `finalize`), not text tags. | With our own talker the tool set is fixed per conversation, so caching isn't hurt; structured and versionable. (Text tags were only needed for the earlier Claude-fork idea.) |
| D13 | 2026-11-01 | **No modes; tool packs** chosen per conversation (prompt, chat, omarchy, tmux…). | One fluid conversation; packs keep environment-specific actions out of the core. |
| D14 | 2026-11-01 | **Safety tiers + action gate**; confirmation only from the user; shell disabled by default. | Misheard speech + small model + OS access is the main safety risk. |
| D15 | 2026-11-01 | Delivery **pastes without Enter** by default; submitting needs confirmation. | Last look before the agent acts. |
| D16 | 2026-11-01 | **Omarchy hotkey pops up the Passe widget** (state + level meter, transcript, prompt to send), implemented as `passe-tui --widget` in a floating terminal. | One UI codebase for terminal and desktop; native layer-shell later if wanted. |
| D17 | 2026-11-01 | **Explain / recap** is a core feature: `session` tool pack with `recap(scope)`, backed by `ContextSource.exchanges()`; default scope is the last question/answer pair, wider scopes on request. | Hearing a dense agent answer rephrased is valuable on its own, and it leads straight into prompt crafting. |
| D18 | 2026-11-01 | **Product framing: a conversational `/btw` for the current agent session.** Understanding (ask, explain, recap), prompt crafting, and just thinking it through are outcomes of one side conversation, not modes. | Keeps the feature set coherent; the session is read-only and the only write is a confirmed prompt delivery. |
| D19 | 2026-11-01 | **English only.** | Parakeet (English-centric, streaming, very fast) becomes a real STT option; benchmark it against Whisper turbo in M0. Resolves Q1. |
| D20 | 2026-11-01 | **Entrypoints serve different needs; the daemon is the "motherbrain".** One central daemon owns the shared resources (models, audio devices, cost meter, briefs). Each entrypoint runs its **own conversations with its own profile** (tool packs, default mode, speech style). The Omarchy widget is for quick desktop/OS + prompt hand-off; the terminal TUI is for deep work alongside one agent session. | Different jobs want different tools and pacing. Centralizing resources keeps one model load, one mic owner and one budget. Resolves Q3. |

## Open questions

| # | Question | Blocks |
|---|---|---|
| ~~Q1~~ | ~~Language~~ → resolved by D19. | |
| Q2 | Remote audio (`audio.push`): ever needed? Method names are reserved; no work planned. | Nothing now |
| ~~Q3~~ | ~~Shared vs separate conversations~~ → resolved by D20. | |
| Q4 | Hotkey choices and voxtype's existing bindings. **Deferred.** | M4 |
| Q5 | Exact Baseten model ids and how to switch reasoning/thinking per model. **Deferred; expected to be easy.** | M0 |
