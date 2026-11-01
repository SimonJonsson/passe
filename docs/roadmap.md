# Roadmap

Milestones are ordered by risk: the hardest unknown, a fluid voice loop, comes first. Each has an exit
criterion. Estimates are in [`feasibility.md`](feasibility.md).

## M0: Voice and model spike *(throwaway, in `spikes/`)*

Goal: prove the conversation can feel fluid, and pick the talker.

- [ ] Pipecat local audio transport + Silero VAD + smart-turn + faster-whisper large-v3-turbo + Kokoro, on
      headphones.
- [ ] STT: Whisper large-v3-turbo vs Parakeet (English only, D19). Compare latency and accuracy on technical speech.
- [ ] Talker via Baseten (OpenAI-compatible). Candidates: gpt-oss-120b (reasoning low / medium), gpt-oss-20b
      if listed, GLM-5.3 (thinking off). Check `GET /v1/models` for real ids and fast/flash variants.
- [ ] Measure and write to `spikes/m0-results.md`:
  - [ ] interruption reaction (speech start → audio ducked): target < 200 ms
  - [ ] turn time (speech end → first audio): target ~1 s, ≤ 1.5 s
  - [ ] time to first token and tokens/s per model and setting
  - [ ] how often backchannels ("mhm") falsely interrupt
  - [ ] **tool-choice accuracy** on 20 sample voice commands (prompt edits + OS actions + chat)
  - [ ] draft quality on 5 realistic prompt-tuning conversations (subjective, side by side)
- [ ] Optional: a Grok Voice session with a brief + `update_draft`/`finalize` function calls; compare feel and cost.

**Exit:** a talker default chosen, latency targets met (or the go/no-go triggered).

## M1: Core, daemon, CLI *(text only)*

- [ ] uv workspace, `passe-core` package skeleton, Python 3.12+.
- [ ] Conversation manager + state machine + profiles (D20), draft store (versions, revert, pinned intent).
- [ ] Model gateway: per-role config, Baseten client, cost meter, budget cap.
- [ ] Talker with the `prompt` and `chat` tool packs; action gate with tiers.
- [ ] `passed` serving protocol v0; `passe` CLI (`start`, `say`, `send`, `cancel`, `status`, `bench`).
- [ ] Clipboard delivery.
- [ ] Fakes for STT/TTS/LLM; e2e tests driven by `passe say`; protocol contract tests.

**Exit:** a full prompt-crafting conversation by typing, ending with the prompt on the clipboard.

## M2: Voice in the daemon

- [ ] Pipecat cascade backend inside `passe-core/voice`, wrapped behind Passe interfaces.
- [ ] Audio focus across conversations.
- [ ] Open mic + push-to-talk; interrupt controller (duck → confirm → cancel, trim to what was spoken).
- [ ] Local command matcher (stop / wait / hold on / undo / yes / no).
- [ ] `audio.level` events; idle model unloading; `systemd --user` unit.
- [ ] Confirmation flow for consequential actions by voice.

**Exit:** using it for real for a week (with the CLI + clipboard). **Product-fit checkpoint.**

## M3: Terminal entrypoint + session context

- [ ] `passe-tui`: draft (with changes highlighted, version timeline), transcript, status line, pending action; typing edits;
      hold or toggle talk key depending on terminal; `--widget` mode.
- [ ] tmux + herdr resolvers and delivery.
- [ ] Context sources: `claude_code`, `pane_capture`; briefer with incremental cached briefs; `lookup`.
- [ ] `polish` on demand.
- [ ] **Explain / recap**: `exchanges()` in `claude_code` and `pane_capture` (herdr/tmux), `session` tool pack with
      `recap(scope)`, explaining-mode speech rules, `passe recap` CLI.

**Exit (MVP):** from a split pane next to Claude Code, talk a session-aware prompt into shape and have it
pasted into the agent pane.

## M4: Omarchy entrypoint

- [ ] Hotkey → hyprland resolver → conversation start → **Passe widget** pops up (window rules: float, pin, place).
- [ ] Widget lifecycle: closes itself after delivery; `esc`/hotkey cancels; Waybar click brings it back.
- [ ] Recap hotkey: opens the widget and explains the last exchange immediately.
- [ ] Waybar module with mic-open indicator (signal-refreshed).
- [ ] Theme colors from `~/.config/omarchy/current/theme`.
- [ ] `omarchy` tool pack: volume/media, launch app, workspaces, theme, screenshot, notify, allowlisted commands.
- [ ] Idempotent `install.sh` / `uninstall.sh`; check for keybind clashes (voxtype).
- [ ] `wtype` delivery for agents not in tmux or herdr.

**Exit:** look at any agent terminal, press the hotkey, talk, and "let's use that" pastes the prompt into it.

## M5: Breadth and stability

- [ ] Context sources: `codex`, `aider` (+ `opencode` if used).
- [ ] tmux/herdr tool pack.
- [ ] Freeze protocol **v1**.

## Later / maybe

- Grok Voice realtime backend.
- Speakers profile (PipeWire echo cancel).
- Native layer-shell widget (GTK4).
- Prompt templates and prompt history.
- `tools.register` for external tool packs; nvim client.
- Parakeet streaming STT as default if it wins the M0 comparison.
