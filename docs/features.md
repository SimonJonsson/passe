# Features

What Passe does, from the user's point of view.

**The core idea: a conversational `/btw` for the current agent session.** `/btw` in Claude Code lets you ask a
side question with the session's context without disturbing the agent. Passe does the same by voice, as a
whole conversation instead of a single question, and for any agent. It reads the session (never writes
it), and the side conversation can end in one of three ways:

- **understanding:** explain or recap what the agent did (§3), or answer questions about the session
- **a prompt:** talk the next instruction into shape and deliver it (§2)
- **nothing:** you just wanted to think it through

These aren't modes. One conversation moves freely between them ("simplify that… okay, so tell it to go with
option B"). Each feature is tagged with the milestone that delivers it
(see [`roadmap.md`](roadmap.md)); **MVP** = usable after M3.

## 1. The Passe widget (Omarchy)

The main way to use Passe on the desktop. Press a key and a small floating window pops up over whatever
you're doing:

```
╭─ Passe ───────────────────────────── → Claude Code · ~/Repos/uploader ─╮
│  ● LISTENING   ▁▂▅▇▅▃▂▁▂▄▆▇▆▄▂                     open mic · $0.04/mo │
├────────────────────────────────────────────────────────────────────────┤
│  you   also make sure it keeps the existing tests passing              │
│  passe Added a test constraint. Anything about the retry limits?       │
├─ Prompt to send ───────────────────────────────────────── v3 · ↶ v2 ───┤
│  Add retry logic with exponential backoff to `upload_chunk` in         │
│  upload.py. Retry on network errors and 5xx only, max 5 attempts.      │
│+ Keep all existing tests in tests/test_upload.py passing; add tests    │
│+ for the retry path.                                                   │
├────────────────────────────────────────────────────────────────────────┤
│  "let's use that" / ⏎ send   ⇧⏎ send+submit   ⌫ revert   e edit   esc  │
╰────────────────────────────────────────────────────────────────────────╯
```

| Element | Behavior |
|---|---|
| **Header** | The target: which agent and directory the prompt will go to. Shows "no agent: chat/OS only" when none was found. |
| **State + level meter** | `LISTENING` (live mic level), `THINKING`, `SPEAKING` (output level), `PAUSED`, `MUTED`. Color-coded with Omarchy theme colors. |
| **Transcript strip** | The last few turns. Passe's text that hasn't been spoken yet is greyed out. |
| **Prompt to send** | The current draft, with the change from the previous version highlighted. The version number and an undo hint are on the right. |
| **Pending action** | When an action needs confirmation it replaces the footer: "Close 4 windows? yes / no". |
| **Footer** | Keyboard equivalents of everything you can say. |

**Lifecycle**

1. **Hotkey** (e.g. `SUPER+ALT+SPACE`) → resolve the target from the focused window → widget pops up → open
   mic starts. Holding the hotkey instead does push-to-talk for that turn.
2. Talk. The draft updates live in the widget, and Passe speaks short replies.
3. Finalize by voice ("let's use that") or `⏎`. The prompt is pasted into the agent pane, the widget shows
   "Sent to Claude Code ✓" for about a second, then closes.
4. `esc` or the hotkey again → cancel and close. The conversation is kept in history.
5. Unfocusing the widget doesn't close it. It stays pinned until finalized or cancelled.

The Waybar module mirrors the state while the widget is hidden, and clicking it brings the widget back.

**Implementation:** `passe-tui --widget` in a terminal with its own window class; Hyprland rules float,
pin, size and place it. Version 1 is a TUI, so the terminal widget and the desktop widget are the same code. A
native layer-shell overlay (GTK4 + gtk4-layer-shell) is a possible later upgrade for a frameless,
always-on-top look; it would be one more protocol client.

## 2. Prompt crafting

| Feature | Description | Milestone |
|---|---|---|
| **Talk a prompt into shape** | Describe what you want; Passe drafts it and refines it as you talk. **MVP** | M2 |
| **Session-aware** | Reads the agent's session (like `/btw`) through a brief, so the draft refers to real files, decisions and failures. **MVP** | M3 |
| **Lookup** | "What did we decide about the cache earlier?": searches the full transcript and repo when the brief isn't enough. | M3 |
| **Versioned draft** | Every edit is a version. "Go back to the previous one" / `⌫` reverts. | M2 |
| **Pinned intent** | Your first stated goal is kept as an anchor; Passe tells you if the draft starts drifting from it. | M2 |
| **Edit by hand** | `e` in the widget, or type in the TUI. Passe sees your edits on the next turn. | M3 |
| **Polish** | "Polish it": a stronger model does a final pass over the draft. | M3 |
| **Finalize by voice** | "Okay, sounds good, let's use that" → delivered. Pasted without Enter by default; "send it and run it" submits too, after confirmation. **MVP** | M2 |
| **Delivery anywhere** | herdr pane, tmux pane, any Wayland window (wtype), or the clipboard. | M3–M5 |
| **Prompt templates** (later) | Saved shapes such as "bug report", "refactor plan", "review request" that Passe fills in through conversation. | later |
| **Prompt history** (later) | Browse, reuse and re-send earlier finalized prompts. | later |

## 3. Explain and recap

Hear what the agent just said, rephrased. Useful when an answer is long, dense or full of jargon, or when
you've been away from the terminal.

> "Simplify the last answer for me." · "What did it just do?" · "Explain that like I haven't seen the code."
> · "Recap the last three exchanges." · "Summarize the whole session so far."

| Feature | Description | Milestone |
|---|---|---|
| **Ask about the session** | "Why did it pick SQLite?", "What's left to do?", "Did the tests pass?": answered from the brief, recent exchanges and `lookup`. Plain `/btw`, by voice. **MVP** | M3 |
| **Simplify the last exchange** | Reads the latest question/answer pair from the agent session (herdr/tmux pane or transcript) and says it back in plain language. **MVP** | M3 |
| **Adjustable scope** | `last` (default), `last N` exchanges, `since I last asked`, `whole session` (through the brief + `lookup`). Said in words or picked with a key. | M3 |
| **Listening-friendly output** | The one place Passe talks at length: chunked by sentence, interruptible, and code is described rather than read aloud ("it adds a retry loop around the upload call"). Tool calls are summarized as actions ("it edited three files and ran the tests; two failed"). | M3 |
| **Steer while listening** | "Slower", "skip the code", "go deeper on the second point", "what does backoff mean?": all are normal turns, since interruption works the same way here. | M3 |
| **Straight into a prompt** | "Okay, tell it to go with option B" → becomes a draft. Explaining and prompt crafting are the same conversation. | M3 |
| **Recap hotkey** | A separate Omarchy key (and `passe recap [--scope …]`) opens the widget and starts explaining the last exchange immediately, without you saying anything. | M4 |

## 4. Conversation

| Feature | Description | Milestone |
|---|---|---|
| **Fluid open mic** | Just talk. Smart end-of-turn detection waits while you pause to think. | M2 |
| **Interrupt anytime** | Talk over Passe and it stops within about 200 ms. "Mhm" doesn't stop it. | M2 |
| **Knows what you heard** | After an interruption, Passe's memory of the conversation is cut to what was actually spoken. | M2 |
| **Instant commands** | "stop", "wait", "hold on", "undo that", "yes"/"no" act immediately, without an LLM call. | M2 |
| **Short replies** | One or two spoken sentences, leading with what changed. The draft is shown, never read aloud. | M2 |
| **Push-to-talk** | For noisy rooms. Pressing it while Passe is speaking interrupts. | M2 |
| **Mix typing and talking** | Typed input and speech go into the same conversation. | M3 |
| **Plain chat** | Ask anything; no draft or agent needed. | M2 |

## 5. Operating system actions

| Feature | Description | Milestone |
|---|---|---|
| **Omarchy pack** | Volume and media, launch apps, switch workspaces, change theme, screenshots, notifications, allowlisted `omarchy-*` commands. | M4 |
| **Pane control** | Focus or list tmux/herdr panes, send text to a pane. | M5 |
| **Safety tiers** | Read-only and reversible actions run immediately; consequential ones ask first. A general shell tool is off by default. | M2 |
| **Action log** | Every action executed is logged. | M2 |

## 6. Integration surfaces

| Entrypoint | What you get | Milestone |
|---|---|---|
| **Omarchy** | Hotkey widget, Waybar status + mic indicator, notifications, theme-matched colors, installer. | M4 |
| **Terminal** | `passe-tui` in a split next to your agent: the same panels as the widget, plus typing. | M3 |
| **CLI** | `passe start / say / send / cancel / status / bench` for scripts and testing. | M1 |
| **Protocol** | Versioned JSON-RPC over a Unix socket, so you can write your own client (nvim, a web page, …). | M1 |

## 7. Operations

| Feature | Description | Milestone |
|---|---|---|
| **Model per role** | Talker, briefer and polish models configured separately; one-line swaps. | M1 |
| **Spend tracking + cap** | `passe status` shows the month's spend; paid backends turn off when the cap is reached. | M1 |
| **Benchmarks** | `passe bench`: time to first token per model and setting, tool-choice accuracy, turn time. | M0–M1 |
| **Idle unload** | GPU models are unloaded after inactivity. | M2 |
| **Grok realtime backend** (optional) | Speech-to-speech via Grok Voice with the same tools and widget. | later |
| **Speakers profile** (optional) | PipeWire echo cancellation for use without headphones. | later |
