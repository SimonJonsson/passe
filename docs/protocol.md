# Passe protocol v0 (draft)

The integration surface between `passed` and every client (CLI, TUI, widget, Waybar script, future
editors). Clients depend on this document, not on `passe-core`.

> v0 is unstable and may change freely until M5, when it is frozen as v1.

## Transport

- Unix domain socket at `$XDG_RUNTIME_DIR/passe/passed.sock`, permissions `0600`.
- **Newline-delimited JSON**, one JSON-RPC 2.0 message per line.
- Commands are JSON-RPC requests. Events are JSON-RPC notifications (`"method": "event"`) pushed to
  subscribed clients.
- Clients may start the daemon when the socket is missing: `passed --daemonize`, or
  `systemctl --user start passed`.

## Handshake

The first request on every connection:

```json
{"jsonrpc":"2.0","id":1,"method":"hello","params":{"protocol":0,"client":"passe-tui/0.1","subscribe":["*"]}}
```

```json
{"jsonrpc":"2.0","id":1,"result":{"protocol":0,"daemon":"passed/0.1","conversation":null}}
```

The daemon rejects any protocol version it doesn't support, with error `-32001 protocol_mismatch`.
`conversation` is the active conversation snapshot (see `status`) or `null`.

## Conversations

The daemon hosts several conversations (one or more per client, each with a profile). Commands that act on
a conversation take an optional `conversation` param; if it's omitted, the client's most recent conversation is used.
Every conversation event carries `conversation`. Only one conversation holds the **audio focus** at a time.

## Types

```ts
type State = "idle" | "listening" | "transcribing" | "thinking" | "speaking" | "ducked" | "awaiting_confirm"
type Mode  = "open_mic" | "ptt"

interface Target {
  agent: string | null          // "claude_code" | "codex" | "aider" | … | null
  session: string | null        // adapter-specific session ref
  cwd: string | null
  delivery: { kind: "herdr" | "tmux" | "wtype" | "clipboard", ref?: string }
  label: string                 // human label, e.g. "Claude Code · ~/Repos/uploader"
}

interface DraftVersion {
  version: number
  parent: number | null
  text: string
  author: "model" | "user"
  reason?: string               // e.g. "added test constraint"
}

interface Action {
  id: number
  tool: string                  // e.g. "omarchy.close_windows"
  args: object
  tier: "read" | "reversible" | "consequential" | "shell"
  summary: string               // human description shown for confirmation
}
```

## Commands

| Method | Params | Result | Notes |
|---|---|---|---|
| `hello` | `protocol, client, subscribe[]` | `protocol, daemon, conversation` | Must be first. |
| `status` | — | `{state, mode, target, draft, pending_action, spend}` | Snapshot for late joiners and Waybar. |
| `subscribe` | `events[]` (`"*"` or names) | `ok` | Changes the event filter. |
| `conversation.start` | `profile, target?: Target, mode?: Mode` | `{conversation}` | If no target is given: chat/OS only, clipboard delivery. Takes the audio focus unless `profile` is `headless`. |
| `conversation.list` | — | `[{conversation, profile, target, state, has_focus}]` | |
| `audio.focus` | `conversation` | `ok` | Move mic + output to this conversation. |
| `conversation.cancel` | — | `ok` | Ends without delivering. |
| `mode.set` | `mode` | `ok` | |
| `ptt.start` / `ptt.stop` | — | `ok` | `ptt.start` while speaking = interrupt. |
| `ptt.toggle` | — | `ok` | For terminals without key-release events. |
| `interrupt` | — | `ok` | Stop speaking or thinking now. |
| `mute` | `muted: bool` | `ok` | Mic off without ending the conversation. |
| `say` | `text` | `ok` | Typed user turn; same path as speech after STT. |
| `recap` | `scope?: "last" \| "last:N" \| "since_last_recap" \| "session"` | `ok` | Explain the agent's recent exchanges. Starts a conversation if none is active. |
| `draft.edit` | `text` | `{version}` | User-authored version. |
| `draft.revert` | `version` | `{version}` | Creates a new version copying the given one. |
| `draft.send` | `submit?: bool` | `{delivered: bool}` | Explicit finalize. `submit=true` goes through the action gate. |
| `action.confirm` / `action.reject` | `id` | `ok` | |
| `bench` | `{suite}` | report | Development use. |

**Error codes:** `-32001 protocol_mismatch`, `-32002 no_conversation`, `-32003 budget_exceeded`,
`-32004 delivery_failed`, `-32005 unknown_action`.

## Events

All events have the form `{"jsonrpc":"2.0","method":"event","params":{"type": "...", "ts": <unix ms>, ...}}`.

| Type | Payload | Typical consumer |
|---|---|---|
| `state` | `state, mode` | Widget, Waybar |
| `audio.focus` | `conversation` | All clients, Waybar |
| `audio.level` | `input: 0–1, output: 0–1` (throttled to ~20 Hz, only while a conversation is active) | Widget level meter |
| `vad.speech_start` / `vad.speech_stop` | — | Widget |
| `transcript.final` | `text` | Widget transcript |
| `turn.end` | `reason: "smart_turn" \| "silence" \| "ptt" \| "typed"` | Debugging |
| `reply.delta` | `text, spoken: bool` | Widget transcript; `spoken` flips as TTS plays it |
| `reply.done` | `text` | |
| `reply.interrupted` | `spoken_text` | Widget, debugging |
| `draft.updated` | `DraftVersion` | Widget "prompt to send" |
| `target.changed` | `Target` | Widget header |
| `brief.ready` | `summary_tokens` | Widget (subtle indicator) |
| `action.proposed` | `Action` | Widget confirmation footer |
| `action.executed` | `id, result` | Notifications |
| `action.rejected` | `id` | |
| `finalized` | `target, delivered: bool, submitted: bool, version` | Widget closes, notification |
| `conversation.ended` | `reason: "finalized" \| "cancelled" \| "idle"` | |
| `spend` | `month_usd, cap_usd` per backend | Status line |
| `error` | `code, message, recoverable: bool` | Everything |

## Planned: external platforms (M5)

For host platforms such as Motherbrain (see architecture §5.4). The host keeps its connection open and
answers callbacks from the daemon, so the connection is two-way JSON-RPC.

| Method | Direction | Purpose |
|---|---|---|
| `tools.register` | host → daemon | Declare a tool pack: `{pack, tools: [{name, description, schema, tier}]}` |
| `tools.invoke` | daemon → host | Run a tool after the action gate approves; host returns the result |
| `context.register` | host → daemon | Declare a context source: `{name, capabilities: ["load","exchanges","search"]}` |
| `context.load` / `context.exchanges` / `context.search` | daemon → host | Fetch context for briefs, recaps, `lookup` |
| `delivery.deliver` | daemon → host | Hand over a finalized draft (route kind `callback`) |

Registered packs and sources live only as long as the host's connection.

## Reserved for later

- `audio.push` / `audio.play`: clients stream mic audio in and playback out, for a remote client with the
  daemon on another machine. Not planned; the method names are reserved so they stay free.
