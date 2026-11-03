"""Read coding-agent sessions (Claude Code) so Passe can talk about them: list, brief, answer questions.

Sessions come from herdr (live agent panes, with their titles) and from recent Claude Code transcripts in
~/.claude/projects/*/<session-id>.jsonl. Read-only: Passe never writes to a session.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

PROJECTS = Path("~/.claude/projects").expanduser()
_SPINNER = re.compile(r"^[^\w~/(\[]+\s*")
_BRIEFS: dict = {}  # (path, size) → brief


def _herdr_panes() -> list[dict]:
    try:
        ws = json.loads(subprocess.run(["herdr", "workspace", "list"], capture_output=True, text=True,
                                       timeout=3).stdout)["result"]["workspaces"]
        panes = []
        for w in ws:
            out = subprocess.run(["herdr", "pane", "list", "--workspace", w["workspace_id"]], capture_output=True,
                                 text=True, timeout=3).stdout
            panes += json.loads(out)["result"]["panes"]
        return panes
    except Exception:
        return []


def _title_from_transcript(path: Path) -> str | None:
    title = None
    try:
        with path.open() as f:
            for line in f:
                if '"ai-title"' in line:
                    rec = json.loads(line)
                    title = rec.get("title") or rec.get("aiTitle") or rec.get("content") or title
    except OSError:
        pass
    return title if isinstance(title, str) else None


def list_sessions(limit: int = 8, max_age_hours: float = 48) -> list[dict]:
    """Most recently active first. Herdr panes give live titles; other recent transcripts fill in."""
    found: dict[str, dict] = {}
    for p in _herdr_panes():
        sess = p.get("agent_session") or {}
        if p.get("agent") != "claude" or sess.get("kind") != "id":
            continue
        path = next(PROJECTS.glob(f"*/{sess['value']}.jsonl"), None)
        if path:
            found[sess["value"]] = {"id": sess["value"], "path": path, "cwd": p.get("cwd"),
                                    "title": _SPINNER.sub("", p.get("terminal_title_stripped") or "").strip(),
                                    "pane": p.get("pane_id"), "status": p.get("agent_status")}
    cutoff = time.time() - max_age_hours * 3600
    for path in PROJECTS.glob("*/*.jsonl"):
        if path.stem not in found and path.stat().st_mtime > cutoff:
            found[path.stem] = {"id": path.stem, "path": path, "cwd": None, "title": None, "pane": None,
                                "status": None}
    sessions = sorted(found.values(), key=lambda s: s["path"].stat().st_mtime, reverse=True)
    for s in sessions:
        s["title"] = s["title"] or _title_from_transcript(s["path"]) or s["id"][:8]
        s["age_min"] = round((time.time() - s["path"].stat().st_mtime) / 60)
    # Claude Code continues a long conversation in a new transcript after compaction, while herdr keeps reporting
    # the old session id. Give the newest transcript with the same title the pane, and drop the stale entry.
    by_title = {}
    for s in sessions:
        key = (s["title"] or "").strip().lower()
        if key in by_title:
            newer = by_title[key]
            if s["pane"] and not newer["pane"]:
                newer.update(pane=s["pane"], status=s["status"], cwd=newer["cwd"] or s["cwd"])
            s["stale"] = True
        else:
            by_title[key] = s
    return [s for s in sessions if not s.get("stale")][:limit]


def find_session(which: str, exclude_ids: set[str] | None = None) -> dict | None:
    """'latest', a session id (prefix) or a word from its title."""
    sessions = [s for s in list_sessions(limit=30) if s["id"] not in (exclude_ids or set())]
    if not sessions:
        return None
    if which in ("latest", ""):
        return sessions[0]
    w = which.lower()
    return next((s for s in sessions if s["id"].startswith(w)), None) or next(
        (s for s in sessions if w in (s["title"] or "").lower()), None)


def transcript_text(path: Path, max_chars: int = 40000) -> str:
    """The conversation as plain text: user requests, assistant replies, tool use as one-liners.

    Keeps the first user request (the original goal) plus as much of the recent end as fits."""
    lines = []
    with path.open() as f:
        for raw in f:
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if rec.get("type") not in ("user", "assistant") or rec.get("isSidechain") or rec.get("isMeta"):
                continue
            content = (rec.get("message") or {}).get("content")
            role = "USER" if rec["type"] == "user" else "AGENT"
            if isinstance(content, str):
                if not content.startswith("<"):  # skip command/system wrappers
                    lines.append(f"{role}: {content.strip()}")
                continue
            for b in content or []:
                if b.get("type") == "text" and b.get("text", "").strip() and not b["text"].startswith("<"):
                    lines.append(f"{role}: {b['text'].strip()}")
                elif b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    what = inp.get("description") or inp.get("file_path") or inp.get("pattern") or ""
                    lines.append(f"AGENT [tool {b.get('name')}]: {str(what)[:120]}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    first = next((ln for ln in lines if ln.startswith("USER:")), "")[:3000]
    return first + "\n…\n" + text[-(max_chars - len(first) - 5):]


BRIEF_SYSTEM = """You summarise a coding-agent session for the user, who will talk about it by voice and then
write the next prompt for that agent. Be concrete: name files, commands, decisions and outcomes.
Reply with JSON only: {"title": "<few words>", "summary": "<two or three sentences: what the session is about and
where it stands now>", "key_points": ["<decision or fact worth remembering>", …(max 5)], "results": ["<what got
done or learned, recent first>", …(max 5)], "open_problems": ["<unresolved issue>", …], "next_steps": ["<concrete
next step a prompt could ask for>", …(max 3)]}"""

ASK_SYSTEM = """You answer the user's question about a coding-agent session, from its transcript. The answer is
spoken aloud: two or three short plain sentences, no lists, no markdown, no code. If the transcript doesn't say,
say so briefly."""


def _llm(system: str, user: str, json_mode: bool) -> str:
    import prompt_quality as pq

    cfg = pq.DRAFTER
    extra = {"reasoning_effort": cfg["reasoning_effort"]} if cfg.get("reasoning_effort") else {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    r = pq._client().chat.completions.create(
        model=cfg["model"], max_completion_tokens=3000, **extra,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


_RECENT: dict = {}  # path → (time, brief): an active session grows every second, so exact-size caching never hits
FRESH_SECS = 180


def brief(session: dict, max_age: float = FRESH_SECS) -> dict:
    """A brief under max_age seconds old is reused even if the session has moved on since (an active agent's
    transcript changes constantly and a fresh brief takes 5–20 s); call refresh_brief to update it in the background."""
    recent = _RECENT.get(str(session["path"]))
    if recent and time.time() - recent[0] < max_age:
        return recent[1]
    key = (str(session["path"]), session["path"].stat().st_size)
    if key not in _BRIEFS:
        raw = _llm(BRIEF_SYSTEM, transcript_text(session["path"]), json_mode=True)
        try:
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except ValueError:
            data = {"summary": raw.strip()[:500]}
        data.setdefault("title", session.get("title"))
        _BRIEFS[key] = data
    _RECENT[str(session["path"])] = (time.time(), _BRIEFS[key])
    return _BRIEFS[key]


def refresh_brief(session: dict) -> dict:
    return brief(session, max_age=0)


def brief_text(b: dict) -> str:
    parts = [f"Session: {b.get('title')}", f"Summary: {b.get('summary', '')}"]
    for label in ("key_points", "results", "open_problems", "next_steps"):
        if b.get(label):
            parts.append(f"{label.replace('_', ' ').capitalize()}: " + "; ".join(b[label]))
    return "\n".join(parts)


def ask(session: dict, question: str) -> str:
    return _llm(ASK_SYSTEM, f"Transcript:\n{transcript_text(session['path'], 60000)}\n\nQuestion: {question}",
                json_mode=False).strip()


if __name__ == "__main__":
    for i, s in enumerate(list_sessions(), 1):
        print(f"{i}. {s['title'][:50]:50} {s['age_min']:5}m  {s['pane'] or '':8} {s['id'][:8]}")


# ---------------------------------------------------------------------------------------------------------
# herdr control: status, screen, send. Targets are pane ids that host the agent.


def _herdr(*args: str, timeout: float = 10) -> dict:
    out = subprocess.run(["herdr", *args], capture_output=True, text=True, timeout=timeout)
    raw = out.stdout.strip() or out.stderr.strip()
    try:
        return json.loads(raw)
    except ValueError:
        return {"error": {"message": raw[:300] or f"herdr exited {out.returncode}"}}


def agent_status(pane: str) -> str:
    """idle | working | blocked | done | unknown | gone"""
    r = _herdr("agent", "get", pane)
    return (r.get("result") or {}).get("agent", {}).get("agent_status") or "gone"


def agent_screen(pane: str, lines: int = 40) -> str:
    """What's on the agent's screen right now (plain text; herdr prints it rather than JSON)."""
    out = subprocess.run(["herdr", "agent", "read", pane, "--source", "recent-unwrapped", "--lines", str(lines)],
                         capture_output=True, text=True, timeout=10).stdout
    rows = [r.rstrip() for r in out.splitlines()]
    return "\n".join(r for r in rows if r.strip())[-4000:]


def send_prompt(pane: str, text: str) -> dict:
    """Type the prompt into the agent and submit it (herdr honours bracketed paste, so multi-line is one message).
    Returns {"ok": True} or {"ok": False, "error": "..."}; herdr refuses agents waiting at an approval dialog."""
    r = _herdr("agent", "prompt", pane, text, timeout=60)
    if "error" in r:
        err = r["error"]
        return {"ok": False, "error": err.get("code") or err.get("message") or str(err)}
    return {"ok": True}
