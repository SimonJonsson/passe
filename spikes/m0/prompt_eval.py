"""Scorecard for Passe sessions, from event logs: prompt quality, conversation quality, tool-calling accuracy.

    uv run prompt_eval.py latest | all | FILE…    [--votes 2] [--quiet]

prompt        the final draft against the rubric in prompt_quality.RUBRIC (LLM judge, averaged over --votes)
conversation  Passe's side of the conversation against prompt_quality.CONVO_RUBRIC (LLM judge)
tools         rule checks on the log: every request that should trigger a tool did, and nothing fired wrongly
Voice and audio quality are out of scope; only text is logged.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import prompt_quality as pq
import spike

# user request (regex on what the user said) → tools that should be called in that turn
EXPECTED = [
    ("copy request", re.compile(r"\b(copy|clipboard)\b", re.I), {"copy_to_clipboard"}),
    ("revert request", re.compile(r"\b(go back|undo|previous version)\b", re.I), {"revert_draft"}),
    ("voice request", re.compile(r"^\W*((ok(ay)?|yeah|so|please)\W+)*(switch|change|use|try|let me hear|can i hear)"
                                 r"\b.{0,25}\bvoices?\b", re.I),
     {"set_voice", "preview_voices"}),
]


def sessions(log: Path) -> list[dict]:
    """Split a log into Passe sessions with their turns, tool calls and drafts."""
    out, cur = [], None

    def new():
        return {"log": log.name, "turns": [], "draft": "", "events": []}

    cur = new()
    for e in (json.loads(x) for x in log.read_text().splitlines() if x.strip()):
        k = e["kind"]
        if k in ("session_start", "session_end"):
            if cur["turns"]:
                out.append(cur)
            ctx, cur = cur.get("context", ""), new()
            cur["context"] = ctx  # the agent session stays attached across Passe sessions
            continue
        cur["events"].append(e)
        if k == "agent_brief":
            cur["context"] = f"Coding-agent session brief:\n{e['brief']}\n\nPasse log path: {log}"
        if k == "user_turn" and e.get("text"):
            cur["turns"].append(("USER", e["text"]))
        elif k == "assistant_turn" and (e.get("full") or "").strip():
            cur["turns"].append(("PASSE", spike.conversational(e["full"])))
        elif k == "tool_call" and e["name"] == "update_draft" and e["args"].get("text"):
            cur["draft"] = e["args"]["text"]  # older logs: the talker wrote the draft itself
        elif k == "draft" and e.get("text"):
            cur["draft"] = e["text"]
            cur["turns"].append(("NOTE", f"[draft v{e.get('version')} written]"))
    if cur["turns"]:
        out.append(cur)
    return out


def tool_accuracy(s: dict) -> dict:
    """Rule checks over one session's events."""
    hits, misses, errors = 0, [], []
    evs = s["events"]
    caught = []
    for i, e in enumerate(evs):
        if e["kind"] == "force_ignored":
            caught.append(f"tried to skip the copy check without consent (after: {e.get('last_user')!r})")
        if e["kind"] == "guard_blocked":
            caught.append(f"{e['tool']} blocked: {e['reason']} (after: {e.get('last_user')!r})")
        if e["kind"] != "user_turn" or not e.get("text"):
            continue
        said = e["text"]
        nxt = next((j for j in range(i + 1, len(evs)) if evs[j]["kind"] == "user_turn" and evs[j].get("text")),
                   len(evs))
        called = [x for x in evs[i + 1:nxt] if x["kind"] == "tool_call"]
        names = {x["name"] for x in called}
        # already copied and nothing new drafted since? then not copying again is right
        last_copy = max((j for j in range(i) if evs[j]["kind"] == "tool_result"
                         and evs[j].get("name") == "copy_to_clipboard"
                         and str(evs[j].get("summary", "")).startswith("copied")), default=None)
        already_copied = last_copy is not None and not any(evs[j]["kind"] == "draft" for j in range(last_copy, i))
        for label, rx, tools in EXPECTED:
            if rx.search(said):
                if names & tools or (label == "copy request" and already_copied):
                    hits += 1
                else:
                    misses.append(f"{label} not acted on: {said!r}")
        blocked = {x["tool"] for x in evs[i + 1:nxt] if x["kind"] == "guard_blocked"}
        # claimed the draft was updated before the writer had produced it
        claims = [x for x in evs[i + 1:nxt] if x["kind"] == "assistant_turn" and re.search(
            r"\b(i'?ve|i have|i just)\s+(added|updated|changed|put|included|rewritten)|\bdraft (now|already) "
            r"(has|includes|covers|mentions)|\b(now|already) (has|includes|mentions) the\b",
            spike.conversational(x.get("full") or "").replace("’", "'"), re.I)]
        if "update_draft" in names and claims:
            landed = next((x["t"] for x in evs[i + 1:] if x["kind"] == "draft"), None)
            if landed is None or landed > claims[0]["t"]:
                errors.append(f"said the draft was updated before it was (after: {said!r})")
        if spike.CONTROL_UTTERANCE.search(said) and "update_draft" in names - blocked:
            errors.append(f"drafted from an instruction to Passe: {said!r}")
        if "copy_to_clipboard" in names - blocked and not spike.COPY_REQUEST.search(said):
            errors.append(f"copied without being asked (after: {said!r})")
    expected = hits + len(misses)
    penalty = 2 * len(errors) + 0.5 * len(caught)  # caught = the talker erred but a guard prevented harm
    return {"hits": hits, "expected": expected, "misses": misses, "errors": errors, "caught": caught,
            "score": round((10 * hits / expected if expected else 10) - penalty, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=["latest"])
    ap.add_argument("--votes", type=int, default=2)
    ap.add_argument("--quiet", action="store_true", help="scores only")
    args = ap.parse_args()
    files = sorted(spike.LOG_DIR.glob("spike-*.jsonl"))
    paths = files if args.logs == ["all"] else [files[-1]] if args.logs == ["latest"] else [Path(p) for p in args.logs]
    totals = {"prompt": [], "conversation": [], "tools": []}
    for path in paths:
        for s in sessions(path):
            convo = "\n".join(text if who == "NOTE" else f"{who}: {text}" for who, text in s["turns"])
            print(f"\n=== {s['log']} · {sum(w == 'USER' for w, _ in s['turns'])} user turns")
            if s["draft"]:
                votes = [pq.judge(convo, s["draft"], context=s.get("context", "")) for _ in range(args.votes)]
                ps = {k: statistics.mean(v["scores"][k] for v in votes) for k in pq.RUBRIC}
                p_overall = round(statistics.mean(ps.values()), 1)
                totals["prompt"].append(p_overall)
                print(f"prompt        {p_overall:4.1f}/10 · " + " ".join(f"{k} {v:.1f}" for k, v in ps.items()))
                if not args.quiet:
                    for label in ("invented", "stale"):
                        items = sorted({x for v in votes for x in v[label]})
                        if items:
                            print(f"   {label}: " + "; ".join(items[:4]))
            c = pq.judge_conversation(convo, s.get("context", ""))
            totals["conversation"].append(c["overall"])
            print(f"conversation  {c['overall']:4.1f}/10 · " + " ".join(f"{k} {v:.1f}" for k, v in c["scores"].items()))
            if not args.quiet:
                for issue in c["issues"][:3]:
                    print(f"   - {issue}")
            t = tool_accuracy(s)
            totals["tools"].append(max(0.0, t["score"]))
            print(f"tools         {max(0.0, t['score']):4.1f}/10 · {t['hits']}/{t['expected']} requests acted on, "
                  f"{len(t['errors'])} error(s), {len(t['caught'])} caught by guards")
            if not args.quiet:
                for m in t["misses"] + t["errors"]:
                    print(f"   - {m}")
                for m in t["caught"]:
                    print(f"   · caught: {m}")
    print("\nMean: " + " · ".join(f"{k} {statistics.mean(v):.1f}" for k, v in totals.items() if v))


if __name__ == "__main__":
    main()
