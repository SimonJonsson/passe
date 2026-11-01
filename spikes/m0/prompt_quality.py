"""Prompt quality: a background drafter that writes the prompt, and a rubric judge that scores it.

Both read the conversation, not just the draft, so the judge can tell invented requirements from stated ones.
Models are Baseten Model APIs (OpenAI-compatible); see DRAFTER / JUDGE for the defaults.
"""

from __future__ import annotations

import json

# Writer chosen by an offline comparison on replayed sessions (judged faithful 9.5/10 vs 3.8 for gpt-oss medium,
# 4 s per draft): see README "Prompt quality".
DRAFTER = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "reasoning_effort": None}
JUDGE = {"model": "openai/gpt-oss-120b", "reasoning_effort": "medium"}
QUICK_JUDGE = {"model": "openai/gpt-oss-120b", "reasoning_effort": "low"}

RUBRIC = {
    "faithful": "Contains only what the user said or confirmed. Every requirement, number, tool, UI element or "
                "constraint the user never asked for is invented; each one costs points.",
    "goal": "States the goal the user actually has, about the right subject, in a sentence or two.",
    "context": "A coding agent reading it cold would know what system, files or situation this is about.",
    "constraints": "Requirements are decisive (no 'e.g.' for musts), relevant, and nothing the user contradicted "
                   "or corrected is left in.",
    "verifiable": "'Done' can be checked by a person or a test, and is proportionate to the ask.",
    "open_questions": "Things that matter but weren't settled are listed as open questions instead of guessed.",
}

SELF = ("Passe is the voice assistant the user is talking to: a spike in ~/Repos/passe/spikes/m0 (spike.py) built on "
        "Pipecat, with Whisper speech recognition, a Baseten LLM, and Orpheus/Pocket/Kokoro speech synthesis; it "
        "writes event logs to ~/.local/state/passe/ and has --analyze. When the user describes how Passe itself "
        "behaves (its voice, turn-taking, prompts…), the subject of the prompt is Passe, not 'the coding agent'.")

DRAFT_SYSTEM = f"""You write the prompt that the user is talking through with Passe, for a coding agent (Claude Code,
Codex…) to execute. You get the conversation so far and the current draft; return the updated draft.

{SELF}

Rules:
- Faithful: include only what the user said or confirmed. Never add requirements, numbers, tools, UI, dashboards,
  ratings or constraints they didn't ask for. If something matters but wasn't settled, put it under Open questions.
- When the user corrected or contradicted something, rewrite it; delete anything that no longer holds.
- Lines where the user instructs Passe itself (copy it, as is, go back, undo, change voice, summarise it, read it
  back, pause, start/end session) are not part of the task. Ignore them completely.
- Don't fill sections for the sake of it. If the user hasn't stated concrete requirements, keep Requirements to
  what they did say (or omit it) and turn candidate ideas into Open questions. Out of scope only if the user said
  so. A short faithful draft beats a complete-looking invented one.
- Use the user's own framing of the subject and the problem; include concrete details they gave (examples,
  symptoms, files, commands, names).
- Sections, in this order, omitting empty ones: Goal, Context, Requirements, Out of scope, Done when,
  Open questions. Markdown is fine. Concise: no filler, no restating the same point.
- "Done when" must be checkable and proportionate: what a person or test would verify, nothing grander.

Reply with JSON only: {{"draft": "<full markdown draft>", "changed": "<one short sentence: what changed>",
"open_questions": ["<question>", …]}}"""

JUDGE_SYSTEM = f"""You review a prompt that a voice assistant drafted with a user, for a coding agent to execute.
You get the conversation and the final draft. Score each criterion 0-10, strictly:
{json.dumps(RUBRIC, indent=1)}

{SELF}
Facts about Passe's own stack and files (listed above) are legitimate context, not invention.

Reply with JSON only: {{"scores": {{"<criterion>": <0-10>, …}}, "invented": ["<requirement the user never asked
for>", …], "stale": ["<thing the user corrected that is still there>", …], "gaps": ["<missing piece that matters,
as a short question to ask the user>", …]}}"""


def _client():
    from openai import OpenAI

    import spike

    return OpenAI(api_key=spike.load_api_key(), base_url="https://inference.baseten.co/v1")


def _json_call(system: str, user: str, cfg: dict, max_tokens: int = 6000) -> dict:
    extra = {"reasoning_effort": cfg["reasoning_effort"]} if cfg.get("reasoning_effort") else {}
    r = _client().chat.completions.create(
        model=cfg["model"], max_completion_tokens=max_tokens, **extra,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    raw = r.choices[0].message.content or "{}"
    try:
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except ValueError:
        return {}


CONVO_RUBRIC = {
    "one_question": "Each reply asks at most one question.",
    "diagnostic": "Questions dig into the problem (an example, where it happens, how to verify), never menus of "
                  "options ('friendly, calm or upbeat?').",
    "concise": "Replies are short enough to listen to (about two sentences) unless the user asked for more.",
    "responsive": "Each reply addresses what the user just said; corrections are acknowledged and taken on board.",
    "no_invention": "Passe doesn't push requirements, features or tools the user never raised.",
    "flow": "Natural spoken conversation: no repetition, no reading the draft or lists aloud, no needless "
            "wrapping up.",
}

CONVO_SYSTEM = f"""You review a spoken conversation in which a voice assistant (PASSE) helps a USER shape a prompt
for a coding agent. Speech recognition errors in USER lines are not PASSE's fault. Score each criterion 0-10, strictly:
{json.dumps(CONVO_RUBRIC, indent=1)}
Reply with JSON only: {{"scores": {{"<criterion>": <0-10>, …}}, "issues": ["<short concrete example of a problem,
quoting PASSE>", …]}}"""


def judge_conversation(conversation: str) -> dict:
    out = _json_call(CONVO_SYSTEM, conversation, JUDGE)
    scores = {k: float(out.get("scores", {}).get(k, 0)) for k in CONVO_RUBRIC}
    return {"scores": scores, "overall": round(sum(scores.values()) / len(scores), 1), "issues": out.get("issues") or []}


def conversation_text(messages: list[dict]) -> str:
    """User and assistant turns from an LLM context (tool calls and system messages left out)."""
    lines = []
    for m in messages:
        if m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
            continue
        if m["content"].strip():
            lines.append(f"{'USER' if m['role'] == 'user' else 'PASSE'}: {m['content'].strip()}")
    return "\n".join(lines)


def write_draft(conversation: str, current: str, note: str = "") -> dict:
    user = (f"Conversation so far:\n{conversation}\n\nCurrent draft:\n{current or '(none yet)'}"
            + (f"\n\nWhat Passe noted as the latest change: {note}" if note else ""))
    out = _json_call(DRAFT_SYSTEM, user, DRAFTER)
    return {"draft": (out.get("draft") or current or "").strip(), "changed": out.get("changed", ""),
            "open_questions": out.get("open_questions") or []}


def judge(conversation: str, draft: str, quick: bool = False) -> dict:
    out = _json_call(JUDGE_SYSTEM, f"Conversation:\n{conversation}\n\nFinal draft:\n{draft}",
                     QUICK_JUDGE if quick else JUDGE)
    scores = {k: float(out.get("scores", {}).get(k, 0)) for k in RUBRIC}
    return {"scores": scores, "overall": round(sum(scores.values()) / len(scores), 1),
            "invented": out.get("invented") or [], "stale": out.get("stale") or [], "gaps": out.get("gaps") or []}
