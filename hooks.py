"""Hook callbacks. The gateway knows a message was a voice note; the model only sees a quoted transcript."""
import json
import logging
import os
import re
import threading
import time

from .script_check import problem_with

log = logging.getLogger("voice_reply")

PENDING_TTL_SECONDS = 600
TURN_TTL_SECONDS = 1800
MAX_TRACKED = 200
SPEECH_TOOLS = {"fish_speak", "text_to_speech"}
FALLBACK_SIMPLE_MAX_CHARS = 400
FALLBACK_DETAIL_LINE = "I've put the details in the text. Take a look."
FAILED_NOTE = "(Voice didn't go through this time.)"

INSTRUCTION = """[The user spoke this message as a voice note. Reply with voice by calling `fish_speak`, and use the `media_tag` it returns. Never write a `MEDIA:` path yourself, and never write the script out as text instead of calling the tool. Don't mention this note.

Write a script for talking, not a written answer read aloud: sound like a person talking to a friend, with contractions and short sentences, in the user's language, under about 75 words. Answer only what was asked. Use bracket cues where they fit the feeling, such as [excited], [calm], [chuckling], [sighing], [break]: emotion cues at the start of a sentence, usually one or two per reply. Never put tables, code, JSON, HTML, URLs, file paths, long lists or markdown in the script.

Simple answer: everything is in the voice, plus one short caption line and the `media_tag` on its own line.
Complex answer, with a table, code, data, steps or links: put that detail in the text. The voice gives the gist and says the detail is in the text.

Call `fish_speak` once.]"""

_MEDIA = re.compile(r"MEDIA:([^\s\"'\\]+)")
_MEDIA_LINE = re.compile(r"^[ \t]*(?:\[\[audio_as_voice\]\][ \t]*)?MEDIA:(\S+)[ \t]*$", re.M)
_lock = threading.Lock()
_pending = {}  # (platform, user id) -> [times voice notes arrived]
_turns = {}  # session or task id -> turn record


def _name(value):
    return str(getattr(value, "value", value) or "").lower()


def _is_voice(event):
    kind = getattr(event, "message_type", None)
    return str(getattr(kind, "name", kind) or "").upper() == "VOICE"


def _prune(now):
    for key in [k for k, times in _pending.items() if not [t for t in times if now - t < PENDING_TTL_SECONDS]]:
        del _pending[key]
    for key in [k for k, turn in _turns.items() if now - turn["at"] >= TURN_TTL_SECONDS]:
        del _turns[key]


def on_pre_gateway_dispatch(event=None, **kwargs):
    """Runs before authorization, so a stranger can only mark their own ID, and the table is capped."""
    try:
        if event is None or getattr(event, "internal", False) or not _is_voice(event):
            return None
        source = getattr(event, "source", None)
        user = getattr(source, "user_id", None)
        if not user or getattr(source, "chat_type", "dm") != "dm":
            return None
        now = time.monotonic()
        with _lock:
            _prune(now)
            if len(_pending) < MAX_TRACKED or (_name(source.platform), str(user)) in _pending:
                _pending.setdefault((_name(source.platform), str(user)), []).append(now)
    except Exception:
        log.exception("could not note a voice message")
    return None


def on_pre_llm_call(session_id="", task_id="", platform="", sender_id="", **kwargs):
    try:
        now = time.monotonic()
        key = (_name(platform), str(sender_id))
        with _lock:
            _prune(now)
            times = [t for t in _pending.get(key, []) if now - t < PENDING_TTL_SECONDS]
            if not times:
                return None
            times.pop(0)
            if times:
                _pending[key] = times
            else:
                _pending.pop(key, None)
            turn = {"at": now, "audio": []}
            for turn_key in {session_id, task_id} - {"", None}:
                _turns[turn_key] = turn
        return {"context": INSTRUCTION}
    except Exception:
        log.exception("could not start a voice turn")
        return None


def _turn(kwargs, task_id=""):
    with _lock:
        return _turns.get(kwargs.get("session_id")) or _turns.get(task_id or kwargs.get("task_id"))


def _speech_calls(tool_name, args):
    """Speech tool arguments in a call, looking inside the `tool_call` wrapper used for deferred tools."""
    if tool_name in SPEECH_TOOLS:
        return [args] if isinstance(args, dict) else []
    if tool_name == "tool_call" and isinstance(args, dict):
        calls = args.get("calls")
        return [c.get("arguments") for c in calls if isinstance(c, dict) and c.get("name") in SPEECH_TOOLS
                and isinstance(c.get("arguments"), dict)] if isinstance(calls, list) else []
    return []


def on_pre_tool_call(tool_name="", args=None, task_id="", **kwargs):
    try:
        if not _turn(kwargs, task_id):
            return None
        for call in _speech_calls(tool_name, args):
            problem = problem_with(call.get("text"))
            if problem:
                return {"action": "block", "message": f"This script has {problem}, which doesn't work spoken aloud. "
                                                      "Rewrite it as short, natural speech and put the detail in your text reply."}
    except Exception:
        log.exception("could not check a voice script")
    return None


def on_post_tool_call(tool_name="", args=None, result="", task_id="", **kwargs):
    try:
        turn = _turn(kwargs, task_id)
        if not turn or not _speech_calls(tool_name, args):
            return None
        with _lock:
            turn["audio"].extend(_MEDIA.findall(result if isinstance(result, str) else json.dumps(result, default=str)))
    except Exception:
        log.exception("could not record voice audio")
    return None


def _synthesize(text):
    """Make speech with the Fish Audio tool directly. Returns its media tag, or None."""
    from tools.registry import registry
    outcome = json.loads(registry.dispatch("fish_speak", {"text": text}))
    tag = outcome.get("media_tag") if isinstance(outcome, dict) else None
    return tag if isinstance(tag, str) and _MEDIA.search(tag) else None


def _spoken_fallback(reply):
    """What to say when the model made no audio: the reply itself if it can be spoken, else a pointer to it."""
    bare = reply.strip()
    if bare and len(bare) <= FALLBACK_SIMPLE_MAX_CHARS and not problem_with(bare):
        return bare
    return FALLBACK_DETAIL_LINE


def on_transform_llm_output(response_text="", session_id="", task_id="", **kwargs):
    try:
        with _lock:
            turn = _turns.pop(session_id, None) or _turns.pop(task_id, None)
            if turn:
                for key in [k for k, value in _turns.items() if value is turn]:
                    del _turns[key]
        if not turn:
            return None
        if any(os.path.isfile(path) for path in turn["audio"]):
            return None  # The Fish Audio plugin's own hook adds a tag the model left out.
        cleaned = _MEDIA_LINE.sub(lambda m: m.group(0) if os.path.isfile(m.group(1)) else "", response_text or "")
        if any(os.path.isfile(path) for path in _MEDIA.findall(cleaned)):
            return None if cleaned == response_text else cleaned
        cleaned = cleaned.replace("[[audio_as_voice]]", "").strip()
        log.warning("voice turn produced no audio; using the fallback")
        tag = None
        try:
            tag = _synthesize(_spoken_fallback(cleaned))
        except Exception:
            log.exception("fallback speech failed")
        if tag:
            return f"{cleaned}\n{tag}" if cleaned else tag
        return f"{cleaned}\n\n{FAILED_NOTE}" if cleaned else FAILED_NOTE
    except Exception:
        log.exception("could not finish a voice turn")
        return None
