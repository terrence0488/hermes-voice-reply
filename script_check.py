"""Decide whether text is fit to be spoken aloud."""
import re

MAX_WORDS = 110

_CUE = re.compile(r"\[[^\]\n]{1,40}\]")
_CHECKS = (
    ("a code block", re.compile(r"```|^(?: {4}|\t)\S", re.M)),
    ("a table", re.compile(r"^\s*\|.*\|\s*$|^\s*\|?\s*:?-{3,}:?\s*\|", re.M)),
    ("JSON", re.compile(r"^\s*[\[{]\s*[\"{\[]|\"[\w-]+\"\s*:\s*[\"\[{\d]", re.M)),
    ("HTML", re.compile(r"</?(?:html|head|body|div|span|table|tr|td|th|ul|ol|li|p|a|h[1-6]|script|style|br)\b[^>]*>", re.I)),
    ("a URL", re.compile(r"\b(?:https?://|www\.)\S+", re.I)),
    ("a file path", re.compile(r"(?:^|\s)(?:~?/[\w.-]+){2,}")),
    ("markdown", re.compile(r"^\s*#{1,6}\s|\*\*\S|^\s*(?:[-*+]|\d+\.)\s+\S.*\n\s*(?:[-*+]|\d+\.)\s+\S", re.M)),
)


def problem_with(text):
    """Return what makes this unfit to speak, or None."""
    if not isinstance(text, str) or not text.strip():
        return "no words"
    for label, pattern in _CHECKS:
        if pattern.search(text):
            return label
    if len(_CUE.sub(" ", text).split()) > MAX_WORDS:
        return f"more than {MAX_WORDS} words"
    return None
