"""Review checks that code can decide exactly, so they never depend on the model."""

import html
import re
from collections.abc import Iterable
from html.parser import HTMLParser
from urllib.parse import urlsplit

from wp_content_agent.schemas import CheckResult, Draft
from wp_content_agent.schemas.draft import ALLOWED_TAGS, META_MAX, META_MIN

ALLOWED_ATTRS = {"a": frozenset({"href"})}
SAFE_SCHEMES = frozenset({"", "http", "https", "mailto"})

# Numbers that read as data: thousands groups, decimals, two or more digits, or percents.
_DATA_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+%?|\d+\.\d+%?|\d{2,}%?|\d%")
_PERCENT_WORD = re.compile(r"(\d)\s*(?:percent|per cent)\b", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def meta_description_check(draft: Draft) -> tuple[CheckResult, str | None]:
    length = len(draft.meta_description)
    if META_MIN <= length <= META_MAX:
        return CheckResult(name="meta_description_length", passed=True), None
    note = f"{length} characters; needs {META_MIN} to {META_MAX}"
    direction = "Lengthen" if length < META_MIN else "Shorten"
    fix = f"{direction} the meta description to {META_MIN}-{META_MAX} characters (now {length})."
    return CheckResult(name="meta_description_length", passed=False, note=note), fix


class _TagAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.problems: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in ALLOWED_TAGS:
            self.problems.add(f"<{tag}>")
            return
        allowed = ALLOWED_ATTRS.get(tag, frozenset())
        for name, value in attrs:
            if name not in allowed:
                self.problems.add(f'{name}="" on <{tag}>')
            elif name == "href" and not _safe_href(value or ""):
                self.problems.add(f"unsafe link {value!r}")

    def handle_comment(self, data: str) -> None:
        self.problems.add("HTML comments")

    def handle_decl(self, decl: str) -> None:
        self.problems.add("declarations")

    def handle_pi(self, data: str) -> None:
        self.problems.add("processing instructions")

    def unknown_decl(self, data: str) -> None:
        self.problems.add("CDATA sections")


def _safe_href(value: str) -> bool:
    try:
        scheme = urlsplit(value.strip()).scheme.lower()
    except ValueError:
        return False
    return scheme in SAFE_SCHEMES


def allowed_html_check(draft: Draft) -> tuple[CheckResult, str | None]:
    audit = _TagAudit()
    audit.feed(draft.content_html)
    audit.close()
    if not audit.problems:
        return CheckResult(name="allowed_html", passed=True), None
    found = ", ".join(sorted(audit.problems))
    fix = (
        f"Remove {found}; use only {', '.join(ALLOWED_TAGS)} tags, "
        "with an http(s) or relative href on links."
    )
    return CheckResult(name="allowed_html", passed=False, note=f"not allowed: {found}"), fix


def data_numbers(text: str) -> set[str]:
    plain = html.unescape(_TAG.sub(" ", text))
    plain = _PERCENT_WORD.sub(r"\1%", plain)
    return {match.replace(",", "") for match in _DATA_NUMBER.findall(plain)}


def unsupported_numbers(draft: Draft, allowed_sources: Iterable[str]) -> list[str]:
    """Data-like numbers in the post that appear in none of the allowed texts.

    A percentage only matches a percentage, so a word count in the brief cannot excuse
    an invented "1200%".
    """
    allowed = {number for text in allowed_sources for number in data_numbers(text)}
    written = data_numbers(
        " ".join([draft.title, draft.excerpt, draft.meta_description, draft.content_html])
    )
    return sorted(written - allowed)
