"""Review checks that code can decide exactly, so they never depend on the model."""

import re
from collections.abc import Iterable
from html.parser import HTMLParser

from wp_content_agent.schemas import CheckResult, Draft

META_MIN, META_MAX = 140, 160
ALLOWED_TAGS = frozenset(
    {"p", "h2", "h3", "ul", "ol", "li", "a", "strong", "em", "blockquote", "code"}
)
ALLOWED_ATTRS = {"a": frozenset({"href"})}

# Numbers that read as data: two or more digits, decimals, or anything with a percent sign.
_DATA_NUMBER = re.compile(r"\d+(?:[.,]\d+)+%?|\d{2,}%?|\d%")
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
        for name, _ in attrs:
            if name not in allowed:
                self.problems.add(f'{name}="" on <{tag}>')

    def handle_comment(self, data: str) -> None:
        self.problems.add("HTML comments")


def allowed_html_check(draft: Draft) -> tuple[CheckResult, str | None]:
    audit = _TagAudit()
    audit.feed(draft.content_html)
    audit.close()
    if not audit.problems:
        return CheckResult(name="allowed_html", passed=True), None
    found = ", ".join(sorted(audit.problems))
    fix = f"Remove {found}; use only {', '.join(sorted(ALLOWED_TAGS))}, and href on links."
    return CheckResult(name="allowed_html", passed=False, note=f"not allowed: {found}"), fix


def data_numbers(text: str) -> set[str]:
    return {match.replace(",", "") for match in _DATA_NUMBER.findall(_TAG.sub(" ", text))}


def unsupported_numbers(draft: Draft, allowed_sources: Iterable[str]) -> list[str]:
    """Data-like numbers in the post that appear in none of the allowed texts."""
    allowed = {number.rstrip("%") for text in allowed_sources for number in data_numbers(text)}
    written = data_numbers(" ".join([draft.title, draft.excerpt, draft.content_html]))
    return sorted(number for number in written if number.rstrip("%") not in allowed)
