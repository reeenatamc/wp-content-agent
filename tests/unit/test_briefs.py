from pathlib import Path

import pytest

from wp_content_agent.briefs import load_brief
from wp_content_agent.errors import BriefError

EXAMPLES = Path(__file__).parents[2] / "examples" / "briefs"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_example_briefs_are_valid(path: Path) -> None:
    brief = load_brief(path)

    assert brief.topic


def test_json_brief_loads(tmp_path: Path) -> None:
    path = tmp_path / "brief.json"
    path.write_text('{"topic": "Caching in Django", "audience": "Backend developers"}')

    assert load_brief(path).audience == "Backend developers"


def test_missing_file_raises_brief_error(tmp_path: Path) -> None:
    with pytest.raises(BriefError, match="cannot read brief"):
        load_brief(tmp_path / "missing.yaml")


def test_non_mapping_raises_brief_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")

    with pytest.raises(BriefError, match="must contain a mapping"):
        load_brief(path)


def test_invalid_yaml_raises_brief_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("topic: [unclosed\n")

    with pytest.raises(BriefError, match="not valid YAML"):
        load_brief(path)


def test_schema_errors_name_the_field(tmp_path: Path) -> None:
    path = tmp_path / "short.yaml"
    path.write_text("topic: Caching in Django\naudience: x\nunknown: 1\n")

    with pytest.raises(BriefError, match=r"audience: .*unknown: "):
        load_brief(path)


def test_non_utf8_file_raises_brief_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes("topic: Caché\n".encode("latin-1"))

    with pytest.raises(BriefError, match="cannot read brief"):
        load_brief(path)
