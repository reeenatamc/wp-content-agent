"""Load a content brief from a YAML or JSON file."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from wp_content_agent.errors import BriefError
from wp_content_agent.schemas import Brief


def load_brief(path: Path) -> Brief:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BriefError(f"cannot read brief {path}: {exc.strerror}") from exc
    try:
        # JSON is valid YAML, so one parser covers both formats.
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise BriefError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise BriefError(f"{path} must contain a mapping of brief fields")
    try:
        return Brief.model_validate(data)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
        )
        raise BriefError(f"{path} is not a valid brief: {errors}") from exc
