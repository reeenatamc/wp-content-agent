"""Errors raised when WordPress cannot fulfil a request."""

from typing import Any, Self


class WordPressError(Exception):
    def __init__(
        self, message: str, *, status_code: int | None = None, code: str = "error"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.data: dict[str, Any] = {}

    @classmethod
    def from_payload(cls, status_code: int, payload: Any) -> Self:
        """Build from a WP error body `{"code", "message", "data"}` when present."""
        if isinstance(payload, dict) and "message" in payload:
            err = cls(
                str(payload["message"]),
                status_code=status_code,
                code=str(payload.get("code", "error")),
            )
            data = payload.get("data")
            if isinstance(data, dict):
                err.data = data
            return err
        return cls(f"WordPress returned HTTP {status_code}", status_code=status_code)

    def __str__(self) -> str:
        status = f"HTTP {self.status_code} " if self.status_code is not None else ""
        return f"{status}{self.code}: {self.message}"
