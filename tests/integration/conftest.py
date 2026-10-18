"""Integration tests run against the Docker WordPress from scripts/wp-up.sh.

They are skipped unless WPAGENT_INTEGRATION=1. WP_* settings come from the
environment or `.env`, which scripts/wp-up.sh writes.
"""

import os
from collections.abc import AsyncIterator

import pytest

from wp_mcp.client import WordPressClient
from wp_mcp.settings import WPSettings


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    enabled = os.environ.get("WPAGENT_INTEGRATION") == "1"
    skip = pytest.mark.skip(reason="set WPAGENT_INTEGRATION=1 and run scripts/wp-up.sh")
    for item in items:
        if "tests/integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            if not enabled:
                item.add_marker(skip)


@pytest.fixture
async def wordpress() -> AsyncIterator[WordPressClient]:
    client = WordPressClient.from_settings(WPSettings())  # type: ignore[call-arg]
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def cleanup(wordpress: WordPressClient) -> AsyncIterator[list[int]]:
    """Post ids added here are deleted after the test."""
    post_ids: list[int] = []
    yield post_ids
    for post_id in post_ids:
        await wordpress._http.delete(f"/posts/{post_id}", params={"force": "true"})
