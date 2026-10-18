"""Integration tests run against the Docker WordPress from scripts/wp-up.sh.

They are skipped unless WPAGENT_INTEGRATION=1. WP_* settings come from the
environment or `.env`, which scripts/wp-up.sh writes.
"""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

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


@dataclass
class Cleanup:
    """Ids registered here are deleted after the test, even when it fails."""

    posts: list[int] = field(default_factory=list)
    tags: list[int] = field(default_factory=list)


@pytest.fixture
async def cleanup(wordpress: WordPressClient) -> AsyncIterator[Cleanup]:
    created = Cleanup()
    yield created
    # The client has no delete methods on purpose (the agent never deletes), so the
    # tests call the REST API directly here.
    paths = [f"/posts/{i}" for i in created.posts] + [f"/tags/{i}" for i in created.tags]
    for path in paths:
        response = await wordpress._http.delete(path, params={"force": "true"})
        assert response.status_code in {200, 404}, f"cleanup of {path}: {response.text}"
