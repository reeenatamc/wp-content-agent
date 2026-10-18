import wp_content_agent
import wp_mcp


def test_packages_import() -> None:
    assert wp_content_agent.__name__ == "wp_content_agent"
    assert wp_mcp.__name__ == "wp_mcp"
