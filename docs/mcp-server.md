# The wp_mcp server

`wp_mcp` is an MCP server that speaks stdio and wraps the WordPress REST API. The agent
starts it as a subprocess and talks to it through `langchain-mcp-adapters`, but nothing in
it knows about the agent, so any MCP host can use it.

## Running it

```bash
WP_BASE_URL=http://localhost:8080 WP_USERNAME=admin WP_APP_PASSWORD=... uv run wp-mcp
```

It authenticates with a WordPress application password (Users, Profile, Application
Passwords). `scripts/wp-up.sh` creates one for the local Docker site. WordPress only offers
application passwords over HTTPS, or when `WP_ENVIRONMENT_TYPE` is `local`, which is how
the Docker site is configured.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `list_posts` | `search` (up to 100 chars), `per_page` (1 to 20, default 10) | Published and draft posts: id, title, link, status |
| `get_post` | `post_id` | One post with its content, status and edit URL |
| `list_categories` | | Every category with its id |
| `list_tags` | `search` | Tags with their ids |
| `create_tag` | `name` (1 to 60 chars) | The new tag, or the existing one with that name |
| `create_draft_post` | `title`, `content_html`, `slug`, `excerpt`, `categories`, `tags`, `meta_description` | The post id, status and edit URL, and `reused` |

There is no tool to publish, update or delete. `create_draft_post` always sends
`status: draft`, and the agent checks that WordPress returned a draft before it reports
success. Categories and tags are passed as ids; `create_tag` exists so a caller can turn
new tag names into ids first.

`create_draft_post` is safe to repeat. When a `slug` is given, the server first looks for a
draft with that slug and exactly the same title. If it finds one, it returns that post with
`reused: true` and creates nothing; otherwise it creates the post. This covers a create that
timed out after WordPress made the post, and a retry after a crash. Without a `slug` there
is nothing to look up, so every call creates a post.

`meta_description` is stored in post meta under `WP_META_DESCRIPTION_KEY`
(`meta_description` by default). WordPress ignores meta keys that are not registered for
the REST API, so the server checks the response and logs a warning when the value was not
saved. `docker/mu-plugins/wpagent-meta.php` registers `meta_description` on the local
site. On a site with an SEO plugin, point the setting at that plugin's key instead; the
mu-plugin only covers the default key, so a different key must be registered by the plugin
or by your own `register_post_meta` call.

## Errors, retries and timeouts

WordPress errors become MCP tool errors with a one-line message, so the model or the
caller sees "rest_forbidden: Sorry, you are not allowed to create posts" rather than a
stack trace.

Every request has a timeout (`WP_TIMEOUT_S`, 15 s). GET requests are retried on
connection errors, timeouts, 429 and 5xx, with exponential backoff (`WP_BACKOFF_S`, up to
`WP_MAX_RETRIES` times) and a `Retry-After` header honored up to `WP_MAX_RETRY_AFTER_S`.
POST requests, which create posts and tags, are retried only when the request was never
sent or got a 429. A POST that timed out may already have created the post, so the server
does not resend it. Before creating a draft it looks for one with the same slug, title and content
(a GET, retried like the others), so repeating the call returns the existing draft instead
of a second one. `create_tag` is already idempotent: WordPress answers `term_exists` and the
server returns that tag.

`list_categories` and `list_tags` follow the `X-WP-TotalPages` header up to
`WP_MAX_TERM_PAGES` pages of 100. A site with more terms than that gets a truncated list
and a warning in the log.

## Settings

| Variable | Default | |
|---|---|---|
| `WP_BASE_URL` | `http://localhost:8080` | Site URL, without `/wp-json` |
| `WP_USERNAME` | required | |
| `WP_APP_PASSWORD` | required | Application password, not the login password |
| `WP_TIMEOUT_S` | `15` | Per request |
| `WP_MAX_RETRIES` | `3` | |
| `WP_BACKOFF_S` | `0.5` | Doubles on each retry |
| `WP_MAX_RETRY_AFTER_S` | `30` | Cap on a server's `Retry-After` |
| `WP_MAX_TERM_PAGES` | `10` | Pages of categories or tags to read |
| `WP_META_DESCRIPTION_KEY` | `meta_description` | Post meta key for the description |

## Tests

`tests/unit/wp_mcp` covers the client against a mocked transport and the server through an
in-memory MCP session. `tests/integration/test_wordpress.py` runs the real server as a
subprocess against the Docker WordPress:

```bash
scripts/wp-up.sh
WPAGENT_INTEGRATION=1 uv run pytest tests/integration
```
