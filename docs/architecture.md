# Architecture

This page is the detail behind the README: what each part owns, how a run moves through
the graph, and what happens when something fails.

## Layout

```text
src/wp_content_agent/
  schemas/      Pydantic models: Brief, Outline, ResearchNotes, Draft, Review, RunSnapshot
  graph/        AgentState, the runtime context and build_graph
  nodes/        one module per node, plus the review checks and the Gutenberg converter
  llm/          ModelClient (typed output, usage, prompt caching) and the provider factory
  tools/        fetch_url and the MCP gateway the nodes use for WordPress
  runner.py     AgentRunner: start, resume, get, and failure handling
  api/          FastAPI app and RunService (background runs, limits, shutdown)
  cli/          the wpagent command
src/wp_mcp/     the MCP server and its WordPress REST client
evals/          eval briefs and run.py
```

The nodes do not import the CLI, the API or the MCP client. They get everything through
`AgentContext` (settings, the writer and reviewer clients, a WordPress gateway and an HTTP
client), which LangGraph passes as the runtime context. That is what lets the unit tests
run every node with fakes.

## State

`AgentState` is a `TypedDict`. Each node returns only the keys it changes:

| Key | Written by |
|---|---|
| `brief` | the caller |
| `outline` | `plan` |
| `context` | `gather_context` (sources, related posts, categories, tags) |
| `draft`, `revision_count` | `write` |
| `review` | `review` |
| `approval_note`, `human_feedback`, `human_revision_count` | `human_approval` (`write` clears `human_feedback`) |
| `wp_post_id`, `edit_url` | `publish_draft` |
| `status`, `error` | any node |
| `usage`, `warnings` | appended by any node (`operator.add` reducers) |

`usage` is the cost log: every node appends one `NodeUsage` with the model, input and
output tokens, cache reads and writes, and the estimated cost from `MODEL_PRICES`. The CLI
prints it as each node finishes, and `RunSnapshot` sums it. Prompt caching only applies
above the provider's minimum cacheable length, so short briefs show no cache savings.

## Nodes

`plan` asks the writer model for an `Outline` with structured output.

`gather_context` runs a tool loop of at most `MAX_TOOL_ITERATIONS` turns. The model can
call `fetch_url` (limited to the brief's URLs, with a size cap and a timeout) and the
WordPress research tools (`list_posts`, `get_post`, `list_tags`). A second, tool-free call
turns the tool results into `ResearchNotes`. The notes are then checked against the raw
tool output: a source that was not fetched is dropped, a fact with a number that is not on
the page is dropped, and related posts are replaced by the id, title and link WordPress
returned. Categories and tags are loaded in code, not by the model. If either list was
cut at `WP_MAX_TERM_PAGES` pages, the run gets a warning.

`write` produces a `Draft`: title, slug, excerpt, meta description, semantic HTML,
category ids and tag names. Category ids that are not among the site's categories are
dropped. On a revision it gets the previous draft and the review's fixes. When
`human_feedback` is set, it gets the previous draft and the editor's note instead, and the
note takes precedence over review fixes for that call. `write` clears `human_feedback`, so a
note applies once. A draft written from a note starts `revision_count` at 0 again, so the
review loop has its full `MAX_REVISIONS` budget to fix anything the note broke. A run
makes at most `MAX_HUMAN_REVISIONS × (MAX_REVISIONS + 1)` extra writes.

`review` runs code checks first: meta description length (140 to 160 characters), allowed
HTML tags and attributes, and numbers in the draft that do not appear in the notes. The
reviewer model then fills a checklist (matches the brief, structure, keyword use,
readability, no unsupported claims). The review fails if any check fails, and each failure
comes with a fix. `route_after_review` sends the draft back to `write` while it fails and
`revision_count` is below `MAX_REVISIONS`.

`human_approval` calls `interrupt()` with an `ApprovalRequest`. The run stops, the
checkpoint is saved, and nothing else happens until someone resumes it with an
`ApprovalDecision`. The decision model is strict, so a string like `"yes"` is not an
approval. A decision can approve, reject, or revise (`revise=True`, which needs a note and
cannot also approve). A revise sets `status` to `writing`, stores the note in
`human_feedback`, adds one to `human_revision_count` and `route_after_approval` sends the
run to `write`. `ApprovalRequest.human_revisions_left` is `MAX_HUMAN_REVISIONS` minus the
revisions used. With none left, a revise decision raises `AgentError`; the runner and
`RunService` refuse it earlier with `RunStateError`, before anything runs.

`publish_draft` converts the HTML to Gutenberg block markup, creates the tags that do not
exist yet, and calls `create_draft_post`. If the server reused an existing draft with the
same slug, title and content, the run gets a warning. If WordPress
answers with any status other than `draft`, the run is marked failed and the post id is
kept so the post can be found. The converter is tested against the block grammar, not
the editor itself, so an unusual structure may open as a Custom HTML block.

## Running and resuming

`AgentRunner.open(settings)` builds the model clients first, so a missing API key fails
before anything else starts. It then opens the SQLite checkpointer, starts the MCP server
as a subprocess and opens an HTTP client. All three close together.

`start` refuses a run id that already exists. `resume` refuses a run that is not waiting
for approval, and a revise decision when no human revisions are left. `retry` refuses a
run unless a node failed in it and it has no post id. Each run id has an
`asyncio.Lock`, so two approvals of the same run cannot
create two drafts in one process.

The checkpointer only loads types on an allowlist, which is derived from `AgentState`, so
a new field cannot be left out. Brief URLs are validated as URLs but stored as strings,
because the checkpoint serializer cannot write `HttpUrl`.

## Failures

Errors raised on purpose are subclasses of `AgentError`: `LLMError`, `ToolError`,
`BriefError`, `RunNotFoundError`, `RunStateError` and a few more. When a node raises, the
runner writes `status: failed` and an error message into the checkpoint as if
`publish_draft` had run, so the run is finished rather than stuck. An `AgentError` keeps
its message. Any other exception is reported as "internal error (TypeError); see the
logs", because exception text can carry URLs or file paths. Cancellation and Ctrl-C also
mark the run failed before they propagate. Usage recorded before the failure is kept.

A run that failed in any node can be retried. The runner stores the node that raised in
`failed_node`, and the snapshot reports `can_retry` when it is set, no `wp_post_id`
exists, no approval is pending, and the run is either `failed` or its newest checkpoint is
still the fork a retry wrote (a retry that died mid-way). `failed_node` stays set after a
retry succeeds, but the checkpoints after it come from the graph loop, so a later pass
through the same node is not retryable. `retry` looks in the checkpoint history for
the newest checkpoint whose pending node is `failed_node`, which is the state that node
started from, and streams from it with `Command(update=...)`. That forks the history: the
nodes before it are not run or paid for again, and the failed attempt stays in the old
branch. The update carries `failed_node` and the usage entries the failed attempt
billed (the `LLMError` usage), which the fork does not have, so the snapshot counts every
billed call once. It is applied as input, not with `aupdate_state`, because that call
evaluates the edges of the last writer and the review router needs the run context,
which it does not get.

A failure in `human_approval` (an invalid decision) is retryable: the node starts again,
the run waits for approval and a new decision can be sent. A run waiting for approval is
never retryable. If a failed `publish_draft` request timed out, WordPress may have
created the post anyway; the retry finds the draft with the same slug, title and content
and reuses it instead of creating a second one, and the run gets a warning. A draft that
WordPress changed on save (a filtered title or body) does not match, so on a site with
such filters, check for the draft before retrying a timed-out publish. A run whose post
already has an id is not retryable.

The per-run lock that stops two approvals at once lives in memory, so it covers one
process. Several API workers on the same checkpoint file are not supported.

Model calls have a timeout and retries from the provider SDK (`LLM_TIMEOUT_S`,
`LLM_MAX_RETRIES`). WordPress calls have their own timeouts and retries, described in
[mcp-server.md](mcp-server.md), and every MCP call has `MCP_CALL_TIMEOUT_S`. A tool that
fails inside `gather_context` returns its error to the model as text instead of ending
the run.

## API

`RunService` runs each run as an `asyncio` task, so `POST /runs` returns `202` right away
and the client polls `GET /runs/{id}`. At most `MAX_CONCURRENT_RUNS` run at once; the next
one gets `429`. The busy slot is claimed before the first `await`, so two requests cannot
both take the last one.

An `Idempotency-Key` header becomes the run id. Sending the same key again returns the
existing run with `200` instead of starting a second one. The n8n workflow relies on this
when it retries.

`POST /runs/{id}/retry` follows the same pattern as approve: `404` for an unknown run,
`409` while the run is busy or when it cannot be retried, `202` otherwise.

On shutdown the service cancels its tasks and marks any run that had not finished as
failed with "the server stopped during the run".

| Endpoint | What it does |
|---|---|
| `POST /runs`, `POST /webhooks/n8n` | Start a run. |
| `GET /runs/{id}` | Read a run. |
| `POST /runs/{id}/approve`, `/reject` | Decide, with an optional note. |
| `POST /runs/{id}/revise` | Send the draft back to `write`. The note is required, 1 to 2000 characters. |

| Error | Status |
|---|---|
| `RunNotFoundError` | 404 |
| `RunStateError` (exists, not awaiting approval, busy, no human revisions left) | 409 |
| `RunLimitError` | 429 |
| invalid brief, missing or blank revise note | 422 |
| missing or wrong bearer token or webhook secret | 401 |
| webhook without `WEBHOOK_SECRET` | 503 |

## Configuration

Everything is in one `Settings` class (pydantic-settings), read from the environment and
`.env`. Empty values fall back to the defaults, so the blank lines in `.env.example` do
not turn into empty keys. The WordPress settings live in `WPSettings` in `wp_mcp`, because
the MCP server also runs without the agent; the agent passes them to the subprocess
through its environment.

## Tests

Unit tests use `FakeChatModel`, which replays scripted messages with token usage, and a
fake WordPress gateway. Every node, tool, the runner, the CLI and the API have their own
tests, and one test runs a whole graph against a real SQLite checkpoint. The integration
tests, which need `WPAGENT_INTEGRATION=1` and the Docker site, run the real MCP server
against WordPress.
