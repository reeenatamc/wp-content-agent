# n8n example

`brief-to-draft.json` is an n8n workflow that collects a brief in an n8n form, starts a
run through `POST /webhooks/n8n`, polls `GET /runs/{id}` every 15 seconds and ends with
the run status, the draft title and `approve_url`. Approval stays a human step: send a
POST to `approve_url` with `Authorization: Bearer <API_TOKEN>`, or run
`wpagent approve <run_id>`. Polling stops after 20 minutes with `timed_out: true`.

I tested it on n8n 1.110 with n8n in Docker and the agent on the host.

## Setup

1. Start the agent so n8n's container can reach it. It needs an API token to listen
   beyond loopback:

   ```bash
   API_TOKEN=... WEBHOOK_SECRET=... uv run wpagent serve --host 0.0.0.0
   ```

2. In n8n, create two credentials of type Header Auth:
   - `wpagent webhook secret`: name `X-Webhook-Secret`, value = `WEBHOOK_SECRET`
   - `wpagent API token`: name `Authorization`, value = `Bearer <API_TOKEN>`

3. Import the workflow (Workflows, Import from file). The export carries no credential
   ids, so pick the two credentials by hand in the "Start run" and "Get run" nodes.
   Change `agentUrl` in "Build brief" if the agent is not at
   `http://host.docker.internal:8000`. On Linux that name only exists if the n8n
   container is started with `--add-host=host.docker.internal:host-gateway`.

4. Activate the workflow and open the form URL shown on the "Brief form" node.

The `Idempotency-Key` header is built once per execution, so if n8n retries "Start
run" the agent returns the existing run instead of starting a second one. A brief the
agent rejects (for example a topic under 3 characters) fails "Start run" with the 422
detail, which shows in the execution log; the form itself has already answered.
