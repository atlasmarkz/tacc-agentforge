# PR-Agent with MiniMax

This project includes an opt-in GitHub Actions integration for
[PR-Agent](https://github.com/The-PR-Agent/pr-agent), using the
[MiniMax OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api).
Calls use MiniMax API usage and billing, separately from GitHub Copilot and
the TACC deployment. Check the actual workflow result and published feedback
to confirm that a review completed.

The workflow reviews ready pull requests from branches in this repository when
opened, reopened, marked ready, or updated. It does not run for fork or draft
PRs, check out PR code, or automatically change PR descriptions. PR diffs and
related context are sent to MiniMax. The GitHub token has permissions needed
to read the repository and post feedback. The pinned PR-Agent image is v0.49.0.

## Setup

1. Obtain a MiniMax API key with access to a text model. Keep it private.
2. In GitHub, open this repository's **Settings → Secrets and variables →
   Actions → Secrets**, select **New repository secret**, and create
   `MINIMAX_API_KEY` with your API key. Do not commit or paste the key in chat.
3. In **Settings → Secrets and variables → Actions → Variables**, create
   `PR_AGENT_ENABLED` with the value `true` after the workflow is present on
   the branch used by your PR.
4. The workflow defaults to `MiniMax-M3.1-Flash` and
   `https://api.minimax.cn/v1`. If your account uses a different model or
   regional endpoint, change both the model settings and/or endpoint in
   `.github/workflows/pr-agent.yml` to the values provided by your MiniMax
   console. The `openai/` prefix selects the compatible protocol in LiteLLM;
   it is not part of the MiniMax model ID. Use only a trusted endpoint, because
   it receives your API key and PR content.
5. Open or update a non-draft PR from a branch in the same repository and
   inspect the Actions run and PR feedback. A skipped or green job alone is
   not proof of a successful review. Availability, limits, request parameters,
   response format, and pricing depend on your MiniMax account and model.

## Large PRs and timeouts

The workflow enables PR-Agent v0.49.0's native large-PR chunking with at most
six review chunks. Each request keeps the configured 32,000-token context cap
and an 8,192-token output cap. The output cap includes thinking tokens. These
are request budgets, not a guarantee that every file fits: inspect the review's
coverage footer for omitted files, clipped patches or failed chunks. Partial
feedback is not a complete review.

MiniMax-M3.1-Flash defaults to maximum thinking depth when `reasoning_effort`
is omitted. The workflow explicitly registers the compatible model with
PR-Agent and sends `reasoning_effort=low`; thinking stays enabled. The
registration is necessary when LiteLLM's model metadata does not recognize
the compatible endpoint's model ID. See the
[MiniMax request parameters](https://platform.minimax.io/docs/api-reference/text-openai-api)
and the pinned
[PR-Agent configuration](https://github.com/The-PR-Agent/pr-agent/blob/v0.49.0/pr_agent/settings/configuration.toml).

Each request has a 300-second timeout. Client retries are set to zero and
same-model retries on timeouts are disabled, avoiding repeated long waits
inside the 15-minute Actions job. The six-chunk limit applies to chunk
planning; other transient-error handling can still retry. No fallback model
is configured. If a run fails, inspect **Actions → PR Agent → Review pull
request** before rerunning it. A model timeout is different from a missing
API key, quota exhaustion or a GitHub permission failure; raising the timeout
does not fix those errors. Never print the API credential while debugging.

The workflow does not check out or execute PR code. Keep the pinned container,
trusted endpoint and current permission/branch restrictions when adjusting
these limits.

Set `PR_AGENT_ENABLED` to `false` to stop automatic reviews. Repository
administrators and people able to modify trusted workflows can potentially
use the secret. The key is injected into the pinned PR-Agent container only;
restrict repository write access accordingly.
