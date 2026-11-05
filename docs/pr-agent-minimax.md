# PR-Agent with MiniMax

This project includes an opt-in GitHub Actions integration for
[PR-Agent](https://github.com/The-PR-Agent/pr-agent), using the
[MiniMax OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api).
It has not yet been tested against a MiniMax account. Calls use MiniMax API
usage and billing, separately from GitHub Copilot and the TACC deployment.

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

Set `PR_AGENT_ENABLED` to `false` to stop automatic reviews. Repository
administrators and people able to modify trusted workflows can potentially
use the secret. The key is injected into the pinned PR-Agent container only;
restrict repository write access accordingly.
