**TACC AgentForge** is a shopping advisor that uses a quantized 27B model on TACC to recommend products with source links and verified costs.

![TACC AgentForge logo](docs/images/logo.svg)

## Live Demo

![Live TACC demo showing product search, budget validation, and source links](docs/images/browser-demo.gif)

Recorded on TACC. Idle waits are shortened.

## Architecture

The browser connects through a local backend and an SSH tunnel to the Agent on TACC. The Agent uses Qwen3.8-27B via vLLM, retrieves evidence from a SQLite catalog, and validates costs before returning a recommendation.

![Planned system architecture for TACC AgentForge](docs/images/planned-architecture.svg)
