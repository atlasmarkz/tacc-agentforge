<h1 align="center"><img src="docs/images/logo.svg" width="42" alt=""> TACC AgentForge</h1>

<p align="center"><strong>A TACC-hosted shopping agent that uses a quantized 27B model to recommend products with source links and budget-checked totals.</strong></p>

<p align="center">
  <a href="#live-demo">Live Demo</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="app/README.md#运行">Run the Demo</a> ·
  <a href="docs/experiments/baseline.md">Evaluation</a>
</p>

## Live Demo

<p align="center">
  <img src="docs/images/browser-demo.gif" width="900" alt="The shopping agent searches for earbuds, validates the price against a $100 budget, and returns a recommendation with its source.">
</p>

The demo runs on TACC. The agent recommends earbuds costing **$22.99 before tax** and shows the source alongside the **$77.01** left in the budget. Idle waits are shortened in the recording.

## Architecture

![Target architecture with six layers: Application, Knowledge, Post-training, Inference, Evaluation, and TACC](docs/images/target-architecture.png)

*Target architecture, including cloud-hosted knowledge services and LoRA training.*

[Deployment runbook](deploy/tacc/service/RUNBOOK.md) · [Knowledge base](knowledge/README.md) · [Agent internals](agent/ARCHITECTURE.md)
