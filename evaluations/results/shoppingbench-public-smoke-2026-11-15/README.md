# ShoppingBench public audit smoke (2026-11-15)

This directory is a **scoring smoke**, not a new model run and not a final recovery result.

- Upstream: `204dfba767cf849fe33535e42293e8973fb017ea`
- Corpus: `documents.jsonl.gz` SHA256 `a0ba9a8618ce72a7b383a6de31acf6e5bf138873da0d13ad8a1a6b0271e8e9d1`, 1,471,795,629 bytes
- Index: `index-build.json` (19 files, 3,271,147,867 bytes, SHA256 `2012a83fcbf08e301ce2981413bf2bcb10fb2371316d89d6971aa78fc4cfc080`)
- Image: `agentforge-shoppingbench:204dfba` (`sha256:069f1f...`), CPU torch 2.9.1, openai 1.109.1
- Preflight: `ready=true`, no blockers
- Rollouts: historical interrupted voucher batch (`shoppingbench-interrupted-2026-11-14`), 250 records
- Audit: 250 tasks, missing 0, ASR 4.4%, budget match 6.4%, completion 6.0%, API-incomplete 204, scoring errors 0
- Quality label: `interrupted/API-incomplete; not a clean model comparison`

The audit was run with the official `eval_voucher` inside the isolated image. It verifies the rebuilt index and the audit/provenance path; it does **not** provide a clean model capability number and does not replace the final recovery report.
