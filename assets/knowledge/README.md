# Versioned offline seed snapshot

`pilot-snapshot.zip` contains one complete sanitized, immutable snapshot folder:

```text
snapshot_5bcd00723fb0b7589cbe50c8/
  catalog.sqlite
  raw/
  manifest.json
  schema.sql
  category_schemas.json
  provenance.json
  quality_report.json
```

The pinned archive SHA256 is in `configs/knowledge.json`. Commit this archive
together with the code/configuration so teammates can run `python -m knowledge
bootstrap` after cloning, without API keys or another collection run.

The seed is about 7 MB. Extracted runtime data stay outside Git. It contains
80 detail-bearing provider groups, 380 review records and 3,525 evidence records.
SKU identity/fees are not universally verified; this archived catalog is not a
live-price guarantee. Provenance records retain historical import locations as
audit metadata; runtime lookup uses the snapshot's relative archive paths.

To update a seed, export and verify a new complete snapshot, ZIP its folder
(including its manifest and referenced raw files), then update both snapshot ID
and archive SHA256 in the versioned configuration. Never edit a published snapshot
in place.
