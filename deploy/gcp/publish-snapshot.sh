#!/usr/bin/env bash
# Publish a snapshot archive to GCS once; never overwrites (issue #25).
#
#   bash deploy/gcp/publish-snapshot.sh assets/knowledge/pilot-snapshot.zip <bucket>
#
# The archive is the ZIP `python -m knowledge bootstrap` verifies (top-level folder = snapshot ID). The object
# name is the snapshot ID, so a published snapshot keeps its ID and hash; publishing the same ID twice fails.
# Afterwards put the printed URL and SHA256 in configs/knowledge.json (seed.url, seed.sha256) and commit that
# small change instead of the ZIP.
set -euo pipefail
archive="${1:?archive}"
bucket="${2:?bucket}"
snapshot_id="$(python3 -I -c 'import sys, zipfile; print(zipfile.ZipFile(sys.argv[1]).namelist()[0].split("/")[0])' "$archive")"
case "$snapshot_id" in snapshot_*) ;; *) echo "unexpected archive layout: $snapshot_id" >&2; exit 1 ;; esac
sha256="$(python3 -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$archive")"
url="gs://$bucket/snapshots/$snapshot_id.zip"

# --if-generation-match=0: the upload only succeeds when no object of that name exists.
gcloud storage cp --if-generation-match=0 "$archive" "$url"
gcloud storage objects update "$url" --custom-metadata "sha256=$sha256,snapshot_id=$snapshot_id"

echo "published $url"
echo "  seed.url    = $url"
echo "  seed.sha256 = $sha256"
echo "  VM env      : KNOWLEDGE_SEED_URL=$url"
