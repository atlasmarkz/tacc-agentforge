#!/usr/bin/env bash
# Create the GCE VM and the snapshot bucket (issue #25). Run on your own machine with gcloud logged in.
#
#   PROJECT=<gcp project> BUCKET=<globally unique bucket name> bash deploy/gcp/create-vm.sh
#
# Defaults stay inside the GCP Always Free tier: one e2-micro in us-west1 with a 30 GB standard disk, a
# regional bucket in the same region. For lower latency to the TACC gateway (Hong Kong) use
# ZONE=asia-east2-a MACHINE=e2-small, which is not free (the Google AI Pro monthly Cloud credit can cover it).
set -euo pipefail
: "${PROJECT:?set PROJECT}" "${BUCKET:?set BUCKET}"
ZONE="${ZONE:-us-west1-b}"
REGION="${ZONE%-*}"
MACHINE="${MACHINE:-e2-micro}"
VM="${VM:-agentforge}"

gcloud services enable compute.googleapis.com storage.googleapis.com --project "$PROJECT"

# Snapshots are written once and never replaced: versioning keeps any accidental overwrite recoverable, and
# publish-snapshot.sh refuses to overwrite an existing object.
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
  gcloud storage buckets update "gs://$BUCKET" --versioning
fi

# The VM only needs to read the bucket (storage-ro). No inbound firewall rule is added: the public entry is
# Cloudflare Tunnel (outbound only); SSH for administration goes through `gcloud compute ssh`.
gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
  --boot-disk-size 30GB --boot-disk-type pd-standard \
  --scopes storage-ro,logging-write,monitoring-write --shielded-secure-boot

echo
echo "VM $VM created in $ZONE. Next:"
echo "  gcloud compute scp deploy/gcp/setup-host.sh $VM:~ --zone $ZONE --project $PROJECT"
echo "  gcloud compute ssh $VM --zone $ZONE --project $PROJECT -- sudo bash setup-host.sh <repo url> <branch>"
