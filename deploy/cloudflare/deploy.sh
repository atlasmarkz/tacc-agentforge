#!/usr/bin/env bash
# Build and deploy the page to Cloudflare Pages (issue #25).
# The page is app/static/index.html unchanged: it calls relative /api/* paths, which functions/api/ proxies.
#   bash deploy/cloudflare/deploy.sh            # needs `npx wrangler login` once
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
rm -rf "$here/public"
mkdir -p "$here/public"
cp "$repo/app/static/index.html" "$here/public/index.html"
cd "$here"
npx --yes wrangler@4 pages deploy public --project-name "${PAGES_PROJECT:-tacc-agentforge}" --branch main
