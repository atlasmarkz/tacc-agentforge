#!/usr/bin/env bash
# Register the shared, empty proxy environment with the original SDK.
set -euo pipefail
sdk=${TCLOUD_BIN:-tcloud}
command -v "$sdk" >/dev/null 2>&1 || {
    echo 'tcloud not found. Install the official SDK and add it to PATH; see README.' >&2
    exit 1
}
common=/mnt/home/public/course-envs/CSIT6000V/common
prefix="$common/proxy"
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
if ! environments=$("$sdk" env ls 2>&1); then
    printf '%s\nCould not list platform environments. Check config/init and contact the TA if it persists.\n' "$environments" >&2
    exit 1
fi
if [[ "$environments" == *'[Tcloud Error]'* || "$environments" != *'# conda environments:'* ]]; then
    printf '%s\nPlatform account initialization is incomplete; contact the TA.\n' "$environments" >&2
    exit 1
fi
listing=$("$sdk" ls -a -l ../.Miniconda3/envs/ 2>&1)
[[ "$listing" != *'[Tcloud Error]'* ]]
has_index=false
while IFS= read -r entry; do
    if [[ "$entry" =~ (^|[[:space:]])proxy($|[[:space:]]) ]]; then has_index=true; fi
done <<< "$listing"
packages=$("$sdk" env ls -n proxy 2>&1 || true)
if [[ "$packages" == *'# packages in environment at '* && "$packages" != *"# packages in environment at $prefix:"* ]]; then
    printf '%s\nAn existing proxy environment uses another path; nothing changed. Contact the TA.\n' "$packages" >&2
    exit 1
fi
if [[ "$packages" != *"# packages in environment at $prefix:"* ]]; then
    home_listing=$("$sdk" ls -a -l ../ 2>&1)
    [[ "$home_listing" != *'[Tcloud Error]'* ]]
    has_config=false
    while IFS= read -r entry; do
        if [[ "$entry" =~ (^|[[:space:]])\.condarc($|[[:space:]]) ]]; then has_config=true; fi
    done <<< "$home_listing"
    if "$has_config"; then
        "$sdk" download ../.condarc "$stage/.condarc"
        # Append at the existing list indentation, preserving order and other settings.
        # Complete this check before uploading either a backup or the replacement.
        if ! awk -v common="$common" '
            function reject() {bad=1; exit 1}
            function append_path() {
                print (items ? indent : "  ") "- " common eol
                active=0
            }
            {
                line=$0
                sub(/\r$/, "", line)
                if (NR==1 && $0 ~ /\r$/) eol="\r"
                if (line ~ /^[[:space:]]*($|#)/) {print; next}
                if (line ~ /^envs_dirs[ \t]*:/) {
                    if (seen++ || line !~ /^envs_dirs[ \t]*:[ \t]*($|#)/) reject()
                    active=1; root=1
                    print; next
                }
                if (active && line ~ /^ *-[ \t]+[^ \t]/) {
                    match(line, /^ */)
                    current=substr(line, 1, RLENGTH)
                    if (items && current!=indent) reject()
                    indent=current; items++
                    print; next
                }
                if (line ~ /^[A-Za-z_][A-Za-z0-9_-]*[ \t]*:/) {
                    if (active) append_path()
                    root=1
                    print; next
                }
                # Refuse quoted/flow/document/merge-key layouts and multiline list items.
                if (active || !root || (line !~ /^ +/ && line !~ /^-[ \t]+/)) reject()
                print
            }
            END {
                if (bad) exit 1
                if (active) append_path()
                if (!seen) {print "envs_dirs:" eol; print "  - " common eol}
            }
        ' "$stage/.condarc" > "$stage/merged"; then
            echo 'Existing Conda configuration needs TA review; nothing changed. No configuration was uploaded.' >&2
            exit 1
        fi
        backup=".condarc-before-proxy-$(date +%Y%m%d-%H%M%S)-$$"
        cp "$stage/.condarc" "$stage/$backup"
        if ! backup_log=$("$sdk" upload "$stage/$backup" ../ 2>&1); then
            printf '%s\nCould not save the backup; original configuration was not replaced.\n' "$backup_log" >&2
            exit 1
        fi
        printf '%s\n' "$backup_log" | tr -d '\000'
        if [[ "$backup_log" == *'[Tcloud Error]'* ]]; then
            echo 'Could not save the backup; original configuration was not replaced.' >&2
            exit 1
        fi
        mv "$stage/merged" "$stage/.condarc"
        printf 'Adding proxy search path; preserving existing settings; backup: %s\n' "$backup"
    else
        printf 'envs_dirs:\n  - %s\n' "$common" > "$stage/.condarc"
    fi
    "$sdk" upload "$stage/.condarc" ../ | tr -d '\000'
fi
if ! "$has_index"; then
    mkdir "$stage/proxy"
    "$sdk" upload "$stage/proxy" ../.Miniconda3/envs/ | tr -d '\000'
fi
packages=$("$sdk" env ls -n proxy 2>&1)
[[ "$packages" == *"# packages in environment at $prefix:"* ]]
echo 'Shared proxy environment registered; no packages installed.'
