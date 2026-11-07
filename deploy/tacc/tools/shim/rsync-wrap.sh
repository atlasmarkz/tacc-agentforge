#!/usr/bin/env bash
# Stand-in for the rsync call made by the native Windows tcloud SDK.
#
# The SDK runs:  rsync -av --progress --delete --rsh "/usr/bin/ssh -p P -i KEY" \
#                      E:\local\dir  user@host:\remote\dir
# which fails on this machine: MSYS rsync reads "E:" as a host, the remote path
# uses backslashes, and the bundled MSYS rsync cannot talk through any available
# ssh (Windows OpenSSH: connection reset; Git ssh: dup() failed, runtime mismatch).
#
# For that exact upload shape we copy the directory with `tar | ssh` instead,
# giving the same result as rsync -a --delete: <remote>/<basename> mirrors <local>.
# Any other call is passed through to the real rsync unchanged.
real=/e/Code/.cache/tcloud/bin/rsync.exe
ssh_win='C:/Windows/System32/OpenSSH/ssh.exe'
log() { if [ -n "${RSYNC_SHIM_LOG:-}" ]; then printf '%s\n' "$*" >> "$RSYNC_SHIM_LOG"; fi; }

rsh=''; pos=(); opts=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rsh) rsh=$2; shift 2 ;;
    -*)    opts+=("$1"); shift ;;
    *)     pos+=("$1"); shift ;;
  esac
done

drive='^[A-Za-z]:[\\/]'
remote='^([^/:@ ]+)@([^:/ ]+):(.*)$'
if [ "${#pos[@]}" -eq 2 ] && [[ "${pos[0]}" =~ $drive ]] && [[ "${pos[1]}" =~ $remote ]]; then
  user=${BASH_REMATCH[1]}; host=${BASH_REMATCH[2]}
  rdir=$(printf '%s' "${BASH_REMATCH[3]}" | tr '\134' '/')
  src=$(cygpath -u "${pos[0]}"); src=${src%/}
  base=$(basename "$src"); parent=$(dirname "$src")
  # rsh looks like "/usr/bin/ssh -p PORT -i KEY"
  port=$(sed -n 's/.* -p \([0-9]*\).*/\1/p' <<< "$rsh")
  key=$(sed -n 's/.* -i \([^ ]*\).*/\1/p' <<< "$rsh")
  log "tar-upload: $src -> $user@$host:$rdir/$base (port ${port:-22})"
  tar -C "$parent" -cf - "$base" | "$ssh_win" ${port:+-p "$port"} ${key:+-i "$key"} \
      -o BatchMode=yes "$user@$host" \
      "mkdir -p '$rdir' && rm -rf '$rdir/$base' && tar -C '$rdir' -xf -"
  exit $?
fi

log "passthrough: $*"
exec "$real" "${opts[@]}" ${rsh:+--rsh "$rsh"} "${pos[@]}"
