#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-}"
MOUNTPOINT="${2:-}"
SIZE="${3:-64G}"
if [ -z "$ACTION" ] || [ -z "$MOUNTPOINT" ]; then echo "usage: tmpfs_cache.sh mount|unmount <mountpoint> [size]"; exit 1; fi
ensure_root() {
  if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      exec sudo -E "$0" "$ACTION" "$MOUNTPOINT" "$SIZE"
    else
      echo "need root"; exit 1
    fi
  fi
}
do_mount() {
  mkdir -p "$MOUNTPOINT"
  mount -t tmpfs -o "size=$SIZE" tmpfs "$MOUNTPOINT"
  chmod 777 "$MOUNTPOINT"
  echo "$MOUNTPOINT"
}
do_unmount() {
  umount "$MOUNTPOINT"
  rmdir "$MOUNTPOINT" || true
  echo "ok"
}
case "$ACTION" in
  mount) ensure_root; do_mount ;;
  unmount) ensure_root; do_unmount ;;
  *) echo "unsupported action: $ACTION"; exit 1 ;;
esac
