#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
report="/tmp/ib_health_report_${ts}.txt"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing: $1" >&2; exit 1
  fi
}

require ibstat
require ibv_devinfo
if command -v perfquery >/dev/null 2>&1; then have_perf=1; else have_perf=0; fi
if command -v opensm >/dev/null 2>&1; then have_opensm=1; else have_opensm=0; fi

active_links="$(ibstat | grep -c 'state: Active' || true)"
down_links="$(ibstat | grep -c 'state: Down' || true)"

mtu_lines="$(ibv_devinfo | grep -i 'active_mtu' || true)"

if [ "$have_perf" = "1" ]; then
  perf_errors="$(perfquery 2>/dev/null | grep -E 'Error|Symbol|Link' | awk '$NF!="0"' | wc -l | tr -d '[:space:]')"
else
  perf_errors="NA"
fi

sm_state="$(systemctl is-active opensm 2>/dev/null || echo 'unknown')"

{
  echo "IB Health Report"
  echo "time: $(date)"
  echo "host: $(hostname)"
  echo "active_links: ${active_links}"
  echo "down_links: ${down_links}"
  echo "opensm_state: ${sm_state}"
  echo "mtu_info:"
  echo "${mtu_lines}"
  echo "perf_errors_nonzero_count: ${perf_errors}"
} > "$report"

echo "$report"
