#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/tmp/ib_diag_report.txt}"
if ! command -v ibdiagnet >/dev/null 2>&1; then echo "missing: ibdiagnet"; exit 1; fi
diag="$(ibdiagnet 2>&1 || true)"
errors="$(echo "$diag" | grep -Ei 'error|bad|fail' | wc -l | tr -d '[:space:]')"
active="NA"
down="NA"
if command -v iblinkinfo >/dev/null 2>&1; then
  active="$(iblinkinfo | grep -c 'ACTIVE' || true)"
  down="$(iblinkinfo | grep -c 'DOWN' || true)"
fi
{
  echo "ibdiagnet_error_lines: ${errors}"
  echo "active_links: ${active}"
  echo "down_links: ${down}"
} > "$OUT"
echo "$OUT"
