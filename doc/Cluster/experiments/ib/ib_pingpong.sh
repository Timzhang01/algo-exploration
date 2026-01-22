#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
SERVER_IP="${2:-}"
SIZE="${3:-1024}"
ITERS="${4:-1000}"
OUT="${5:-ib_pingpong.csv}"

run_ib_send_lat_server() {
  ib_send_lat -R -s "$SIZE" -n "$ITERS"
}

run_ib_send_lat_client() {
  out="$(ib_send_lat -R -s "$SIZE" -n "$ITERS" "$SERVER_IP" 2>&1 || true)"
  lat_line="$(echo "$out" | grep -E 'Latency average|Lat Avg' | tail -n1 || true)"
  avg_us="$(echo "$lat_line" | awk '{print $(NF-1)}')"
  echo "size_bytes,avg_latency_us" > "$OUT"
  echo "$SIZE,${avg_us:-0}" >> "$OUT"
  echo "$OUT"
}

run_ibv_rc_pingpong_server() {
  ibv_rc_pingpong
}

run_ibv_rc_pingpong_client() {
  out="$(ibv_rc_pingpong "$SERVER_IP" 2>&1 || true)"
  avg_line="$(echo "$out" | grep -E 'avg' | tail -n1 || true)"
  avg_us="$(echo "$avg_line" | awk -F'[= ]' '{for(i=1;i<=NF;i++){if($i=="avg"){print $(i+1);break}}}')"
  echo "size_bytes,avg_latency_us" > "$OUT"
  echo "$SIZE,${avg_us:-0}" >> "$OUT"
  echo "$OUT"
}

server_main() {
  if command -v ib_send_lat >/dev/null 2>&1; then
    run_ib_send_lat_server
  else
    run_ibv_rc_pingpong_server
  fi
}

client_main() {
  if [ -z "$SERVER_IP" ]; then echo "missing server ip" >&2; exit 1; fi
  if command -v ib_send_lat >/dev/null 2>&1; then
    run_ib_send_lat_client
  else
    run_ibv_rc_pingpong_client
  fi
}

if [ "${ROLE:-}" = "server" ]; then
  server_main
elif [ "${ROLE:-}" = "client" ]; then
  client_main
else
  echo "usage: ib_pingpong.sh server|client <server_ip> <size> <iters> <out_file>"; exit 1
fi
