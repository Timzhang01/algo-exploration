#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
PROTO="${2:-read}"
SERVER_IP="${3:-}"
DURATION="${4:-5}"
SIZES="${5:-64,256,1024,4096,65536,1048576}"
OUT="${6:-ib_bw_${PROTO}.csv}"

cmd_for_proto() {
  case "$PROTO" in
    read) echo "ib_read_bw" ;;
    write) echo "ib_write_bw" ;;
    send) echo "ib_send_bw" ;;
    *) echo "unsupported proto: $PROTO" >&2; exit 1 ;;
  esac
}

run_server() {
  cmd="$(cmd_for_proto)"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing command: $cmd" >&2; exit 1
  fi
  "$cmd" -R -D "$DURATION"
}

to_gbps() {
  bw="$1"; unit="$2"
  if [ "$unit" = "Gb/s" ] || [ "$unit" = "Gbps" ]; then
    awk -v v="$bw" 'BEGIN{printf "%.3f\n", v+0}'
  elif [ "$unit" = "Mb/s" ] || [ "$unit" = "Mbps" ]; then
    awk -v v="$bw" 'BEGIN{printf "%.3f\n", (v+0)/1000.0}'
  else
    awk -v v="$bw" 'BEGIN{printf "%.3f\n", v+0}'
  fi
}

run_client() {
  if [ -z "$SERVER_IP" ]; then
    echo "missing server ip" >&2; exit 1
  fi
  cmd="$(cmd_for_proto)"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing command: $cmd" >&2; exit 1
  fi
  IFS=, read -r -a arr <<< "$SIZES"
  echo "proto,size,avg_bw_Gbps,avg_msg_rate_Mpps" > "$OUT"
  for sz in "${arr[@]}"; do
    out="$("$cmd" -R -s "$sz" -D "$DURATION" "$SERVER_IP" 2>&1 || true)"
    bw_line="$(echo "$out" | grep -E 'BW average|BW Avg' | tail -n1 || true)"
    rate_line="$(echo "$out" | grep -E 'MsgRate average|MsgRate Avg' | tail -n1 || true)"
    bw_val="$(echo "$bw_line" | awk '{print $(NF-1)}')"
    bw_unit="$(echo "$bw_line" | awk '{print $NF}')"
    rate_val="$(echo "$rate_line" | awk '{print $(NF-1)}')"
    gbps="$(to_gbps "${bw_val:-0}" "${bw_unit:-Gb/s}")"
    echo "$PROTO,$sz,$gbps,${rate_val:-0}" >> "$OUT"
  done
  echo "$OUT"
}

usage() {
  echo "usage:"
  echo "server: ib_bw_test.sh server <proto:read|write|send> <ignored> <duration> <sizes_csv> <out_file>"
  echo "client: ib_bw_test.sh client <proto:read|write|send> <server_ip> <duration> <sizes_csv> <out_file>"
}

if [ "$ROLE" = "server" ]; then
  run_server
elif [ "$ROLE" = "client" ]; then
  run_client
else
  usage; exit 1
fi
