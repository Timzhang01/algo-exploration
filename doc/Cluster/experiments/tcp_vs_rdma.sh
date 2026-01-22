#!/usr/bin/env bash
set -euo pipefail
ROLE="${1:-}"
SERVER_IP="${2:-}"
DURATION="${3:-10}"
SIZE="${4:-1048576}"
OUT="${5:-tcp_vs_rdma.csv}"
usage() { echo "usage: tcp_vs_rdma.sh server|client <server_ip> <duration_sec> <size_bytes> <out_csv>"; exit 1; }
if [ -z "$ROLE" ]; then usage; fi

run_servers() {
  if ! command -v iperf3 >/dev/null 2>&1; then echo "missing: iperf3"; exit 1; fi
  if ! command -v ib_read_bw >/dev/null 2>&1; then echo "missing: ib_read_bw"; exit 1; fi
  iperf3 -s >/dev/null 2>&1 &
  iperf_pid=$!
  ib_read_bw -R -s "$SIZE" -D "$DURATION" >/dev/null 2>&1 &
  rdma_pid=$!
  trap 'kill -9 $iperf_pid $rdma_pid 2>/dev/null || true' EXIT
  echo "servers_started"
  wait
}

to_gbps() {
  bw="$1"; unit="$2"
  if [ "$unit" = "Gbps" ] || [ "$unit" = "Gbits/sec" ]; then awk -v v="$bw" 'BEGIN{printf "%.3f", v+0}'
  elif [ "$unit" = "Mbits/sec" ]; then awk -v v="$bw" 'BEGIN{printf "%.3f", (v+0)/1000.0}'
  elif [ "$unit" = "Mb/s" ] || [ "$unit" = "Mbps" ]; then awk -v v="$bw" 'BEGIN{printf "%.3f", (v+0)/1000.0}'
  else awk -v v="$bw" 'BEGIN{printf "%.3f", v+0}'
  fi
}

run_client() {
  if [ -z "$SERVER_IP" ]; then echo "missing server ip"; exit 1; fi
  if ! command -v iperf3 >/dev/null 2>&1; then echo "missing: iperf3"; exit 1; fi
  if ! command -v ib_read_bw >/dev/null 2>&1; then echo "missing: ib_read_bw"; exit 1; fi
  echo "method,size,avg_bw_Gbps,avg_msg_rate_Mpps" > "$OUT"
  t_out="$(iperf3 -c "$SERVER_IP" -t "$DURATION" 2>&1 || true)"
  t_line="$(echo "$t_out" | grep -E 'sender' | tail -n1 || true)"
  t_bw="$(echo "$t_line" | awk '{print $(NF-2)}')"
  t_unit="$(echo "$t_line" | awk '{print $(NF-1)}')"
  t_gbps="$(to_gbps "${t_bw:-0}" "${t_unit:-Mbits/sec}")"
  echo "TCP,${SIZE},${t_gbps:-0},NA" >> "$OUT"
  r_out="$(ib_read_bw -R -s "$SIZE" -D "$DURATION" "$SERVER_IP" 2>&1 || true)"
  r_bw_line="$(echo "$r_out" | grep -E 'BW average|BW Avg' | tail -n1 || true)"
  r_rate_line="$(echo "$r_out" | grep -E 'MsgRate average|MsgRate Avg' | tail -n1 || true)"
  r_bw="$(echo "$r_bw_line" | awk '{print $(NF-1)}')"
  r_unit="$(echo "$r_bw_line" | awk '{print $NF}')"
  r_gbps="$(to_gbps "${r_bw:-0}" "${r_unit:-Gb/s}")"
  r_mpps="$(echo "$r_rate_line" | awk '{print $(NF-1)}')"
  echo "RDMA,${SIZE},${r_gbps:-0},${r_mpps:-0}" >> "$OUT"
  echo "$OUT"
}

case "$ROLE" in
  server) run_servers ;;
  client) run_client ;;
  *) usage ;;
esac
