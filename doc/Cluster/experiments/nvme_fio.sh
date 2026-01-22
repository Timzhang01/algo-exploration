#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-}"
MODE="${2:-seq_read}"
RUNTIME="${3:-10}"
OUT="${4:-nvme_fio.csv}"
if [ -z "${TARGET}" ]; then echo "usage: nvme_fio.sh <target_file_or_device> <seq_read|seq_write|rand_read|rand_write> <runtime_sec> <out_csv>"; exit 1; fi
if ! command -v fio >/dev/null 2>&1; then echo "missing: fio"; exit 1; fi
name="fio_${MODE}"
rw=""
bs=""
case "$MODE" in
  seq_read) rw="read"; bs="1M" ;;
  seq_write) rw="write"; bs="1M" ;;
  rand_read) rw="randread"; bs="4k" ;;
  rand_write) rw="randwrite"; bs="4k" ;;
  *) echo "unsupported mode: $MODE"; exit 1 ;;
esac
out="$(fio --name="$name" --filename="$TARGET" --rw="$rw" --bs="$bs" --time_based=1 --runtime="$RUNTIME" --ioengine=libaio --direct=1 --numjobs=1 --group_reporting 2>&1 || true)"
bw_line="$(echo "$out" | grep -E 'READ:.*BW=|write:.*BW=' | tail -n1 || true)"
iops="$(echo "$bw_line" | sed -n 's/.*IOPS=\([0-9\.kMG]*\).*/\1/p')"
bw_num="$(echo "$bw_line" | sed -n 's/.*BW=\([0-9\.]*\).*/\1/p')"
bw_unit="$(echo "$bw_line" | sed -n 's/.*BW=[0-9\.]*\([A-Za-z\/]*\).*/\1/p')"
to_gibps() {
  n="$1"; u="$2"
  if [ "$u" = "MiB/s" ]; then awk -v v="$n" 'BEGIN{printf "%.3f", (v+0)/1024.0}'
  elif [ "$u" = "MB/s" ]; then awk -v v="$n" 'BEGIN{printf "%.3f", (v+0)/1000.0}'
  elif [ "$u" = "GiB/s" ]; then awk -v v="$n" 'BEGIN{printf "%.3f", (v+0)}'
  else awk -v v="$n" 'BEGIN{printf "%.3f", (v+0)}'
  fi
}
bw_gibps="$(to_gibps "${bw_num:-0}" "${bw_unit:-MiB/s}")"
lat_line="$(echo "$out" | grep -E 'clat .*avg=' | head -n1 || true)"
lat_avg="$(echo "$lat_line" | sed -n 's/.*avg=\([0-9\.]*\).*/\1/p')"
echo "mode,bw_GiBps,iops,avg_latency_us" > "$OUT"
echo "${MODE},${bw_gibps:-0},${iops:-0},${lat_avg:-0}" >> "$OUT"
echo "$OUT"
