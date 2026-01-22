#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-gds_probe.csv}"
gpu_count="0"
driver="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1 || echo 0)"
  driver="$(nvidia-smi --query --display=INFO 2>/dev/null | grep -i 'Driver Version' | awk -F': ' '{print $2}' || echo unknown)"
fi
kernel_drv="no"
if [ -r /proc/driver/nvidia-fs/version ] || lsmod | grep -q '^nvidia_fs'; then kernel_drv="yes"; fi
user_lib="no"
if command -v ldconfig >/dev/null 2>&1; then
  if ldconfig -p 2>/dev/null | grep -q 'libcufile.so'; then user_lib="yes"; fi
fi
cfg_present="no"
if [ -f /etc/cufile.json ]; then cfg_present="yes"; fi
echo "gpu_count,driver_version,gds_kernel_driver,gds_user_lib,cufile_json_present" > "$OUT"
echo "${gpu_count},${driver},${kernel_drv},${user_lib},${cfg_present}" >> "$OUT"
echo "$OUT"
