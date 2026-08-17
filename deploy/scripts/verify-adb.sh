#!/usr/bin/env bash
set -euo pipefail

image_path="${1:-}"
remote_root="${2:-/userdata/ppocrv6-rknn-service}"
if [[ -z "$image_path" ]]; then image_path="$remote_root/input/ocr_smoke_page_001.png"; fi

adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
  adb_args=(-s "$ADB_SERIAL")
fi
prefix="$remote_root/bin/ppocrv6_ocr --models $remote_root/models --dict $remote_root/models/ppocrv6_dict.txt"
adb "${adb_args[@]}" shell "$prefix --input '$image_path' --output '$remote_root/output/verify.json'"
adb "${adb_args[@]}" shell "cat '$remote_root/output/verify.json'"

printf '%s\n' \
  "{\"id\":\"verify-1\",\"input\":\"$image_path\"}" \
  "{\"id\":\"verify-2\",\"input\":\"$image_path\"}" \
  | adb "${adb_args[@]}" shell "$remote_root/bin/ppocrv6_ocr_daemon --models '$remote_root/models' --dict '$remote_root/models/ppocrv6_dict.txt'"
