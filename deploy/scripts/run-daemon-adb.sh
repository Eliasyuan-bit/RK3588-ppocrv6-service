#!/usr/bin/env bash
set -euo pipefail

remote_root="${1:-/userdata/ppocrv6-rknn-service}"
adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
  adb_args=(-s "$ADB_SERIAL")
fi

echo "Starting ppocrv6_ocr_daemon on the board. Send JSONL requests to this terminal; Ctrl-C stops it." >&2
exec adb "${adb_args[@]}" shell \
  "$remote_root/bin/ppocrv6_ocr_daemon --models '$remote_root/models' --dict '$remote_root/models/ppocrv6_dict.txt'"
