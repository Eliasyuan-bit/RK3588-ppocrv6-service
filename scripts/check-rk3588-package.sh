#!/usr/bin/env bash
set -euo pipefail

stage_dir="${1:-}"
if [[ -z "$stage_dir" ]]; then
  echo "Usage: $0 <runtime-package-dir>" >&2
  exit 2
fi

for file in bin/ppocrv6_ocr bin/ppocrv6_ocr_daemon bin/libppocrv6_rknn_core.so; do
  if [[ ! -f "$stage_dir/$file" ]]; then
    echo "error: missing runtime artifact: $stage_dir/$file" >&2
    exit 1
  fi
done

file "$stage_dir/bin/ppocrv6_ocr" | grep -q 'aarch64' || {
  echo "error: ppocrv6_ocr is not an aarch64 ELF binary" >&2
  exit 1
}

echo "runtime package is complete: $stage_dir"
