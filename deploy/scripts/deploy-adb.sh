#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${1:-}"
remote_root="${2:-/userdata/ppocrv6-rknn-service}"
stage_dir="${STAGE_DIR:-$repo_root/dist/rk3588}"

if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
  echo "Usage: ADB_SERIAL=<serial> $0 <model-dir> [remote-root]" >&2
  exit 2
fi
"$repo_root/scripts/check-rk3588-package.sh" "$stage_dir"

required_models=(
  ppocrv6_det_480x480_logits_fp.rknn
  ppocrv6_cls_48x192_logits_fp.rknn
  ppocrv6_rec_48x320_logits_fp.rknn
  ppocrv6_dict.txt
)
for model in "${required_models[@]}"; do
  [[ -f "$model_dir/$model" ]] || { echo "error: missing model: $model_dir/$model" >&2; exit 1; }
done

adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
  adb_args=(-s "$ADB_SERIAL")
fi
adb "${adb_args[@]}" shell "mkdir -p '$remote_root/bin' '$remote_root/models' '$remote_root/input' '$remote_root/output'"
adb "${adb_args[@]}" push "$stage_dir/bin/." "$remote_root/bin/"
for model in "${required_models[@]}"; do
  adb "${adb_args[@]}" push "$model_dir/$model" "$remote_root/models/$model"
done
adb "${adb_args[@]}" push \
  "$repo_root/tests/fixtures/ocr_smoke_page_001.png" \
  "$remote_root/input/ocr_smoke_page_001.png"
adb "${adb_args[@]}" shell "chmod 0755 '$remote_root/bin/ppocrv6_ocr' '$remote_root/bin/ppocrv6_ocr_daemon'; test -r /usr/lib/librknnrt.so"
adb "${adb_args[@]}" shell "ldd '$remote_root/bin/ppocrv6_ocr'"
echo "deployed to $remote_root"
