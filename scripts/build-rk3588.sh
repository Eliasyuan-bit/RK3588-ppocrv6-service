#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="${RK3588_SDK_ROOT:-}"
build_dir="${BUILD_DIR:-$repo_root/build-rk3588}"
stage_dir="${STAGE_DIR:-$repo_root/dist/rk3588}"

if [[ -z "$sdk_root" ]]; then
  echo "error: set RK3588_SDK_ROOT to the aibox_3588 SDK root" >&2
  exit 2
fi
if [[ ! -x "$sdk_root/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" ]]; then
  echo "error: RK3588_SDK_ROOT does not contain the expected aarch64 GCC toolchain: $sdk_root" >&2
  exit 2
fi

cmake -S "$repo_root/native" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$repo_root/native/cmake/toolchains/rk3588-aarch64.cmake" \
  -DRK3588_SDK_ROOT="$sdk_root" \
  -DCMAKE_INSTALL_PREFIX="$stage_dir"
cmake --build "$build_dir" -j"$(nproc)"
rm -rf "$stage_dir"
cmake --install "$build_dir"
"$repo_root/scripts/check-rk3588-package.sh" "$stage_dir"

echo "RK3588 runtime package: $stage_dir"
