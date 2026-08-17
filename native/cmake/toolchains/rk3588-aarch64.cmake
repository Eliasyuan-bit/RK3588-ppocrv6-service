# Usage:
# cmake -S native -B build-rk3588 \
#   -DCMAKE_TOOLCHAIN_FILE=native/cmake/toolchains/rk3588-aarch64.cmake \
#   -DRK3588_SDK_ROOT=/path/to/aibox_3588

if(NOT DEFINED RK3588_SDK_ROOT)
  set(RK3588_SDK_ROOT "$ENV{RK3588_SDK_ROOT}")
endif()
if(NOT RK3588_SDK_ROOT)
  message(FATAL_ERROR "Set RK3588_SDK_ROOT to the aibox_3588 SDK root")
endif()
# CMake re-evaluates toolchain files for compiler ABI checks; forward this
# user-provided SDK root into those internal try-compile projects as well.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RK3588_SDK_ROOT)

set(_toolchain_root
  "${RK3588_SDK_ROOT}/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu")
# Pair the compiler with its own sysroot and with the static OpenCV shipped by
# the RKNN examples.  Do not use the Debian rootfs OpenCV here: it was built
# with a newer host ABI than this GCC 10.3 toolchain.
set(_sysroot "${_toolchain_root}/aarch64-none-linux-gnu/libc")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER "${_toolchain_root}/bin/aarch64-none-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${_toolchain_root}/bin/aarch64-none-linux-gnu-g++")
set(CMAKE_SYSROOT "${_sysroot}")

set(CMAKE_FIND_ROOT_PATH "${_sysroot};${RK3588_SDK_ROOT}/debian/binary")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Use the RKNN SDK artifacts, not the host headers/runtime.
set(RKNN_API_INCLUDE_DIR
  "${RK3588_SDK_ROOT}/external/rknpu2/runtime/Linux/librknn_api/include" CACHE PATH "")
set(RKNN_RT_LIBRARY
  "${RK3588_SDK_ROOT}/external/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" CACHE FILEPATH "")
set(OpenCV_DIR
  "${RK3588_SDK_ROOT}/external/rknpu2/examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV" CACHE PATH "")
