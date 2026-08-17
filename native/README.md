# PP-OCRv6 RKNN C++ Core

该目录保持 Rockchip Model Zoo 的 CMake / `rknn_api.h` / OpenCV 组织方式。核心库在 `src/` 与 `include/`，可执行入口在 `apps/`。

## RK3588 交叉编译

```bash
export RK3588_SDK_ROOT="$HOME/data/rk3588/aibox_3588"
cmake -S native -B build-rk3588 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=native/cmake/toolchains/rk3588-aarch64.cmake
cmake --build build-rk3588 -j"$(nproc)"
```

部署时需确保板端的 `librknnrt.so` 与模型兼容，并提供三个 `.rknn` 和 `ppocrv6_dict.txt`。命令行验证：

```bash
./ppocrv6_ocr --models /models --dict /models/ppocrv6_dict.txt \
  --input page.jpg --output result.json
```

`--disable-cls` 可跳过方向分类。

## 常驻 JSONL daemon

`ppocrv6_ocr_daemon` 启动时只初始化一次 Det、Cls、Rec，随后持续从标准输入读取 JSON Lines，并在标准输出返回一行结果。它不输出日志到 stdout，因此可由 Python、C++ 或 shell supervisor 通过管道稳定调用。

```bash
./ppocrv6_ocr_daemon --models /models --dict /models/ppocrv6_dict.txt
```

请求：

```json
{"id":"page-001","input":"/data/page-001.jpg"}
```

响应包含 `ok`、文本、检测/识别置信度、方向和四点坐标；单条请求错误只返回 `ok:false`，不会使 daemon 退出。
