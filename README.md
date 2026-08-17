# PP-OCRv6 RKNN Service

可独立构建、部署和运行的 RK3588 本地 PP-OCRv6 组件。它不依赖 Rust 或网络服务：C++17 daemon 在启动时各加载一次 Det / Cls / Rec，然后以 JSONL 标准输入/输出协议为上层进程提供 OCR。

```text
上层解析进程 ── JSONL stdin/stdout ──> ppocrv6_ocr_daemon ──> RKNN Runtime / NPU
```

## 仓库内容

| 目录 | 用途 |
|---|---|
| [native](native/README.md) | C++17 RKNN 核心、单图工具、JSONL daemon、RK3588 工具链文件。 |
| [scripts](scripts/) | 可复现的 RK3588 构建与运行包检查脚本。 |
| [deploy](deploy/README.md) | ADB 部署、板端验收与目录约定。 |
| [docs](docs/README.md) | 架构、JSONL 协议、模型与运行边界。 |
| [tests](tests/) | 小型测试夹具；模型不纳入 Git。 |

## 快速开始

### 1. 在 x86 构建机交叉编译

构建机需要 RK3588 aibox SDK（含 RKNN Runtime 头文件、aarch64 OpenCV 与 GCC 工具链）。

```bash
export RK3588_SDK_ROOT="$HOME/data/rk3588"
./scripts/build-rk3588.sh
```

产物固定写入 `dist/rk3588/bin/`：

```text
ppocrv6_ocr
ppocrv6_ocr_daemon
libppocrv6_rknn_core.so
```

### 2. 部署到 RK3588

模型目录必须含三个 FP/logits `.rknn` 文件和 `ppocrv6_dict.txt`。例如：

```bash
export ADB_SERIAL=<Your ADB device>
./deploy/scripts/deploy-adb.sh \
  <model_address> \
  /userdata/ppocrv6-rknn-service
```

脚本将运行包部署到 `<remote>/bin/`，模型部署到 `<remote>/models/`。板端只要求系统已有兼容的 `/usr/lib/librknnrt.so`。

### 3. 单图验收与启动 daemon

```bash
./deploy/scripts/verify-adb.sh

./deploy/scripts/run-daemon-adb.sh
```

daemon 是由上层解析进程持有 stdin/stdout 的常驻子进程，并非 HTTP 服务。生产中应由文档解析服务启动并保持这个进程；JSONL 协议见 [docs](docs/README.md)。

## 随仓库测试资产

`tests/fixtures/` 随仓库提交一个 160 DPI 的 PNG 页面图片。部署脚本会将它复制到板端 `input/`，而 `verify-adb.sh` 默认使用它做 OCR 冒烟测试。
