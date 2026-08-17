# 技术说明与构建教程

## 从源码构建到板端运行

### 前置条件

- x86_64 Linux 构建机；
- RK3588 aibox SDK，包含 aarch64 GCC、RKNN Runtime 头文件和 aarch64 OpenCV；
- 已连接的 RK3588 板端，且板端存在兼容的 `/usr/lib/librknnrt.so`；
- 外部模型目录，包含四个文件：

```text
ppocrv6_det_480x480_logits_fp.rknn
ppocrv6_cls_48x192_logits_fp.rknn
ppocrv6_rec_48x320_logits_fp.rknn
ppocrv6_dict.txt
```

### 交叉编译

在仓库根目录执行：

```bash
export RK3588_SDK_ROOT="$HOME/data/rk3588"
./scripts/build-rk3588.sh
```

脚本会配置 CMake、交叉编译、安装到 `dist/rk3588/`，并检查以下运行包：

```text
dist/rk3588/bin/
├── ppocrv6_ocr
├── ppocrv6_ocr_daemon
└── libppocrv6_rknn_core.so
```

可用环境变量覆盖默认输出路径：

```bash
BUILD_DIR=/tmp/ppocr-build STAGE_DIR=/tmp/ppocr-package ./scripts/build-rk3588.sh
```

### ADB 部署

```bash
export ADB_SERIAL=< Your ADB devices >
./deploy/scripts/deploy-adb.sh < model_address >
```

默认部署位置为 `/userdata/ppocrv6-rknn-service/`；可用第二个参数指定其他位置：

```bash
./deploy/scripts/deploy-adb.sh /path/to/models /userdata/my-ppocr-service
```

部署后，板端目录为：

```text
<remote>/
├── bin/       # 程序和 libppocrv6_rknn_core.so
├── models/    # 三个 .rknn 模型和字典
├── input/     # 随仓库部署的测试 PNG
└── output/    # 验收输出
```

### 验收与启动

```bash
./deploy/scripts/verify-adb.sh
./deploy/scripts/run-daemon-adb.sh
```

第二条命令会以前台方式启动 daemon。随后在同一终端输入一行 JSON：

```json
{"id":"manual-1","input":"/userdata/ppocrv6-rknn-service/input/ocr_smoke_page_001.png"}
```

它返回一行 JSON 结果；按 `Ctrl-C` 停止。生产上由上层解析程序直接启动 `ppocrv6_ocr_daemon`，并持续持有其 stdin/stdout。

## 架构与 NPU 核心策略

当前稳定推理拓扑：

```text
上层解析服务（后续接入）
  -> ppocrv6_ocr_daemon (C++17, JSONL)
      -> Det: one RKNN context, Core 0+1+2
      -> Cls: one RKNN context, Core 0
      -> Rec: one RKNN context, Core 0
```

三个模型在 daemon 启动时各初始化一次。当前 Runtime 对同进程复制 FP16 PP-OCRv6 context 不稳定，因此默认不使用 `rknn_dup_context`。

## 模型与后处理

| 模型 | 输入 | NPU 输出 | CPU 后处理 |
|---|---|---|---|
| Det | 480×480 FP16 | `[1,1,480,480]` logits | Sigmoid、DB 后处理、坐标映射。 |
| Cls | 48×192 FP16 | 2 类 logits | argmax；类别 180 且置信度高时旋转文本框。 |
| Rec | 48×320 FP16 | `[1,40,18710]` logits | Softmax、CTC greedy decode。 |

Rec 字典使用基础字典加空格 token，并保留 CTC blank，因此与 18,710 类输出对齐。

Rec 预处理始终先将文本框高度缩放至 48；宽度超过 320 的文本框会按 320 像素、32 像素重叠的窗口分别识别。每个窗口的 CTC token 会按 time-step 映射回完整文本条带，并在相邻窗口重叠区的中点分配归属，不依赖 UTF-8 字符串前后缀匹配。空文本或平均识别置信度低于 0.50 的结果不会输出。

Det 会在 DB unclip 与坐标反变换后将四点框限制在原图边界内。页面任意边超过 `960px` 时，Det 以 `960×960`、`96px` 重叠切块执行；各块框映射回页面坐标后，先以边界框 IoU ≥ 0.50 去重，再将垂直对齐且明显水平重叠的跨块局部框拼为同一行，避免同一文字行被重复送入 Rec。小于该阈值的图片仍只执行一次 Det。

## JSONL daemon 协议

`ppocrv6_ocr_daemon` 从 UTF-8 标准输入读取 JSON Lines，并在标准输出返回一行响应。每行一个对象；标准输出不写日志。

请求：

```json
{"id":"page-001","input":"/data/page-001.jpg"}
```

成功响应：

```json
{"id":"page-001","ok":true,"texts":[{"text":"示例","rec_score":0.99,"det_score":0.85,"rotated_180":false,"box":[0,0,100,0,100,20,0,20]}]}
```

单条请求失败时，daemon 返回 `ok:false` 与 `error`，并继续处理后续请求。

```json
{"id":"page-001","ok":false,"error":"cannot open image"}
```
