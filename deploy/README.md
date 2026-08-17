# RK3588 部署

部署脚本假定已经执行过仓库根目录的 `./scripts/build-rk3588.sh`。模型不进入 Git，由调用者提供本地模型目录。

## 运行目录

`deploy-adb.sh` 默认在板端创建如下目录（可用第二个参数覆盖）：

```text
/userdata/ppocrv6-rknn-service/
├── bin/
│   ├── ppocrv6_ocr
│   ├── ppocrv6_ocr_daemon
│   └── libppocrv6_rknn_core.so
├── models/
│   ├── ppocrv6_det_480x480_logits_fp.rknn
│   ├── ppocrv6_cls_48x192_logits_fp.rknn
│   ├── ppocrv6_rec_48x320_logits_fp.rknn
│   └── ppocrv6_dict.txt
├── input/
└── output/
```

运行包不复制 `librknnrt.so`，以板端 `/usr/lib/librknnrt.so` 为准，避免 Runtime 与 NPU 驱动不匹配。

## ADB 部署

```bash
export ADB_SERIAL=172.16.15.232:5555
./deploy/scripts/deploy-adb.sh /path/to/ppocrv6-rknn-models
```

模型目录必须恰好包含 README 中列出的四个文件。脚本会在推送前检查，并在板端检查动态链接依赖。

脚本还会将仓库内的测试 PNG 放入 `<remote>/input/`；该文件不参与模型加载。

## 验收

```bash
./deploy/scripts/verify-adb.sh
```

该脚本默认使用仓库附带的页面 PNG；也可用 `./deploy/scripts/verify-adb.sh /board/path.jpg` 覆盖。它先执行单图 OCR，保存 `<remote>/output/verify.json`，再运行 daemon 的两条连续 JSONL 请求，确认同进程协议可用。

## 前台启动 daemon

```bash
./deploy/scripts/run-daemon-adb.sh
```

它会直接把板端 daemon 的 JSONL 标准输入/输出连接到当前终端。手工调试时可输入：

```json
{"id":"manual-1","input":"/userdata/ppocrv6-rknn-service/input/ocr_smoke_page_001.png"}
```

集成时，上层程序应直接在板端启动 `bin/ppocrv6_ocr_daemon` 并长时间持有其 stdin/stdout，而不是针对每张图片重新启动进程。
