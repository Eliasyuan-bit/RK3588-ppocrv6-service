# RK3588 Benchmark Records

## 2026-08-17 — PP-OCRv6 FP/logits

测试方式：目标板执行 `rknn_benchmark`；每项为单次模型调用的平均值，不包括图像 resize、Det DBPostProcess、CPU Sigmoid/Softmax、CTC decode、内存拷贝和服务框架开销。

运行环境：

```text
rknn_api / rknnrt: 2.4.2a2 (2026-07-31)
NPU driver:         0.9.8
```

| 模型 | 输入 / 输出 | 平均延迟 | 平均吞吐 |
|---|---|---:|---:|
| `ppocrv6_cls_48x192_logits_fp.rknn` | `1x48x192x3` → `1x2` | 0.97 ms | 1031.672 FPS |
| `ppocrv6_det_480x480_logits_fp.rknn` | `1x480x480x3` → `1x1x480x480` | 30.53 ms | 32.756 FPS |
| `ppocrv6_rec_48x320_logits_fp.rknn` | `1x48x320x3` → `1x40x18710` | 18.55 ms | 53.898 FPS |

### NPU core-mask 实测（同一目标板）

core mask 的位定义为 `1=Core 0`、`2=Core 1`、`4=Core 2`、`7=Core 0+1+2`。各单核表现基本一致；三核掩码对 Det 有明显收益，但不能提升小型 Cls，Rec 的收益也很小。

| 模型 | 单核最佳 | 三核（mask 7） | 结论 |
|---|---:|---:|---|
| Det 480×480 | 30.52 ms | 21.38 ms | 三核快 29.9%，适合低延迟 Det。 |
| Cls 48×192 | 0.97 ms | 1.35 ms | 三核更慢，固定单核。 |
| Rec 48×320 | 18.59 ms | 18.10 ms | 仅快 2.6%，固定单核并行多个文本框更合适。 |

注意：

- Cls/Rec 的数据是**每个文本框**一次调用；一页文档的总延迟约为 `Det + N * (Cls + Rec) + CPU 后处理`。
- Rec 输出约 1.43 MiB（FP16），CPU Softmax 与 CTC decode 是端到端性能评估的重点。
- Det 与 Rec 均输出 logits，需按模型目录说明在 CPU 完成概率与解码后处理。
