# SigMF 导入与量化一致性（v0.25）

本阶段在固定时隙 I/Q 回放前增加标准文件适配层。字段与 I/Q 交错顺序参照 [SigMF 官方规范 v1.2.6](https://sigmf.org/)，实现一个明确受限的 1.2.x 子集；不是通用 SigMF 验证器，也没有接入硬件录制数据。

## 支持范围与输入约定

| 项目 | 本模块的支持范围 |
| --- | --- |
| 文件对 | 同名 `.sigmf-meta` 与 `.sigmf-data`，元数据按 UTF-8 读取 |
| 数据类型 | `cf32_le`：每分量 little-endian float32；`ci16_le`：每分量 little-endian signed int16，均为 I/Q 交错 |
| 录制布局 | 单通道，零数据索引偏移，空 captures 或一个从索引 0 开始的连续 capture |
| 采样率 | 必须显式提供有限正的 `core:sample_rate`；用于单位换算，不进行重采样 |
| 时隙选择 | 用户给出 `--first-sample`（默认 0）和 `--slots`；一个时隙 1,984 个复样本 |
| 已知导频 | 另给 `.npy` 数组，形状 `(slots,20,28)`，严格单位模；不是从未知数据或标注推导 |
| 可用性标记 | 可选 Boolean `.npy`，形状 `(slots,)`；省略表示所有选定时隙均有输入 |
| 完整性 | 核对已有的 `core:sha512`；另记录输入、转换输出和实现的 SHA-256 |

有符号整数必须显式给出 `--integer-scale`，其单位是“每整数计数对应的输出幅度”。例如 `1/32768` 可以是用户选择的归一化约定，但 **SigMF 并不替你确定真实 ADC 增益或物理电压**。浮点输入保持原幅度，不接受整数缩放参数。

不支持大端、实数、多通道、多 capture、非零索引偏移、非符合数据布局、嵌入头部字节或必需的未知扩展。这些情况明确报错，不猜测数据布局。注释内容不参与选帧或解码，仅记录忽略的注释条数。

`--first-sample` 的单位是复样本，不是字节或单个 I / Q 分量。从这个位置选取完整的 `slots×1984` 个复样本，并记录未选的前后样本数；请求超过文件长度时拒绝转换。选定范围必须无 NaN / Inf，文件必须包含完整的 I/Q 对。

## 使用方法

安装项目后，在仓库根目录先生成受控演示文件。目标目录必须不存在：

```bash
python -m experiments.sigmf_demo --output results/sigmf_demo_001
```

该命令保留三个编码案例及转换 / 解调结果。以浮点案例单独执行导入与核验：

```bash
python -m wavelink.sigmf_import convert \
  --metadata results/sigmf_demo_001/float32/capture.sigmf-meta \
  --pilots results/sigmf_demo_001/original/pilots.npy \
  --available results/sigmf_demo_001/original/available.npy \
  --first-sample 7 --slots 96 \
  --output results/sigmf_converted_001
python -m wavelink.sigmf_import verify \
  --metadata results/sigmf_demo_001/float32/capture.sigmf-meta \
  --pilots results/sigmf_demo_001/original/pilots.npy \
  --available results/sigmf_demo_001/original/available.npy \
  --output results/sigmf_converted_001
python -m wavelink.iq_replay run \
  --recording results/sigmf_converted_001 \
  --output results/sigmf_decoded_001
python -m wavelink.iq_replay verify \
  --recording results/sigmf_converted_001 \
  --output results/sigmf_decoded_001
```

整数案例需要把 `--metadata` 改成对应文件，并给 `--integer-scale` 一个已知的换算系数；演示的系数保存在其 `converted/import.json` 和汇总 CSV 中。不要把某个演示系数当成所有录制设备的系数。

转换目录含 v0.24 的四个回放输入文件，以及 `import.json`，记录源文件指纹、参数、所选范围、未选前后长度、端点分量数和当前实现。`verify` 使用显式传入的原文件核对转换链；`iq_replay verify` 再核对转换后的录制与接收输出。两步提供文件一致性，不提供签名或物理真实性认证。

## 三种受控编码的结果

三种编码使用同一个 96 时隙合成夹具，包含初始统计启动、缺失、拒收、超范围 CFO 和后续统计变化；不是三个独立性能试验。文件各增加 7 个前置、5 个后置复样本，再明确选择中间的 96 个完整时隙。

| 编码案例 | 采样误差 NMSE | 接收路径相对原始数据的判决 / 擦除位置变化 | 输出判决中的错误比特 | 编码器已知削顶分量 |
| --- | ---: | ---: | ---: | ---: |
| float32 | 6.43×10⁻¹⁶ | 0 | 5,421 | 0 |
| int16，无削顶 | 5.06×10⁻⁹ | 12 | 5,419 | 0 |
| int16，故意削顶 | 0.04348 | 1,280 | 5,619 | 20,703 |

采样误差 NMSE 是 `Σ|量化采样−原采样|² / Σ|原采样|²`，不是信道 NMSE 或 BER。每种编码仍有 91 个时隙输出硬判决、1 个拒收、4 个输入缺失；输出数量不能代表正确接收。载荷错误由独立评分器计算，导入器和接收机不读取真实载荷。

无削顶整数的量化步长约 0.0001635583，实测最大复采样误差 0.0001154331，满足每分量四舍五入的复误差上界 `sqrt(2)×步长/2`。单个夹具中错误数少 2 个不构成量化改善性能的证据。

削顶案例中，导入器观测到 **20,707** 个位于整数端点的分量，而编码器掌握的实际削顶数为 **20,703**。端点值也可能自然产生，因此端点数量不是未知模拟信号削顶数量的认证结果。

所有编码的转换采样均与编码器直接解包得到的期望值逐元素相同；每条文件接收路径的判决与事件，也与直接输入同一量化数据的路径完全一致。这里的“一致”针对同一种量化表示，不声称量化前后等价。

## 验证与当前限制

本机总计 **212 项测试通过**。新增 9 项覆盖：独立 `struct.pack` 字节夹具、I/Q 顺序和样本偏移、整数端点及显式缩放、奇数字节 / 请求越界、未支持布局拒绝、SHA-512 和可用性校验、UTF-8 注释忽略、输入 / 输出篡改、非有限值 / 导频验证，以及三种量化表示的完整回放一致性。

`make check` 已包含临时 SigMF 夹具生成、转换、两段指纹链核验及接收比较。临时文件结束后清理，Git 只保存源码和小型汇总记录。脚本处理所选样本和输出数组时会占用随选择范围增长的内存，不作无限流、吞吐或实时性保证。

[汇总记录](../reports/sigmf/runs.csv) · [源码、环境和转换链指纹](../reports/sigmf/metadata.json) · [导入模块](../src/wavelink/sigmf_import.py) · [演示与评分器](../experiments/sigmf_demo.py)

真实数据还必须匹配本项目训练头、导频、采样率和时隙安排。未知录制仍需前端下变频、增益标定、重采样和连续流包检测等工作；本接口不承诺解码任意 SigMF 录制。硬件验证仍待独立完成。
