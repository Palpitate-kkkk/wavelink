# WaveLink

**面向信道统计失配与非理想同步的 OFDM 接收机实验平台。**

电磁场与无线技术 / 通信方向的项目实践。以可复现的仿真比较 LS、LMMSE 与因果统计跟踪，记录有效条件、失败边界和资源成本。当前版本 **v0.21**，所有性能数字均为软件仿真结果。

## 研究问题与已实现能力

当多径功率分布、频偏和接收起点偏离先验时，接收机会怎样退化？只利用当前及历史导频的统计学习，能否在相同发送预算下改善接收损失？

- **通信基线**：QPSK、酉 OFDM、循环前缀、复基带 AWGN 与理论 BER 对照；多径、保护子载波、梳状导频、LS / 时域 LS / LMMSE 和 ZF 均衡。
- **统计学习与跟踪**：独立校准、因果滑窗更新、漂移触发、协方差创新监测、延迟复核，以及模型支撑失配的告警与恢复实验。
- **非理想接收**：有限时隙的未知包起点与整数 / 分数 CFO 捕获；捕获缺失、统计门限、物理时隙与观测双时钟。
- **证据与工程**：冻结配置、独立种子、配对标准误、Oracle 对照、反例、更新成本和本机耗时审计；新报告增加源码及产物 SHA-256 校验入口。

这里的整数 / 分数 CFO 指相对子载波间隔的恒定频偏，并非连续分数时延模型。

## 快速运行

需要 Python 3.11+，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/noiseless_loopback.py
python -m wavelink.audit --root . --report reports/reference
```

Windows 激活命令为 `.venv\Scripts\activate`。无噪声示例应得到 0 个比特错误；这仅验证基础回环。校验命令检查已保存结果与所声明的源码 / 文件哈希是否一致，不重新运行实验，也不证明来源真实性。

## 最新研究：保持原窗的统计相位参考

v0.20 发现：向显著训练抽头移动 FFT 窗口可能跳过弱首径。v0.21 保留捕获器给出的物理窗口，只对统计接收机输入作频域相位变换，估计后换回原参考，再均衡原始数据。

该设计避免了**由这次窗口移动新增**的物理定时风险，原捕获错误与变换后的模型失配仍可能存在。它并不保证找到了真实首径，也不保证损失优于每个对照。

[最新方法、结果及局限](docs/reference.md) · [全部报告索引](docs/index.md) · [路线图](docs/roadmap.md)

重新运行本阶段的冻结比较：

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.reference_ber --config experiments/reference.json --output results/reference
python -m wavelink.audit --root . --report results/reference
```

线程环境变量用于控制数值库线程数，运行时间取决于本机环境。224 个场景 / SNR / 种子组合共享样本比较 8 个接收方法；参数继承前一阶段，不用本轮结果调参。脚本运行前后检查源码与协议未改变；冻结历史源文件被修改时会拒绝运行。

## 如何审阅

1. 从 [AWGN 基线](docs/awgn.md)、[多径接收](docs/multipath.md)检查通信与能量约定。
2. 通过 [LMMSE](docs/lmmse.md)、[统计学习](docs/learned.md)、[冻结泛化审计](docs/generalization.md)了解第一条研究主线。
3. 阅读 [突发捕获](docs/acquisition.md)、[捕获后的统计跟踪](docs/captured_tracking.md)、[物理窗口对齐反例](docs/alignment.md)和[最新相位参考实验](docs/reference.md)。
4. 在各报告对应的 `reports/` 中复核逐种子记录、汇总、事件、配置与来源。指标、样本单位和成本以各阶段协议为准，不能直接混合比较。

## 模型边界

目前是 SISO、未编码 QPSK 的离散时间复基带模型；64 点酉 FFT、16 点 CP 是最新突发实验的固定配置。多径在单个突发内保持不变，跨突发改变统计分布。当前不包含真实多普勒、任意连续流多包检测、SDR 实测、MIMO 或完整 5G 协议栈，也不是 Maxwell 方程全波求解。

本机耗时报告不构成硬实时保证。低 SNR、超捕获范围、弱首径和变换后的非因果支撑仍是已记录的局限，完整问题尚未解决。

## 目录与复现约定

| 目录 | 内容 |
| --- | --- |
| `src/wavelink/` | 信号处理、估计、捕获与跟踪模块 |
| `tests/` | 数学性质、因果性、边界与回归验证 |
| `experiments/` | 冻结协议、独立种子及实验脚本 |
| `reports/` | 提交保存的逐次结果、汇总与来源 |
| `docs/` | 方法、结果、反例与局限 |
| `examples/` | 最小回环示例 |

[实验规范](docs/experiments.md) · [验证记录](docs/validation.md) · [阶段记录归档](docs/history.md)

项目适合按“项目实践 / 科研实践”展示。仅将实际实现和已经验证的结果写入材料；外场测量、实时部署及未实现功能不列为成果。
