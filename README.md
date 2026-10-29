# WaveLink

面向多径与移动信道的 OFDM 接收机：建模、同步与鲁棒信道估计。

Reproducible wireless-link experiments for propagation-aware receiver design.

**状态：开发初期 / 项目实践。** 已完成 QPSK/OFDM 基础模块、复基带 AWGN 与理论 BER 对照，含循环前缀能量约定、独立重复及置信区间；现已加入多径、梳状导频、保护子载波、LS/时域 LS/LMMSE 与 ZF 均衡，并完成长度、噪声与 PDP 失配的基线比较。当前结果限于指定仿真模型，不包含硬件实测。

## 研究问题

当多径时延、移动条件和噪声统计发生变化时，接收机如何退化？低复杂度信道估计与跟踪方法能否改善可靠性、导频开销与计算成本之间的权衡？

第一条研究主线为信道统计失配。先建立可信基线，再比较 LS、DFT 去噪与 LMMSE；改进是否有效以可复现实验为准。

## 快速开始

要求 Python 3.11+。在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/noiseless_loopback.py
```

Windows 使用 `.venv\Scripts\activate` 激活环境。

示例使用固定随机种子，发送 100 个 OFDM 符号、12,800 比特；无噪声、理想同步回环应得到 0 个比特错误。这只是基础正确性检查，不代表实际无线链路性能。

## 当前约定

- SISO 复基带离散时间模型；QPSK 平均符号能量为 1。
- IFFT/FFT 使用 `norm="ortho"`；子载波采用 NumPy 原生 FFT 顺序。
- AWGN 基线仍使用全数据子载波；多径实验采用独立资源栅格，包含 DC/保护子载波和导频。两者均未加入信道编码。
- 循环前缀（CP）长度按样本计。无噪声多径测试使用已知信道和理想符号边界。
- 传播模型不等于 Maxwell 方程全波求解；仿真数据不作为实测数据。

## 目录

```text
src/wavelink/       基带信号处理模块
tests/             能量、映射、回环与多径边界验证
examples/          可运行的最小示例
docs/roadmap.md     阶段范围与验收标准
docs/experiments.md 实验记录与公平比较约定
```

## 后续里程碑

1. **已完成**：AWGN 基线与理论 BER 对照，明确 SNR / Eb/N0 及开销定义。
2. **已完成**：静态多径、导频、LS 信道估计与均衡；同步误差和补偿尚待实现。
3. 信道统计失配下的算法对比、独立重复实验及失败边界。
4. 根据结果延伸到调制识别、SDR 回放/接收、无线边缘任务调度。

详见 [路线图](docs/roadmap.md) 和 [实验规范](docs/experiments.md)。

## 参考入口

- [GNU Radio: Basic OFDM Tutorial](https://wiki.gnuradio.org/index.php/Basic_OFDM_Tutorial)
- [NVIDIA Sionna: Neural Receiver](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Neural_Receiver.html)

上述资料作为参考，当前代码为基础模块实现，不宣称复现其完整系统。后续引入外部代码或数据时单独记录来源、版本及许可。

## AWGN 基线（已验证）

```bash
python -m pip install -e '.[plots]' -c experiments/requirements.txt
python experiments/awgn_ber.py --config experiments/awgn.json --output results/awgn --plot
```

默认进行 30 次固定预算仿真，10 个汇总点，每点 3,072,000 比特。`useful` 排除 CP 能量；`transmitted` 包含期望 CP 能量。两种约定须分别与各自理论曲线比较。

![AWGN BER](reports/awgn/ber.svg)

[方法、实际结果与局限](docs/awgn.md) · [汇总数据](reports/awgn/summary.csv) · [逐种子数据](reports/awgn/runs.csv) · [来源与环境](reports/awgn/metadata.json)

## 静态多径与导频接收（已验证）

```bash
python experiments/multipath_ber.py --output results/multipath --plot
```

比较不均衡、理想 CSI + ZF、导频 LS + 线性插值 + ZF。两种固定信道、两种 CP、两种导频密度、四个信噪比、三组种子；接收机共享发送数据、信道和噪声。21 项测试通过，CP 足够时的 16 个理想 CSI 汇总点通过理论回归检查。

![Multipath benchmark](reports/multipath/multipath.svg)

[多径实验报告与边界](docs/multipath.md) · [结果](reports/multipath/summary.csv) · [无噪声诊断](reports/multipath/diagnostics.csv)

已知 CSI 只作为参考；无噪声插值仍可能有偏差。该阶段为静态离散多径仿真，尚未验证移动信道、频偏或硬件接收。

## 时域结构约束与长度失配（已验证）

```bash
python -m experiments.tapls_ber --output results/tapls --plot
```

采用部分 DFT 矩阵对导频观测做时域抽头 LS 拟合；比较长度 L=2/4/8/13 与线性插值。CP 固定为 16，避免把 CP 不足和长度先验错误混在一起。所有接收机共享样本，不依赖真实信道选取 L。

![Tap-domain LS](reports/tapls/tapls.svg)

[公式、先验边界与结果](docs/tapls.md) · [汇总数据](reports/tapls/summary.csv) · [矩阵条件数与噪声放大](reports/tapls/geometry.csv)

27 项测试通过；64/64 偏差-方差一致性检查和 16/16 oracle BER 检查通过。时域约束并非总是更好：过短模型产生偏差，过长模型可能放大噪声。该实现是已有 LS 方法的可复现比较，不宣称新算法。

## LMMSE 与统计信息失配（已验证）

```bash
python -m experiments.lmmse_ber --output results/lmmse --plot
```

在每帧独立的随机块衰落中，对比线性 LS、时域 LS、准确统计信息的 LMMSE，以及噪声功率低估/高估 10 倍、PDP 均匀化或错误缩短的 LMMSE。共享链路样本，报告期望之比的 ensemble NMSE、BER 和帧级/配对标准误。

![LMMSE statistical mismatch](reports/lmmse/lmmse.svg)

[模型、理论校验与实际结果](docs/lmmse.md) · [汇总数据](reports/lmmse/summary.csv) · [信号/噪声误差分解](reports/lmmse/predictions.csv)

33 项测试、112/112 估计器 NMSE 检查和 16/16 Oracle BER 检查通过。统计信息匹配可以改善均方误差，但错误 PDP 或噪声假设可使 LMMSE 劣于 LS。准确 PDP 是直接提供的理想统计基线，尚未从接收数据估计；本阶段随机信道与前阶段固定信道的 BER 不直接比较。
