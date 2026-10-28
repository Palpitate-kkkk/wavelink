# WaveLink

面向多径与移动信道的 OFDM 接收机：建模、同步与鲁棒信道估计。

Reproducible wireless-link experiments for propagation-aware receiver design.

**状态：开发初期 / 项目实践。** 已完成 QPSK/OFDM 基础模块、复基带 AWGN 与理论 BER 对照，含循环前缀能量约定、独立重复及置信区间。尚无研究性能提升结论，不包含硬件实测。

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
- 当前全部子载波承载数据，未加入 DC/保护子载波、导频或信道编码。
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
2. 多径、导频、LS 信道估计与均衡；分别引入同步误差和补偿。
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
