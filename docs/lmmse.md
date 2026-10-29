# LMMSE 信道估计与统计信息失配

本阶段研究：当接收机使用的噪声功率或功率时延谱（PDP, power delay profile）不准确时，LMMSE 如何退化？实现既有线性估计器，并用可计算的理论误差检查完整 OFDM 仿真，不宣称算法创新。

## 模型与上一阶段的区别

上一阶段 tapls 使用两个确定性静态信道。本阶段使用随机块衰落：每帧独立生成零均值、不相关复高斯抽头，帧内 21 个 OFDM 符号保持固定，其中第一个为预热符号，20 个计分。抽头满足

\[
h\sim\mathcal{CN}(0,R),\quad R=\operatorname{diag}(p),\quad \sum_l p_l=1.
\]

每帧不再将抽头强制归一化。否则会改变既定分布，使 `diag(p)` 不再准确描述协方差。样本间深衰落被保留，Eb/N0 按平均发送能量定义，不按每帧接收功率重设噪声。因此本报告的 BER 与前一阶段的 BER 不能直接横向比较。

- 64 点 OFDM，Fs=1 MHz，激活子载波为 -26..-1 和 1..26，DC 与边缘保留。
- CP=16 样本，真实最大时延 12 样本，理想同步；不混入 CP 不足、频偏或多普勒。
- 导频间隔为 2/4，分别为 28/16 个导频、24/36 个数据子载波。导频和数据均为单位能量 QPSK。
- 稀疏 PDP：时延 0/4/12 样本的平均功率为 0.80/0.12/0.08；其余为零。
- 指数 PDP：连续 13 个抽头，`p[l] ∝ exp(-l/3)`，归一化总平均功率为 1。
- 每个估计器逐 OFDM 符号工作，未利用帧内信道不变性做时间平均。跨帧突变为统计试验的块衰落模型，尚不代表连续运动信道。

总发送能量约定沿用前阶段：`Eb = Na/(2 Nd) × (N+CP)/N`，复噪声方差 `N0 = Eb / 10^(EbN0_dB/10)`。酉 FFT 后每个导频的噪声方差仍为 N0。

## 估计器与可用先验

令单位导频相除后的观测为 `z=Fp h+n`，待估计的数据子载波为 `Hd=Fd h`。Fp/Fd 使用未归一化 DFT 指数项，符合物理 FIR 的频率响应；OFDM 调制本身仍为酉 FFT/IFFT。

采用接收机假设的协方差 Ra 与噪声方差 Na：

\[
\widehat H_d=Az,\qquad
A=F_dR_aF_p^H(F_pR_aF_p^H+N_aI)^{-1}.
\]

代码用线性方程求解，未显式形成逆矩阵。允许 PDP 中存在零功率；正噪声方差提供正则化。输入必须为有限的非负实数 PDP、正的假设噪声方差及单位模导频；不会自动归一化接收机提供的 PDP。

| 名称 | 接收机获得的信息 |
|---|---|
| Oracle CSI + ZF | 真实瞬时 Hd，仅作为参考 |
| Linear LS | 导频位置与已知导频，实虚部线性插值 |
| Tap LS | 连续支撑 0..12、已知导频，不需要 PDP 或 N0 |
| LMMSE matched | 真实总体 PDP 和 N0，作为理想统计基线；未知瞬时抽头 |
| LMMSE noise ×0.1/×10 | 真实总体 PDP，噪声分别低估/高估 10 倍 |
| LMMSE uniform | 假设 0..12 均匀平均功率，总功率 1；真实 N0 |
| LMMSE short | 错误假设仅 0..3 均匀平均功率，总功率 1；真实 N0 |

噪声失配与 PDP 失配分别改变一个因素。所有方法共享同一信道、发送数据、导频和噪声。matched PDP 从试验分布直接提供，尚未实现从数据中学习 PDP 或估计 N0，因此不能视为部署性能。

均匀 PDP 的时域形式等价于 ridge 正则化：`Fd (FpᴴFp + N0 L I)^(-1) Fpᴴ`。matched 方法的最优性针对该统计模型下的均方误差与线性估计器，不直接保证每次有限样本的 BER 最小。

## 理论校验与统计口径

本报告的指标为“期望之比”：

\[
\mathrm{NMSE}_{ens}=\frac{E\|\widehat H_d-H_d\|^2}{E\|H_d\|^2}
=\frac{\|(AF_p-F_d)R^{1/2}\|_F^2+N_0\|A\|_F^2}{N_d\sum_lp_l}.
\]

这里用真实 R 与真实 N0 进行评价，即使构造 A 时使用了错误先验也适用。分子分为信号重建误差和噪声传播误差，记录于 `predictions.csv`。信号项包含正则化造成的收缩误差，不能一概解释成固定未知信道的偏差平方。

仿真对每帧所有数据子载波、所有计分符号的平方误差取平均。由于总体平均信道功率为 1，该平均值就是上述指标的估计。它不等于“逐帧除以瞬时信道功率后再平均”的 NMSE，也不应与 tapls 的条件 NMSE 混用。

每个配置有 3 个固定种子，每个种子 120 个独立帧，共 360 个帧。帧内共享衰落会关联比特，标准误（SE）以帧为独立单位计算。`paired_nmse_minus_matched` 与 `paired_nmse_se` 记录同帧配对的误差差值及其 SE，正值表示估计误差高于 matched 基线。

NMSE 回归检查为 `|观测-理论| <= 6 × 帧级SE + 1e-10`；这是预先固定的工程一致性检查，不是多重比较校正后的显著性检验。Oracle 的条件 BER 为各数据子载波 `0.5 erfc(|H|/sqrt(2N0))` 的平均；使用每帧“实测 BER 减条件理论 BER”的均值与帧 SE 检查。没有根据结果重新选择种子或提前停止。

## 本次结果

配置：`experiments/lmmse.json`，2 个 PDP × 2 个导频间隔 × 4 个 Eb/N0 × 3 个种子。每条物理链路比较 8 种接收机，共 384 行逐种子接收机结果、128 个汇总点。共计 5,760 个独立衰落帧、6,912,000 个不同的计分发送比特；接收机共享的比特不重复算入数据量。

下表是稀疏 PDP、18 dB、导频间隔 4、CP=16 的结果，每种方法计分 518,400 比特：

| 接收机 | 错误数 | BER | ensemble NMSE |
|---|---:|---:|---:|
| Oracle CSI | 3964 | 0.007647 | 0 |
| Linear LS | 41957 | 0.080936 | 0.164302 |
| Tap LS, L=13 | 7518 | 0.014502 | 0.013493 |
| LMMSE matched | 4723 | 0.009111 | 0.002873 |
| LMMSE noise ×0.1 | 4741 | 0.009145 | 0.002891 |
| LMMSE noise ×10 | 4979 | 0.009605 | 0.004195 |
| LMMSE uniform | 7204 | 0.013897 | 0.012209 |
| LMMSE short | 50148 | 0.096736 | 0.197435 |

matched 的预测 NMSE 为 0.00284751，观测为 0.00287318。在这个配置下，Tap LS 相对 matched 的配对 NMSE 差为 `0.0106203 ± 0.0000463`（±1 SE）；uniform 为 `0.00933559 ± 0.0000408`；short 为 `0.194562 ± 0.00738070`。这些误差差值支持先验对估计质量的影响，但此处没有对 BER 做配对显著性结论。

均匀 PDP 提供一定正则化，但未获得稀疏位置的准确知识，收益明显小于 matched。错误 short PDP 丢失真实晚到路径，18 dB 仍有很高重建误差。噪声错估 10 倍也会改变收缩强度；在指数 PDP、6 dB 下，matched NMSE=0.124065，高估噪声 10 倍为 0.305737，甚至高于 Tap LS 的 0.215169。失配的 LMMSE 不保证优于 LS。

112/112 个非 Oracle 估计器 NMSE 检查、16/16 Oracle BER 检查通过；另有 16 个 Oracle 的 NMSE 为零检查通过。33 项自动测试通过，包含后验协方差交叉验证、均匀 PDP/ridge 等价、低噪声 LS 极限、错误先验 Monte Carlo、无真值估计接口及小规模复现。

![LMMSE statistical mismatch](../reports/lmmse/lmmse.svg)

## 复现与局限

在仓库根目录：

```bash
python -m pip install -e '.[plots]' -c experiments/requirements.txt
python -m unittest discover -s tests -v
python -m experiments.lmmse_ber --output results/lmmse --plot
```

[逐种子结果](../reports/lmmse/runs.csv)、[汇总与配对 SE](../reports/lmmse/summary.csv)、[理论分解](../reports/lmmse/predictions.csv)、[配置及源码哈希](../reports/lmmse/metadata.json)均已保存。metadata 的 `base_commit` 是生成结果前的父提交，dirty 标记表示当时尚未提交；源码 SHA-256 用于核对本次运行的具体代码。结果和代码一起提交后可检出该提交复现数值；平台、计时及 SVG 元数据可能变化。

预计算主要为导频协方差矩阵求解，约 O(Np³)，每符号映射约 O(Nd Np)；尚未做运行时间基准，不能据此宣称低延迟或低复杂度优势。当前仅覆盖不相关零均值高斯抽头、整数延迟、单天线 QPSK、理想同步和无编码链路。尚未覆盖相关抽头、Rician 均值、连续时变、分数延迟、测得的协方差、实际噪声估计及硬件实测。下一步适合在相同评测规则下加入仅使用接收数据的噪声/PDP估计，再衡量理想先验收益能保留多少。

## 参考

- [NXP / Freescale AN3059: Channel Estimation in OFDM Systems](https://www.nxp.com/docs/en/application-note/AN3059.pdf)：LS/MMSE 与信道二阶统计的基础参考。
- [NVIDIA Sionna: LMMSEChannelEstimator](https://nvlabs.github.io/sionna/phy/api/ofdm/sionna.phy.ofdm.LMMSEChannelEstimator.html)：LMMSE 估计与协方差先验的官方实现文档。仅用于方法核对，本项目未引入其依赖，也未声称复现其完整接收机。
