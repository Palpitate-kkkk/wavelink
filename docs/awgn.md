# AWGN 基线：理论、实验与边界

## 已实现范围

SISO、未编码 Gray QPSK、64 子载波全部承载数据、16 点 CP、酉 FFT/IFFT、理想同步。噪声加入串行时域样本（含 CP），接收端去 CP 再做 FFT。此处不包含导频、频偏、多径衰落或硬件实测。

## 噪声与能量约定

复噪声 n=sqrt(N0/2)(a+jb)，a、b 独立服从 N(0,1)，所以 E|n|²=N0，各实维方差 N0/2。噪声按声明的期望能量产生，不按每帧实测功率重新缩放。

QPSK Es=1，每符号 2 比特，令 gamma=10^(EbN0_dB/10)，alpha=(N+CP)/N=1.25。

| 约定 | Eb | 复噪声方差 N0 | 接收端有效 Eb/N0 | 样本 SNR（线性） |
|---|---|---|---|---|
| useful（不计 CP） | 1/2 | 1/(2 gamma) | gamma | 2 gamma |
| transmitted（计入期望 CP 能量） | alpha/2 | alpha/(2 gamma) | gamma/alpha | 2 gamma/alpha |

Es/N0=1/N0。本模型的期望时域样本功率为 1，因此样本 SNR 与 Es/N0 相同。这个关系只适用于当前全载波、单位能量、酉变换模型；以后加入空载波、导频、编码、过采样时必须重新推导。

理论 BER = 0.5 erfc(sqrt(gamma_effective))。同一目标 BER 下，计入 CP 的横轴需要增加 10 log10(1.25)=0.9691 dB。CP 被丢弃不会在固定 N0 下额外改变 BER；差异来自横轴能量定义。

参考：[MathWorks 的 QPSK AWGN 解析公式](https://www.mathworks.com/help/comm/ug/analytical-expressions-used-in-berawgn-function-and-bit-error-rate-analysis-app.html)。公式直接用 Python math.erfc 计算，不调用仿真结果拟合理论线。

## 实验设计

配置：`experiments/awgn.json`。0、2、4、6、8 dB；三组种子；每组每点固定 1,024,000 比特。每个汇总点 3,072,000 比特，共处理 30,720,000 比特。固定预算停止，不按错误数提前停止，避免将固定样本置信区间误用于可选停止。

不同种子与 SNR 点采用独立 SeedSequence；两种能量约定共享随机数以减小比较方差，因此两条曲线不是独立样本。逐次计数见 runs.csv。图中误差棒为逐点 95% Wilson 区间。理论一致性检查使用 Bonferroni 校正的 Wilson 区间，目标为近似 99% 家族覆盖；Wilson 是近似区间，不宣称严格有限样本保证，也不代表证明实现绝对正确。不会因单点未落入普通 95% 区间就换种子。

## 实际结果

| useful Eb/N0 (dB) | 错误数 / 3,072,000 | 仿真 BER | 理论 BER |
|---|---:|---:|---:|
| 0 | 241388 | 0.07857682 | 0.07864960 |
| 2 | 115398 | 0.03756445 | 0.03750613 |
| 4 | 38587 | 0.01256087 | 0.01250082 |
| 6 | 7320 | 0.00238281 | 0.00238829 |
| 8 | 576 | 0.00018750 | 0.00019091 |

两种约定的 10/10 汇总点均通过预定义的一致性检查。8 dB useful BER 的 95% Wilson 区间为 [0.00017280, 0.00020345]。未出现错误的情况仍须报告正的统计上界，不能声称真实 BER=0。

这说明当前基础链路与 AWGN 理论在所测范围内一致，不是新算法相对传统方法的性能提升。

## 复现与来源

```bash
python -m pip install -e '.[plots]' -c experiments/requirements.txt
python -m unittest discover -s tests -v
python experiments/awgn_ber.py --output results/awgn --plot
```

`reports/awgn` 为已提交的参考结果；默认新运行写入被忽略的 `results/awgn`，避免覆盖参考数据。metadata.json 记录基础 commit、运行时工作区是否有变动、源文件 SHA256、配置和 Python/NumPy 环境。源文件哈希明确标识尚未提交时的实际运行代码；基础 commit 本身不代表已包含本阶段实现。

12 项测试覆盖原有链路、复噪声均值/方差/实虚相关/滞后相关、FFT 后噪声功率、能量换算、固定种子重复性、统计区间边界与小规模 BER 回归。统计测试是合理性检查，不是高斯性或独立性的完整证明。

干净 `.venv` 已完成安装验证：Python 3.13.5、NumPy 2.4.1、Matplotlib 3.10.8；无需设置 PYTHONPATH 即可运行 12 项测试、完整 30 次仿真和无噪声示例。完整实验与初次运行的计数一致。
