# 一次检查项目与复跑保存结果

## 推荐入口

在仓库根目录创建并激活 Python 3.11+ 环境、完成 `python -m pip install -e .` 后运行：

```bash
make check
```

Make 只是聚合已有 Python 命令；Windows 或没有 Make 时可逐条执行：

```bash
python -m unittest discover -s tests -v
python -m wavelink.audit --report reports/reference
python -m wavelink.audit --report reports/support
python -m wavelink.audit --report reports/crosscheck
python -m wavelink.audit --report reports/overview
python -m wavelink.audit --report reports/iq_replay
python -m experiments.iq_file_demo
python -m experiments.crosscheck --output results/crosscheck
python -m wavelink.audit --report results/crosscheck
python examples/noiseless_loopback.py
python -m experiments.replay_reference
```

该流程只读版本控制内的报告，不覆盖它们。任一子步骤失败会返回非零状态。它覆盖 203 项自动测试、五份报告的源码 / 产物清单、12,800 bit 无噪声回环，四段突发接收实验，以及独立标准库参考的数值交叉核对。

## 四段复跑具体检查什么

`experiments.replay_reference` 默认先校验 `reports/reference/` 的完整性，再执行以下保存案例，根种子均为 321001：

| 场景 | Eb/N0 | 目的 |
| --- | ---: | --- |
| stationary | 6 dB | 检查噪声、捕获拒绝与擦除计分 |
| weak_drift | 18 dB | 检查统计漂移与因果更新 |
| weak_first | 6 dB | 检查弱首径与低信噪比 |
| weaker_first | 18 dB | 检查更弱首径压力场景 |

每段均完整执行初始 64 + 计分 320 个物理时隙、8 个接收方法，并核对保存的完整 / 晚期记录。合计比较 64 行计分记录、24 行成本记录、87 行更新事件和 4 行窗口记录。错误数、擦除比特、摄入掩码哈希、启动时隙、更新事件和窗口哈希要求完全一致；浮点比例容差为绝对 1e-12。复跑种子用于确认保存结果，可以重复使用，不能被计为新的独立研究样本。

通过表示当前环境重现了这些保存案例，不表示全部 224 段已重跑，也不保证所有数值库 / 架构在决策阈值附近都得到相同结果。如果新平台得到差异，应保留错误字段和环境信息分析；不要为了显示通过而扩大误码计数或哈希的容差。

## 完整实验与结果写入位置

完整 v0.21 端到端比较与 v0.22 理想对角信道审计分别运行：

```bash
python -m experiments.reference_ber --output results/reference
python -m wavelink.audit --report results/reference
python -m experiments.support_audit --output results/support
python -m wavelink.audit --report results/support
```

这些命令在忽略的 `results/` 下产生新文件，不覆盖已提交的 `reports/`。支持范围与指标分别见[相位参考报告](reference.md)和[先验审计](support.md)。

## 本机安装验证

本阶段使用本机已有的 setuptools 84.1.0 / Python 3.12 构建标准可编辑 wheel，无需下载新依赖，再安装到项目的 Python 3.13.5 虚拟环境。安装后的包元数据确认为 0.22.0，导入路径指向本仓库 `src/wavelink/`；随后重新运行 `make check`。此验证不等于已验证所有平台或网络镜像下的首次安装。

## 来源边界

清单校验确认文件与声明哈希匹配；实际复跑进一步检查当前环境的计算输出。两者仍不替代可信签名、独立实现或硬件实验。旧阶段代码与协议故意冻结，新阶段通过新模块扩展；改动冻结文件会触发拒绝，需要明确建立新协议，不能悄悄重写旧结果。

## v0.23 独立实现与自动检查

本机无 MATLAB / Octave，本阶段改用标准库独立参考：`validation/reference_receiver.py` 不导入 NumPy 或 WaveLink，以直接 DFT、逐样本卷积和带主元消元计算同一固定夹具。8 个案例用于验证两套数值实现与硬判决一致，不作为新增 BER 性能证据。它是同语言不同实现，不能称为跨语言验证。

`.github/workflows/ci.yml` 在 push / pull request / 手动触发时安装项目并执行 `make check`，配置 Ubuntu 24.04、Python 3.11 / 3.13。实际成功与否以 [Actions](https://github.com/Palpitate-kkkk/wavelink/actions) 为准。动作版本依据官方 [checkout](https://github.com/actions/checkout) 与 [setup-python](https://github.com/actions/setup-python) 用法。

重生成总报告图需安装可选绘图依赖 `python -m pip install -e ".[plots]"`，再执行 `python tools/build_overview_figures.py`。默认自动检查仅核验保存的 SVG 及其输入，不安装绘图依赖或重跑完整 224 段实验。

首次 Linux CI 在旧 NMSE 保存值回归测试中暴露了约 5.6–6.7×10⁻¹⁶ 的平台舍入差异：原断言要求小数点后 15 位相同。该处现以 `rtol=1e-12, atol=1e-14` 比较浮点 NMSE；原误码数仍逐项严格相等，四段复跑的整数计数、事件、掩码与文件哈希也未放宽。没有修改旧实验数据或接收算法来通过测试。

## v0.24 文件回放检查

`make check` 现在包含临时合成文件生成、96 时隙文件 / 内存判决及事件比较，并在结束时清理临时二进制文件。报告记录的是受控文件接口一致性；实际外部文件的输入 / 输出校验请按 [I/Q 回放说明](iq_replay.md)使用专用 `run` / `verify` 命令。
