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
python examples/noiseless_loopback.py
python -m experiments.replay_reference
```

该流程只读版本控制内的报告，不覆盖它们。任一子步骤失败会返回非零状态。它覆盖 194 项自动测试、两份报告的源码 / 产物清单、12,800 bit 无噪声回环，以及四段突发接收实验。

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
