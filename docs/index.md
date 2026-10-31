# 实验报告索引

当前入口为[项目首页](../README.md)。每份报告的指标、预算、种子与限制只适用于其对应协议；旧阶段的数字不能直接当成新阶段的配对比较。

| 主题 | 方法与结果 |
| --- | --- |
| 通信基线 | [AWGN 与理论 BER](awgn.md)、[多径与导频](multipath.md)、[时域约束 LS](tapls.md) |
| 统计估计 | [先验失配与 LMMSE](lmmse.md)、[导频统计学习](learned.md)、[阶跃漂移](drift.md)、[随机相位及缓变](robustness.md) |
| 因果更新 | [触发更新](triggered.md)、[冻结泛化](generalization.md)、[支撑告警](guard.md)、[恢复](recovery.md) |
| 资源与跟踪 | [导频密度](density.md)、[协方差创新](innovation.md)、[延迟复核](followup.md)、[本机运行成本](runtime.md) |
| 同步与捕获 | [已知边界 CFO](cfo.md)、[未知起点与整数 CFO](acquisition.md)、[捕获后的统计跟踪](captured_tracking.md) |
| 弱首径审计 | [移动物理窗口及其反例](alignment.md)、[保留原窗的相位参考（最新）](reference.md) |
| 工程约定 | [实验规范](experiments.md)、[验证记录](validation.md)、[路线图](roadmap.md)、[阶段归档](history.md) |

## 文件完整性

v0.21 的 `reports/reference/metadata.json` 同时声明实验源码 / 配置与 CSV / 冻结协议的 SHA-256。使用 `python -m wavelink.audit --report reports/reference` 核验，失败时返回非零退出状态。命令只读文件，并限制校验路径在项目和报告目录内部。

旧报告未统一保存产物清单，因此不能期待它们全部通过这一新入口。哈希一致仅表明文件与当前清单匹配；清单本身也能被修改，所以它不替代签名、可信发布或独立复现。
