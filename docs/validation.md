# 初始本地验证

日期：2026-10-28（Asia/Shanghai）

环境：Python 3.13.5，NumPy 2.4.1。以下为本地源码路径运行结果；尚未验证干净虚拟环境安装，尚未配置远程 CI。

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 examples/noiseless_loopback.py
```

结果：6 项测试通过。

1. 已知 QPSK 映射与硬判决。
2. 单位符号能量与酉变换能量守恒（不计 CP）。
3. CP 长度为 0、16、64 时的无噪声回环。
4. CP 足够长时，线性多径卷积与频域乘法一致，理想 CSI 均衡可恢复比特。
5. CP 不足时，频域循环卷积模型失效。
6. 非法比特、CP 参数、残缺符号与非有限值的拒绝。

示例实际输出：

```text
Noiseless simulation | bits=12800 | bit_errors=0 | BER=0
```

这里的零误码仅对应无噪声仿真。AWGN 理论对照、移动信道、同步算法和统计性能结论均待后续实现。
