# 🏦 JPM Chooser Option Pricing Dashboard
> **基于 BSM 与深度学习残差修正的时序定价分析工具**

本工具是一个专门针对 JPMorgan (JPM) 期权设计的交互式定价与压力测试平台。它对比了传统的 **Rubinstein (1991)** 选择权解析解与现代 **GRU、LSTM、Hybrid** 等机器学习模型的性能差异。

---

## 🌟 核心亮点
- **精度对齐**：GRU 结合对数残差修正，在测试集上实现 $R^2 \approx 0.57$。
- **动态寻优**：内置验证集 T1 决策点寻优引擎。
- **压力推演**：支持 VIX 飙升及基准利率骤增的黑天鹅场景模拟。
- **新手导览**：内置步进式交互向导，降低金融模型的使用门槛。

## 模型解说
| 模型名称 | 核心逻辑 | 适用场景 |
| :--- | :--- | :--- |
| **Paper_Original_BSM** | 经典的 Rubinstein 解析解 | 基准线对比 |
| **GRU (Proposed)** | 对数空间下的残差预测 + 偏差校准 | 捕获非理性溢价 |
| **LSTM** | 双向长短期记忆网络 (Bi-LSTM) | 捕捉长周期波动趋势 |
| **Hybrid** | 随机森林预测波动率 + 梯度提升树定价 | 复杂波动率环境 |
| **E2E** | 纯数据驱动的端到端定价 | 快速预测 |

## 🚀 快速开始
1. **克隆项目**：
   ```bash
   git clone [https://github.com/你的用户名/JPM-Option-Pricing.git](https://github.com/你的用户名/JPM-Option-Pricing.git)
