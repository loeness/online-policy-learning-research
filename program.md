# 自动 RL 实验规程：阶段 A / B

此项目适配 autoresearch 的“提出候选—运行—测量—保留/拒绝”实验闭环。原版 nanochat 训练器不用于本项目；没有下载语言模型或文本训练数据。脚本自动运行已声明配置，代码修改由本会话 agent 完成并接受测试。

## 固定研究范围

固定源数据预训练得到基础策略，部署后 PPO 继续学习小策略修正。阶段 A 比较冻结、全参数微调、单修正；阶段 B 比较全局修正、无情境标签的历史路由、保留与重置修正。oracle 仅供诊断。

本轮没有验证共享父节点提升、压缩、容量最优性、任意 POMDP 或实际机器人部署。所有新修正在本轮为实验试用；策略改进可靠性通过结果分别报告，不声称已实现安全晋升门控。

## 不可改变的评测规则

- 隐藏环境编号、旋转值和切换时刻不得进入正式方法的 actor、critic 或路由输入。
- 初始数据仅来自 normal 源环境；部署旋转情境排除在预训练外。
- 所有方法读取相同初始策略与源数据。在线轨迹由各方法自己的动作产生。
- 外部切换按交互步计时，不发送通知；少数回合可以跨切换。累计结果包含这些回合，分情境分析单独标识。
- 验证 seed 901 用于自动配置搜索；主运行 seeds 11、23、37 不用于此轮选参。
- 选择指标在候选运行前固定：final CW success + 0.1 × mean online episode return。原始曲线、交互与成本一起保存；此标量仅选配置，不替代研究结论。
- 固定 rollout 256 步。持久序列化核心 + 数值 rollout 数组上限 256 KiB；运行时框架/Python 全部进程 RSS 另报，不声称 256 KiB 是进程总内存。
- 中止任何越过 agent 状态预算的运行。保留失败和负结果，不伪造结果。
- 小样本结果只作机制试验，不能声称统计确证；后续独立复现另行设置 seeds 与冻结配置。

## 运行顺序

1. `.venv/Scripts/python.exe -m pytest tests -q`（设置 PYTHONPATH=src）。
2. `.venv/Scripts/python.exe scripts/run_experiments.py prepare`。
3. `.venv/Scripts/python.exe scripts/autoresearch.py`。
4. `.venv/Scripts/python.exe scripts/run_experiments.py suite --config configs/selected.json --output outputs/pilot`。
5. 阅读原始 CSV、summary 和图。若存在实现 bug，记录修正并重跑受影响运行；若只是性能差，保留负结果。

## 本次执行记录与修订

2026-10-13：验证 seed 901 的三个配置全部保留，选出的配置写入 configs/selected.json。初轮 seeds 11、23、37 完成 24 次 A/B 运行，结果保留在 outputs/pilot。

初轮发现路由器会把切换期间的混合窗口作为新情境，在稳定尾部已经匹配已有节点时仍创建重复节点。修正为：创建前重新比较稳定尾部与已有原型，匹配时复用已有节点。增加 CW → CCW → CW 回归检查；补充随机状态容量核算和最后一轮 rollout 的优化器/重置版本记录。

使用新 seeds 41、53、67 与原先选定的超参数，完成另外 24 次运行，结果在 outputs/validated。没有根据主实验结果重新选择学习率或熵系数。由于初轮用于实现问题诊断，复验仍定位为机制验证，不作为预注册的统计确证研究。

容量只在 rollout/训练边界核查，RSS 也是检查点采样。已记录 evaluation_elapsed_s 包含评估和结果序列化，不能解释成纯评估计算时间。online_elapsed_s 单独计时。
