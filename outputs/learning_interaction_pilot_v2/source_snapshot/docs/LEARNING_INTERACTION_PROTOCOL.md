# 持久保存与未来参数学习的交互：冻结规程

日期：2026-10-18。在新的 source-dose cohort 任何结果分析前固定。代码入口为 `scripts/run_learning_ablation.py`，数据输入为 source-dose 的完整共同前缀。

## 目的

将 FOLD 的“持久巩固 vs 暂态携带”在当前原型中操作化：replace 把新候选写入共享位置；none 使共享位置为零，但补偿当前驻留局部项，两者在编码前的当前有效策略相同。之后自然淘汰时，replace 可继承新共享知识，none 只剩局部携带。

保存的普通收益已在前轮测试。本轮新增问题是：这种差值在未来继续 PPO 与不再执行梯度时是否不同。不能把总回报提高归因于新知识组织算法。

## 对照与运行

使用 source-dose 同一 W0、32个共同前缀，以及其 replace/none×两未来的有梯度分支。它们不依赖 old-source 剂量，相同状态 hash 的alias不作为新样本。

另从相同完整前缀恢复，执行 replace/none×两未来、各12288个真实步骤。保持相同任务日程、动作uniform配对机制、reset种子和LRU容量；允许不同策略生成不同轨迹。所有actor/critic梯度和优化器步骤禁用；保留真实动作、奖励和自然slot淘汰/先验继承。角色“无梯度”不等于不使用已学参数或不发生缓存替换。

完整保存rawsteps、episodes、rollouts、post-action/final checkpoint、模拟器状态及真实CPU/wall。无梯度分支PPO minibatch updates必须为0，fitted_steps不得增长；单列rollouts collected，不能计为trained。critic和W0参数在每个分支起止逐项核对。actor局部张量可以因LRU分配和既有先验继承而改变，不得把它们当梯度更新。

main新增32×4×12288=1572864真实环境步，prefix新成本0（引用原已计费前缀）；独立pilot引用source-dose pilot，新增4×3072=12288步。预训练、前缀和有梯度对照成本引用原账本，不重复计费；报告合并总成本。所有方法同actor形状和完整包上限，无梯度计算更便宜如实报告，不宣称计算匹配。

## 估计量与判读

S_train=R_replace,train−R_none,train；S_frozen=R_replace,no-gradient−R_none,no-gradient，R为完整未来每步奖励。

核心新估计量 T=S_train−S_frozen，按 relation 以及 future regime报告，另给主条件history=future及全未来平衡平均。先同seed聚合历史，再做10000次seed-cluster bootstrap；每个实际seed仍为独立单位，分支不是独立重复。

不预设 T>0 是idea成立的必要条件：

- T明显正：保存在这个设置下与未来参数学习存在正交互。
- T明显负：未来学习缩小了保存初始化差距；不支持正协同主张，但不能据此否定学习、样本效率或持久知识价值。
- T接近0：当前证据更符合可加或未辨明的关系，不能宣称协同。

实际量级参考±0.002 reward/step。仅当描述区间完全在+0.002以上或−0.002以下才称有明确实际量级的交互；区间覆盖范围较宽则为不确定。逐seed差、前256/512步和每bout曲线保留，用于解释初始化与重学，不据其挑新主指标。

这些是一次持久写入、已知组、固定小型任务族的因果交互对照，不是AMAGO/FOLD完整复现、未知边界持续巩固、自动分组或SOTA算法比较。编码漂移需与原post-action日志一起解释。

## 费用、失败与同步

任何中断或软件失败的实际rawsteps都计入；未知成本保持未知，不记成0。源码、协议、配置、来源hash保存，原数据不重写。完成后与其他未同步研究内容一起推送GitHub，机器私有规则和临时原件提取继续排除。
