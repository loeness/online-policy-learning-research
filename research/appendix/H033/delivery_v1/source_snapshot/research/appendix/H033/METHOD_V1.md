# H033 最小排序诊断 · 方法合同 v1

日期：2026-10-21。归属：算法实现。问题沿用 H033 的 E161 结果后细化；用户已选择本方向的规格与实现交付。本合同先于本版程序实现固定，不是 E161 预注册，也不把已见结果变成独立确认。未分配新 E、未批准真实诊断运行预算；主控保留问题接受、实验编号、推进条件及状态写入权。

## 1. 唯一问题与目标

在固定的未来分布下，三个操作前量是否足以支持比简单规则更好的 **replace_behavioral / keep** 排序？不预测评估器将随机安排哪个未来，不改在线 actor、保存算法、绑定、编码或优化器。

每个独立 seed 只有一行输入。令 R 为 6144 次未来交互的 reward/step：

`y_s = (1/4) Σ_{f∈{old_major,new_major}, r∈{0,1}} [R(s,f,r,replace_behavioral) − R(s,f,r,keep)]`。

这是指定两个完整日程各占 1/2 的目标，不是任意复现比例的阈值。balanced 不参加目标、训练、标准化或规则选择；同一留出决策在 balanced 上另行计值。replace_mean 不进入本诊断。不能把每条未来分支当成一个独立样本。

## 2. 输入时序和三个固定特征

操作前只读每个 prefix 的 `checkpoint.pt`、`actor.bin`、`evidence.npz`；固定策略 temperature=2、epsilon=.05。checkpoint 中 W0、source_basis、local/shared、parent binding、cache、generation、每组至多 256 行 evidence、最近至多 8 个 group 均已存在。sidecar 是同一保存包内两个当前新来源各最新 128 行的有界观测；其 owner/slot/generation 必须与当前 resident 对应。禁止从 observer、simulator、原始完整历史、旧 source 矩阵档案或未来输出构造特征。

观测记 x；W0 actor 只用 x 的前 4 维，route 及其余 9 维不参与本版概率重建。策略计算沿用 H026 的 `FrozenPolicyFeatures`、temperature-softmax 和 epsilon 混合。令 p_C 为 local=0、parent=C 的策略；p_0 是 local=0、无 parent 的 W0；p_P 为 local=0、当前旧父 P 的策略。P 必须绑定旧组。新 teacher p_{j,i} 取当前第 j 个 resident 的有效矩阵，并按观测第一维选 band。

1. `recent_frequency_contrast = [count(new_group) − count(old_group)] / len(recent_groups)`。历史必须非空、有界；不使用 seed/family 标签替代它。
2. `new_fit_gain = D_new(0) − D_new(C)`，其中 `D_new(C) = .5 Σ_{j=0,1} mean_{i∈j} KL(p_{j,i} || p_C(x_i))`。这里用两个 owner 的全部允许观测作操作前诊断均值；每个 owner 总权重相等。C 由相同前置状态和 sidecar 重新调用原 H026 拟合器确定：每 owner 最新 128 行，前 75% 训练、后 25% 诊断，ridge=1e-4，L-BFGS 64 iter / 256 eval；原选择/回退规则不变。不能按未来回报选择 C。C 是 float32 **编码前**候选，此量不是编码后收益预测保证。
3. `old_prior_loss = mean_{i∈old evidence} KL(p_P(x_i) || p_0(x_i))`。旧 evidence 取 checkpoint 旧组全部现存观测（至多 256 行），逐行等权。保护对象是旧任务重新分配、local 清零时的 keep 初始化先验；不是已淘汰旧来源的完整策略，也不是 evidence 中采样时的 behavior_probs。

三量均直接使用，无对数/裁剪/特征搜索。使用自然对数；数值下 KL 的极小负舍入误差仅在 KL 单行计算处截到 0。缺少输入、旧绑定不匹配、owner/generation 不一致或非有限数值时拒绝该整批，不删 seed、不补值、不尝试替代特征。

**C 的原拟合语义：** `M=mean_{owner,band}(E_new)` 以 float64 求均值，初始化为 float32(M) 后转 float64 优化；目标是训练行 owner 等权 forward KL 加 `1e-4 * mean((C−M)^2)`。L-BFGS 使用 lr=1、history_size=10、strong_wolfe、tolerance_grad=1e-9、tolerance_change=1e-12，最多 64 iter / 256 closure eval（初末额外目标计算不计入 closure）。从初始化开始，只接收目标不高于已见最优、且训练 KL 不高于初始 KL+1e-12 的点；相等允许后点替换。预算耗尽返回已见最优；最终转 float32 后训练 KL 若超过初始+1e-12，回到 float32 均值。holdout 不作选择。其他异常报错，不发明新的候选回退。本版只允许两个 owner 各 128 行，因此不足支持的历史分支不应触发。

**绑定解释：** E161 replace 绑定新组，旧组新槽因而没有父先验；keep 对新组新槽也没有旧父先验。因此上述两项分别量化“新初始化可能改善”和“旧初始化失去的行为”。不能用 C 在旧观测上的 KL 表示实际 replace 旧初始化。驻留补偿、淘汰后继承、codec、actor Adam 清理和 critic 保留均沿用原 E161；本程序不安装任何候选，不改 checkpoint。

## 3. 唯一预测器和留出规则

固定 16 个 E161 seeds（配置列出），以 `(old_family, history_regime)` 四个 strata 分层；每层恰 4 seeds。每层按 seed 升序，第 k 个分到第 k 折，得到 4 折，每折 train=12 / test=4。family 和 history 只作分层元数据，seed/文件路径/未来日程均不是模型特征。

每折仅用训练 12 行计算各列均值 μ、总体标准差 σ（ddof=0）；σ<1e-12 时设为 1。Z=(X−μ)/σ，截距 b=mean(y_train)。固定 ridge α=1：

`β = solve(Z_trainᵀ Z_train + I, Z_trainᵀ (y_train−b))`。

目标为 `Σ_train (y−b−Zβ)^2 + ||β||²`；截距不正则。不调参，不用 balanced 或测试标签选择模型。留出预测严格 `ŷ>0` 才 replace，否则 keep，含并列 keep。没有模型失败后的静默选择器回退；无效批次报错。

对照固定为：always keep、always replace、训练集 mean(y)>0 的 intercept-only、`recent_frequency_contrast>0` 的频率规则。总共一个可训练模型、四个简单对照；不增加其他拟合器或扫参。

## 4. 数据接口和职责边界

`h033_features_v1`：rows 每行 seed、old_family、history_regime、三个具名 features、输入哈希与数据行/候选诊断 provenance。metadata 标记 pre-action extraction、CPU/wall、输入限制。

`h033_outcomes_v1`：rows 每行 seed、future_regime、replica、arm、horizon=6144、reward_per_step；每 seed 恰 3 futures × 2 replicas × 2 arms。标签从 E161 已封存 analysis001 的 seed-level arm rows 提取，单独成包，不能进入特征提取器。adapter 对原报告按明确字段过滤其他 horizon/arm；拒绝所需标签缺失、重复、不完整。

`h033_crossfit_v1`：每折 train/test seed、标准化参数、系数、截距；每 seed 预测增量、真实配对增量、五种决策及 primary/balanced 下的所选值、相对 keep 增量。聚合仅作描述均值；无 p 值、置信区间或自动推进结论。正式统计、成功/停止条件及协议由消融实验（1）独立审查并由主控接受，程序交付不替代它们。

三步入口为 `scripts/h033_diagnostic.py features`、`outcomes`、`crossfit`。每次指定绝对输入/输出路径，输出文件必须不存在。features 不接受 outcome 路径；crossfit 严格验证完整 seed 配对。具体可复制命令、源包身份见同目录交接文件。

## 5. 成本、公平与限制

离线诊断复用既有付费全分支标签，不声称部署学习可获得未选动作反馈。新增环境交互=0；每 seed 至多重新拟合一次原候选，记录实际 CPU/wall。新模型只有 3 个 slope + 1 截距，尚未集成部署；部署需要的标准化参数、训练标签来源、反馈机制与完整持久字节均未获验证，不能宣称同预算在线保存器已改进。

特征不增加在线缓冲；读取已存在有界 evidence，不复制旧原始日志或旧 source 教师。所有策略对照使用相同的旧实验结果，不产生新增不公平训练预算。拟合及输出仅为诊断临时成本。E161 只有 16 个 prefix，四折训练高度重叠，正结果最多提供可行性信号；负结果只约束这三个特征与这个固定线性模型，不能证明所有过去信息都无用。

主控需要在运行前单独确定实际执行 E、离线 CPU/wall 上限和与简单规则比较的推进/停止标准。原 E161 剩余预算不能沿用。本轮只做合成正确性检查，不运行真实数据的特征拟合、交叉拟合或新训练。

## 6. 实现归属和最小检查

本版独占新增：`src/online_rl/h033_diagnostic.py`（数值核心）、`src/online_rl/h033_inputs.py`（有界 prefix 与原报告 adapter）、`scripts/h033_diagnostic.py`（三步入口）、`configs/h033_diagnostic_v1.json`、`tests/test_h033_diagnostic.py`、`tests/test_h033_inputs.py`、`tests/test_h033_cli.py` 和本 H033 appendix 交付文件。已完成 E161、其他研究记录、共享索引/状态/预算、在线策略代码均不修改。

必要检查：完整 seed 留出；测试标签/特征不能改变对应折训练与标准化；balanced 不能进入主目标；常量列与并列行为；重复/缺失/非有限输入拒绝；旧先验 local=0、route 中性、new-bound 解释正确；两个 teacher owner 等权；输入不变；原拟合器重建使用相同 frozen float32 W0 特征。测试仅用合成夹具；技术 PASS 不建立 H033 科学结论。
