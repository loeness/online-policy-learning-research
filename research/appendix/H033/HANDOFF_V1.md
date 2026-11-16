# H033 v1 · 算法实现交接

交付日期：2026-10-21。问题：固定未来分布下，操作前信息是否支持替换旧知识的净价值排序。用户已选择 H033 最小诊断；H036 继续保留设计候选。

交付的是可核验方法实现，尚无 H033 科学结果。主控负责接受方法、分配 E、冻结推进/停止判据和离线资源上限；消融实验（1）负责独立比较审查、执行、统计和证据。本包不改变共享 H 状态、E161 停止决定或预算，也没有向其他聊天发送消息。

## 可审查版本

- 唯一方法语义：[METHOD_V1.md](METHOD_V1.md)。三个特征、精确教师与 KL、候选原拟合、new-bound 语义、目标、四折、固定 ridge 和四个简单对照均已确定。
- 冻结源包：`D:/MovedFromC/ChatGPT/online/research/appendix/H033/delivery_v1/source_snapshot`。
- `delivery_v1/manifest.json`：完整父 Git SHA、逐文件 SHA256、源包身份、输入绝对路径及哈希、运行环境；`tracked_changes.patch` 保留允许清单内的已跟踪脏改动。新文件以完整快照为准，HEAD 单独不标识本版。
- `delivery_v1/verification.json` 与 `checks.txt`：合成正确性检查及验证范围。图表：NOT_APPLICABLE，本轮不产生科学结果图。原始科学数据仍在 E161，不拷贝进本源包。

本次独占新增方法/源码/config/tests 与 H033 appendix。E161 source、原始输入/结果、在线策略及其他研究记录均受保护。实际运行必须用冻结副本；如审查发现错误，返回算法实现并另建版本，不能在实验执行中悄悄修改语义。

## 三阶段接口

下列命令以本机实际 D: checkout 为准。`expNNN` 是**唯一待主控分配的管理占位**，不对应已获批准的实验。运行前由实验负责人把它替换成登记的 E 目录并创建 run001；方法参数没有待猜测项。不得沿用 E161 的远端预算或自动上服务器。

```powershell
$h033Python = 'D:/MovedFromC/ChatGPT/online/.venv/Scripts/python.exe'
$h033Source = 'D:/MovedFromC/ChatGPT/online/research/appendix/H033/delivery_v1/source_snapshot'
$h033Run = 'D:/MovedFromC/ChatGPT/online/experiments/expNNN/runs/run001'
$h033Producer = 'D:/MovedFromC/ChatGPT/online/experiments/exp161/runs/run001/producer'
$h033Analysis = 'D:/MovedFromC/ChatGPT/online/experiments/exp161/analyses/analysis001/analysis.json'
& $h033Python "$h033Source/scripts/h033_diagnostic.py" features --producer $h033Producer --output "$h033Run/features.json"
& $h033Python "$h033Source/scripts/h033_diagnostic.py" outcomes --analysis $h033Analysis --output "$h033Run/outcomes.json"
& $h033Python "$h033Source/scripts/h033_diagnostic.py" crossfit --features "$h033Run/features.json" --outcomes "$h033Run/outcomes.json" --output "$h033Run/crossfit.json"
```

每条命令检查成功后才进入下一条；输出已存在则拒绝覆盖。features 只读允许的三个 prefix 文件，不接收标签路径；一 seed 一次候选重构。outcomes 单独抽取 192 个所需标签单元。crossfit 输出 16 个留出决策、4 个折模型、5 种规则的配对值以及描述均值，均带可追溯输入身份/CPU/wall。新增环境交互为 0，候选拟合和离线模型计算仍计入本轮成本。

包内计时是程序阶段内部成本；实验负责人仍需从外部核算启动、依赖加载、失败/重试及总墙钟，不能用它冒充完整 campaign 成本。候选 C 在当前本机从允许状态重建；冻结包记录本机依赖与平台，不承诺浮点结果逐位等于原服务器拟合 C。它是本合同定义的操作前代理量；如核对原候选哈希，仅可作独立审计，不能把 observer 值补入特征。

结果接口详细字段见源码 docstring 和方法合同。`provenance` 是审计字段，不是预测器输入；程序只能验证输入身份/schema 与计算时序，不能凭 schema 证明人工提供的数字从未被造假。真实提取必须走冻结 adapter，并与 manifest 的输入哈希核对。

## 接下来严格按这六步执行

1. **主控接收合同并登记。** 在新 E 协议中明确本轮是 E161 结果后探索复用，设定离线 CPU/wall 上限、与简单规则比较的实际价值门槛及停止条件。此处不由算法程序猜测阈值。
2. **消融实验（1）独立审查。** 验证源包/输入哈希；核对新、旧教师、new-bound 规则及 16 seed 四层分折。尤其检查旧 teacher 为 local=0 的 keep 先验，sidecar owner 代次正确，observer 信息没有进入特征。审查不通过则退回实现责任方。
3. **只提取操作前量。** 得到 16 行 features，检查每行来源哈希、允许数据量、候选有限性、文件未变及实际成本。任何 seed 无效则停止整批并保留失败，不删除坏 seed 或补特征。
4. **一次性生成标签和留出预测。** 验证每 seed 主目标 4 个配对单元，balanced 只按同一决策计值；生成固定四折输出。不按结果换折、调 α、加特征或补种子。
5. **独立统计选择价值。** 以完整 seed 的配对增量评价 crossfit 相对四个简单规则，报告选择频率、个体差值及预先接受的不确定性分析；总体 MSE 不作推进证据。四折模型训练集重叠，不能把它们或 192 个分支标签当独立重复。统计方式在看本轮预测结果前由实验负责人固定。
6. **主控做下一决策。** 如果达到事先固定的选择价值门槛，只支持安排未见 seed 的后续确认设计，不直接升级完整流；若无法超越简单规则或只能改善 MSE，则停止该最小选择器路线。负结果只约束本特征/模型，不能证明所有操作前信息都无效。转向 H036 仍需另立受控合同与资源授权。

本包的技术检查不占用或扩大真实实验授权；实际 H033 的 E、预算和科学推进判据尚待主控登记，不能把实现交付写成 H033 已验证。
