# H036 实现规格 v1

日期：2026-10-22。问题、范围与决策由 [主控合同](COORDINATOR_CONTRACT_20261022.md) 固定。本文解析实现接口，不修改主控合同、原提案或 E163 门槛。E163 仍未获得真实执行许可；本次仅做合成开发检查。

## 输入与时序

`prepare_inputs(prefix, *, config=None, expected_seed=None, expected_identity=None, logical_case=None)` 仅读 `checkpoint.pt`、`actor.bin`、`evidence.npz` 与固定配置。`expected_identity` 必须提供三个 basename 到冻结 SHA256 的完整映射，缺失或不符即拒绝；case 固定为 `s{seed}_dynamics_late`，可省略自动生成，显式提供时必须匹配。禁止用 observer、simulator、未来日程/回报、完整日志或已淘汰 source 的参数补充输入。真实 prefix 文件 SHA256、代次、行序和绑定须与冻结引用核对。

文件路径只记录放置位置，不进入 prepared_id 或拟合输入诊断哈希。相同文件与相同派生数组可迁移到 `/data/robot/p001/e163` 后保持身份；不同平台的派生数值身份在实际执行时另记，不预先保证浮点逐位一致。SSH、远端路径与依赖记录由实验方负责。

checkpoint 是可信项目的 `storage_decision_learner_v1`。当前旧父绑定组0、冻结，两个逻辑新来源均为组1，物理父标记均−1。完整 actor 必须为原 int8 89 B，解码 local/shared/groups 与 checkpoint 对应。source owner 与 slot/generation/route/时间顺序必须一致。旧 FIFO 观测、bout/task 的行数与顺序可核验，现有容量至多256，至少128行；不排序修复、不补行或删除 seed。

新 owner 各128行，分别前96训练、后32诊断。旧 FIFO 最新128行，同样96/32。旧教师为 prefix 中**实际解码 P_old、local=0** 的 keep 初始化 prior。新教师为各 resident 的冻结完整有效矩阵，按每行 band 选择。W0 actor 只依赖 obs[:4]，保持 temperature=2、epsilon=.05；旧 route 与已复用 slot residual 不混入旧教师。

返回 `h036_prepared_v1` plain dict，包括 frozen features、教师、明确 split 索引、local/shared/binding、完整 checkpoint 工作模板、原 actor/evidence bytes、输入身份与固定 foundation 身份。准备不调用 fitter、环境或 RNG，不把教师/临时对象放入最终 learner。`prepare_from_payload` 为相同语义的纯内存接口，供合成检查。

## 损失和优化

`D_new=.5 Σ_owner mean_train KL(T_owner || p_C)`，`D_old=mean_old_train KL(T_old || p_C)`。

`M=mean_{2owner×3band} E_new`，以float64求均值；`R(C)=1e-4 mean((C−M)^2)`。

`L_lambda=D_new+lambda D_old+R`，lambda仅为0/1。各项独立归一化，ridge只加一次。若复用原带ridge的两次objective，组合为 `L_new+lambda*(L_old−R)`，梯度同理；lambda0直接复用原新目标/梯度，避免多余舍入变化。

`fit_protected_candidate(prepared, lambda_value)` 返回 `h036_fit_v1`：float32 3×3 candidate、prepared_id、lambda和JSON诊断。C0=float32(M)后转float64优化。L-BFGS：lr1、history10、max_iter64、max_eval256、strong_wolfe、grad_tol1e-9、change_tol1e-12。

1. 先将 C0 与其完整 L_lambda 设为 incumbent；单独记录初始化计算。
2. 每次 closure 先检查剩余256次上限，再计数；计算完整目标/梯度、新旧训练KL，拒绝任何非有限输入、目标、梯度或概率，不将错误伪装成回退。
3. 新训练KL≤C0新训练KL+1e-12且完整L≤已见最优才接收；相等允许后点替换。逐次记录候选哈希、完整L/newKL/oldKL/ridge、接受与拒因。
4. 搜索按完整L执行；接收过滤不等于精确约束优化。closure预算耗尽合法返回已见最优，其他错误抛出。
5. 先记录获选float64点；转float32后**只**复查原新训练KL门槛，必要时回C0。不新增完整L或holdout择优。最终float32完整L仍记录。
6. 单独计初始化、closure、末端与诊断计算；报告 returned_is_C0、termination/fallback、次数和实际CPU/wall。holdout只报告，不驱动选择。

B/C共享同一个lambda0结果，不重复拟合；D为lambda1。只有32次真实拟合的完整计划可交实验方，合成测试输出不属于其证据。

## 安装、完整编码与 fresh-slot 评价

`prepare_install(local, shared, physical_groups, logical_tags, candidate, binding)` 和 `encode_and_score_stage0(prepared, fit_result, binding)` 归新输入/评价模块。binding只允许 `new`(parent_group1)或`wildcard`(−2)。C/D使用wildcard。高层mean/merge会平均旧父与候选，禁止替代此操作。

评分顺序固定：

1. 在工作副本调用原 `compensated_single_parent_install`，保持编码前 resident 有效矩阵；用 atol=rtol=1e-6核验。
2. 对完整2×27 local、1×9 shared、2个physical tags调用 `encode_actor(..., codec='int8', budget_bytes=89)`，再 `decode_actor`，不单独量化parent。
3. 在仅用于评价的工作副本清零local：新/旧组按实际binding决定是否继承解码父，调用冻结策略函数计算train/holdout forward KL。教师始终来自原prepared，不能用补偿后的resident函数代替fresh-slot学生。
4. W0对照用同一冻结features/教师/holdout、零local与零parent，得到K_new,W0。新KL始终owner均权。输出K_old、K_new、K_new,W0、G_new以及绝对旧KL减少量所需数据。
5. 另对编码后的 resident 全策略报告相对prefix的概率/有效矩阵漂移；这不是fresh-slot保护评分。
6. 克隆完整checkpoint，安装解码参数和parent_group；按原命名local/shared optimizer组清actor状态，保留critic及其他状态。以torch原生zip序列化工作副本；返回的checkpoint bytes是实际计量对象，若wrapper保存必须逐字节写入，不再次序列化。额外教师、lambda、诊断、L-BFGS状态均不进入learner。

完整持久字节定义沿用E161：`foundation_bytes + serialized_checkpoint_bytes + actor_bytes + raw_evidence_npz_bytes`。foundation的固定元数据来自E161源清单：55477 B，SHA256写入config，须与checkpoint.foundation_sha256一致。输入提取不额外读取foundation内容。actor必须恰89 B，总包≤262144 B。C/D实际序列化大小分别报告，不用cap相同冒充实测文件大小相同。

结果含可JSON化的metrics及独立deployment bytes；实验runner保存bytes时不得把临时教师和optimizer workspace一起持久化。A keep不在Stage0重复编码，未来沿E161 keep路径；所有臂未来rollout更新、actor Adam清理和critic保留规则不在本次更改。

## 内存、成本和归属

接口报告冻结数组/教师字节、L-BFGS临时状态与工作区估计，以及本进程可用的生命周期峰值内存元数据。进程峰值包含解释器/依赖和此前工作，不能称为某个拟合的独占增量；估计工作区也不能冒充OS实测值。真实runner须另行完整核算进程树CPU、启动、失败、统计与campaign墙钟。

本次新增两个科学模块、一个固定config、必要合成tests及本附录冻结交付；不改H026/H033、旧E数据、主控合同、原提案、共享H/索引/预算。真实E163 runner和统计归消融实验（1），它只编排此科学接口。预算和screen由主控拥有，本实现不会自行启动实验或增加候选。

Stage0固定全部16个旧prefix，lambda0/1各一次；至少12/16同时满足 `K_old,C>1e-10`、`G_new,C>1e-10`、`K_old,D≤.75 K_old,C`、`G_new,D≥.5 G_new,C`。比例筛查只是机制代理，合法C0/预算终止返回照常评分，不删除、不补跑。技术错误整批停止，不当作机制阴性结论。
