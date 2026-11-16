# H036 算法实现交付 v1

日期：2026-10-22。交付范围为受控实现与合成正确性检查；**没有真实 prefix 拟合、SSH、环境交互或 E163 结果**。方法见 [METHOD_V1](METHOD_V1.md)，科学约束见[主控合同](COORDINATOR_CONTRACT_20261022.md)。真实执行须由主控接受冻结交付、消融实验（1）独立审查并完成 E163 readiness 后进行。

## 冻结身份与文件

使用 [delivery_v1/manifest.json](delivery_v1/manifest.json) 中的完整 Git 父 SHA、`source_identity_sha256`、逐文件哈希及 tracked patch 识别代码。父提交不能单独代表此 dirty checkout。运行来源为 `delivery_v1/source_snapshot/`，不得从共享工作树临时拼出另一版科学实现。

新增科学文件为 `src/online_rl/retention_protected_candidate.py`、`retention_protected_inputs.py`；固定参数为 `configs/h036_protected_v1.json`。快照包含其本地相对导入闭包、既有 requirements 与三个合成测试文件。旧 H026/H033 模块只作为冻结依赖复制，本轮没有改写其科学语义。

48 个真实输入只登记哈希引用，不复制到交付源快照，也不反序列化用于本轮拟合。manifest 的 `input_files` 以 `logical_case`、basename、SHA256、字节数绑定身份；Windows local_path 只是当前放置位置。实验方已准备的 `experiments/exp163/preparation/validation/input_manifest_planned.json` 应与这些值逐项匹配。

## 精确科学调用

从冻结根设置 `PYTHONPATH=<frozen-root>/src`，使用项目已有依赖；默认 config 在该根的 `configs/` 内。下面是实验 runner 应组合的科学调用，**不是已授权执行的 campaign 命令**：

```python
import json
from pathlib import Path
import torch
from online_rl.retention_protected_inputs import (
    validate_config, prepare_inputs, encode_and_score_stage0,
)
from online_rl.retention_protected_candidate import fit_protected_candidate

torch.set_num_threads(1)
torch.set_num_interop_threads(1)  # 在新进程中、任何相关计算前仅设置一次
config = validate_config()

# 由实验方按 manifest 解析显式 prefix 路径和三文件 SHA，禁止自行重算后
# 当作“expected”身份，也不把未来 outcome 或 observer 加入输入。
prepared = prepare_inputs(
    Path(prefix_path), config=config, expected_seed=seed,
    expected_identity={
        "checkpoint.pt": frozen_checkpoint_sha256,
        "actor.bin": frozen_actor_sha256,
        "evidence.npz": frozen_evidence_sha256,
    },
    logical_case=f"s{seed}_dynamics_late",
)
fit_c = fit_protected_candidate(prepared, lambda_value=0)
score_c = encode_and_score_stage0(prepared, fit_c, binding="wildcard")
fit_d = fit_protected_candidate(prepared, lambda_value=1)
score_d = encode_and_score_stage0(prepared, fit_d, binding="wildcard")

# 下列值由 runner 分别保存到其已冻结输出路径；不能 dump 含 ndarray/bytes
# 的整个 prepared/fit/score 对象，更不能把教师混入部署 checkpoint。
json.dumps(fit_c["diagnostics"], allow_nan=False)
json.dumps(fit_d["diagnostics"], allow_nan=False)
json.dumps(score_c["metrics"], allow_nan=False)
json.dumps(score_d["metrics"], allow_nan=False)
```

Stage0 每个固定 prefix 只拟合 lambda0、lambda1 各一次，共32次；C/D均为 wildcard，不添加 B/A 拟合或额外真实 smoke。未来如单独获准评价 B，应将同一个 `fit_c` 传入 `binding="new"`，不重复 lambda0 优化。A沿原 keep 路径，不用本接口重编码。

`fit_result` 是 `h036_fit_v1`，包含 float32 `candidate`、`prepared_id`、`lambda_value` 和 JSON `diagnostics`。后者记录所有 closure 的完整目标/新旧KL/接受拒绝与候选哈希、次数、初始化/获选float64/最终float32目标、回退、C0返回、CPU/wall和内存。holdout只报告；不得用于重选结果。

`score_result` 是 `h036_stage0_score_v1`，分为：

- `metrics`：JSON对象，含 `K_old`、`K_new`、`K_new_W0`、`G_new`；编码前、编码后及W0的train/holdout分项；解码父矩阵/哈希；补偿、resident漂移、actor优化器清理、持久字节和资源。
- `deployment`：`actor_bytes`、`checkpoint_bytes`、`evidence_bytes`。如 runner 需要落盘，分别**逐字节**写入对应文件。`checkpoint_bytes` 已通过 `torch.save(BytesIO)` 得到；重新 `torch.save` 会改变序列化身份/计量对象。foundation不在返回值内，使用 manifest/config 指定的同一份已验证55477 B foundation。

完整包字节为 foundation + checkpoint + actor + evidence；metrics.capacity 记录实际值并拒绝超262144 B。诊断与source快照属于实验记录，不应计作learner保存状态，也不能不计入campaign输出上限。

## 实验方应独立核验的协议

1. 核对冻结代码、config、48输入、foundation身份；审查教师、split、lambda0原 fitter一致性、安装/编码与fresh评分路径。实测平台信息单独记录，不把本机检查当远端通过。
2. 固定16个 seed：26201–26213、26215、26216、26218。每个owner与旧FIFO都是128行，96训练/32诊断；输入不符整批停止，不删seed、不补历史观测。
3. 对全部32次实际拟合与32次C/D编码评分保存上述diagnostics/metrics。合法closure预算终止、C0或float32回退照常计入，不重跑；错误保存部分证据并停止。数值闭包错误可能携带 `partial_diagnostics`，runner可保存，不将其当有效候选。
4. 使用编码后 holdout：一个prefix同时满足 `K_old,C>1e-10`、`G_new,C>1e-10`、`K_old,D<=.75*K_old,C`、`G_new,D>=.5*G_new,C` 才通过。报告绝对KL、新收益和旧KL减少量；至少12/16通过只支持探索性机制筛查。完整有效但不足12为NO_ADVANCEMENT，错误/不全单列INVALID/INCOMPLETE。
5. 主控负责解释与后续授权。没有奖励、early recovery、完整流或新seed结论；StageA不会由本接口自动触发。

## SSH 与预算交接

主控合同的后续用户指示指定 **robot、`/data/robot/p001/e163`**。本轮算法方没有连接服务器。实验方拥有runner、显式传输清单、输入/输出根映射、精确argv、实际依赖身份、限额监控与返回证据；不得从源快照漏带依赖后临时修改科学代码。

允许将快照根放置在上述工作根的中性子目录，现存E161输入的显式文件引用也由实验方核验；不把历史Windows路径当身份。现存远端E161文件可按已登记的项目内精确路径与SHA复用；新增E163输出只写E163根。代码/输入/缓存/临时文件都须遵守主控允许路径和明示清单，不扩展账号、项目目录或系统设置。

真实campaign计划边界仍为120累计CPU秒、180墙钟秒、单worker/单数值线程、0 GPU/0新交互、256 MiB新输出/缓存/临时文件；实际准入以主控E163预算状态为准，交付本身不改变 `execution_authorized`。相关环境设 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`，进程内torch线程也设1。项目缓存/temp/日志明确放在E163允许路径，不改HOME，不安装依赖。

计量须覆盖启动、零模型预检、实际解释器及已启动子进程、失败与统计；方法内CPU/wall只作分项，不能代替campaign计量。进程内峰值是生命周期high-water，不能解释成每次fit独占峰值。不能可靠实施限额则不启动真实拟合；没有免费重试或自动补跑。

## 已完成的工程检查

本机项目虚拟环境，单数值线程，三个合成测试文件共23项通过。覆盖目标/梯度、ridge只计一次、lambda0原fitter一致性、旧教师和行序/代次、禁止输入、身份搬迁、闭包上限/回退、完整codec、fresh与resident区分、完整持久包字节以及输入/RNG不变性。原始开发检查保存在 `development_checks/attempt001.*` 与 `attempt002.*`；冻结快照独立检查记录在 [delivery_v1/verification.json](delivery_v1/verification.json) 与 [checks.txt](delivery_v1/checks.txt)。

这些是合成工程证据，不占用或替代E163真实机制证据；本机测试通过也不说明真实prefix满足合同或机制门槛。没有修改共享H状态/索引、预算或旧结果，没有提交/推送。
