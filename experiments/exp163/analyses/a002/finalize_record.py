"""Close the single E163 run002 record after the independent offline audit."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUN = BASE / "runs" / "run002"
RAW = RUN / "remote" / "run002"
FINALIZED_DATE = "2026-10-23"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(path):
    data = path.read_bytes()
    return {"path": path.relative_to(BASE).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    audit, summary = read(HERE / "audit.json"), read(HERE / "summary.json")
    assert audit["audit_status"] == "COMPLETE_VALID" and audit["failure_count"] == 0
    assert audit["scientific_validation"] == "COMPLETE_AUDIT"
    assert audit["scientific"]["computed_pass_count"] == summary["pass_count"] == 8
    state = read(RAW / "state.json")
    mirror = read(RUN / "mirror_manifest.json")
    observation = read(RUN / "remote_result_observation.json")
    spec = read(BASE / "preparation" / "validation" / "repair002" / "run_spec.json")
    cost = {
        "experiment_id": "E163", "campaign": "run002", "finalized_date_asia_singapore": FINALIZED_DATE,
        "measurement_status": "VALID_OBSERVED_SUBTOTAL_AND_RESERVED_FOOTER_UPPER_BOUND",
        "independent_audit": "audit.json", "exact_complete_campaign_cpu_seconds": None,
        "exact_complete_campaign_wall_seconds": None,
        "exact_complete_cost_limitation": "Final footer after the last sample is not observed; explicit reserves bound it. SSH, parent shell, transfer and offline evidence audit are separate operational work.",
        "observed_cpu_subtotal_seconds": state["cpu_snapshot_before_final_footer"]["total_cpu_seconds"],
        "observed_wall_before_footer_seconds": state["campaign_wall_seconds_before_final_footer"],
        "cpu_upper_including_reserved_footer_seconds": state["bounded_campaign_cpu_seconds_upper"],
        "wall_upper_including_reserved_footer_seconds": state["bounded_campaign_wall_seconds_upper"],
        "footer_reserve": state["finalization_reserve"],
        "supervisor_self_cpu_seconds": state["cpu_snapshot_before_final_footer"]["supervisor_rusage_self_cpu_seconds"],
        "completed_child_wait4_cpu_seconds": state["cpu_snapshot_before_final_footer"]["reaped_wait4_cpu_seconds"],
        "child_wait4_records": state["child_wait4_records"],
        "wait4_children_audit": state["wait4_children_audit"],
        "worker_body_subtotal": read(RAW / "state_worker.json"),
        "environment_interactions": 0, "gpu_devices": 0, "scientific_workers_started": 1,
        "candidate_fit_calls": 32, "prefixes_scored": 16,
        "new_e163_total_apparent_bytes_after_return_archive": observation["e163_bytes_after"],
        "project_apparent_bytes_after_return_archive": observation["project_bytes_after"],
        "storage_measurement_utc": observation["measured_utc"],
        "retained_raw_evidence_bytes": mirror["raw_total_bytes"],
        "operational_cost": "NOT_METERED_AS_SCIENTIFIC_CAMPAIGN; admission, transfer/packing, byte-return and local offline audit recorded separately, no new fits/interactions",
        "cumulative_E163_cpu_seconds": None, "cumulative_E163_wall_seconds": None,
        "cumulative_status": "UNKNOWN due to run001 ACCOUNTING_INVALID; run002 does not reconstruct run001 cost",
        "settlement": "Single run002 reservation closed; no numerical remainder released and no additional start authorized.",
    }
    write(HERE / "cost_accounting.json", cost)
    with (HERE / "per_prefix.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = {
        "K_old_C_gt_1e-10": "旧KL分母",
        "G_new_C_gt_1e-10": "C无正新收益",
        "K_old_D_le_0.75_K_old_C": "旧KL降幅不足",
        "G_new_D_ge_0.5_G_new_C": "新收益保留不足",
    }
    table = []
    for row in rows:
        failure = "、".join(labels[k] for k in row["failed_conditions"].split(";") if k) or "—"
        table.append(f"| {row['seed']} | {float(row['K_old_C']):.6g} | {float(row['K_old_D']):.6g} | {float(row['G_new_C']):.6g} | {float(row['G_new_D']):.6g} | {'通过' if row['passed']=='True' else failure} | {'是' if row['D_returned_is_C0']=='True' else '否'} |")
    report = f"""# E163 run002：NO_ADVANCEMENT（8/16）

运行日期：2026-10-22；最终镜像、独立审计与结算：{FINALIZED_DATE}（Asia/Singapore）。

**有效、完整的 Stage0 结果为 8/16，通过数低于预设 12/16；本固定版本停止推进。** [独立审计](audit.json)为 COMPLETE_VALID，零失败；[原服务器报告](../../runs/run002/remote/run002/scientific_report.json)与独立复算一致。研究解释与下一步由主控决定，本记录不启动 Stage A、新种子或参数搜索。

## 问题与固定实施

检验同一个九系数父模块中，加入旧 keep 初始化行为保护项的 D（λ=1），能否相对 C（λ=0）在最终编码后降低旧行为损失，同时保留新来源收益。16 份固定 E161 prefix，每份两个候选，共 32 次拟合；新交互 0、GPU 0、单 CPU worker、数值线程 1。旧/新教师、128 行按前96训练/后32留出、ridge=1e-4、L-BFGS/过滤/回退、wildcard 补偿及原89 B codec 均保持冻结规格。

每个 prefix 必须同时满足 K_old,C>1e-10、G_new,C>1e-10、K_old,D≤0.75K_old,C、G_new,D≥0.5G_new,C；G_new=K_new,W0−K_new。评价使用完整 actor 编码、解码后 fresh local=0 的 holdout；训练值或编码前值不用于挽救门槛。

## 逐 prefix 结果

| Seed | K_old,C | K_old,D | G_new,C | G_new,D | 门槛结果／失败原因 | D返回C0 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(table)}

完整精度和四个条件见 [per_prefix.csv](per_prefix.csv)，32 次拟合的训练/留出、编码前后、漂移、回退、closure、容量与分项成本见 [per_fit.csv](per_fit.csv)。所有 seed 均保留，没有替换或删除。

旧KL数值分母失败0个；C无正新收益3个（26206/26208/26218）；旧KL降幅不足2个（26204/26212）；新收益保留不足5个（26206/26208/26210/26213/26216）。失败类型有重叠，不能相加为独立 prefix 数。

## C0、编码与实际容量

C 的16次和 D 的11次返回非 C0；D 在26204、26206、26209、26212、26213合法返回 C0。全部32次 termination=optimizer_completed，float32 fallback_applied=0；返回 C0 与触发 float32 回退是不同记录。26209 是8个通过项中唯一 D 返回 C0 的 prefix，不能将其通过归因于学到的保护更新；仍按预设规则计入。

总 closure=2290，单次61–83，均≤256。actor均89 B，完整持久包233500–234681 B，均≤262144 B。checkpoint为接口返回的原序列化字节，未再次 torch.save；evidence与foundation按准确原路径、大小及SHA引用计入包。

补偿前后有效参数的最大误差≤9.536743e-7；完整编码后的 resident KL范围9.462210e-7–0.04870864，概率最大绝对漂移最高0.2184344。编码后减编码前的旧holdout KL变化范围−0.211278至+0.575262，新holdout为−0.0357775至+0.0883840。这些是描述性诊断，不能用补偿近似保持代替最终编码评价。

科学进程所记录 lifetime memory high-water mark为365060096 B（约348.15 MiB），不是32个隔离拟合的各自峰值。教师数组12288 B、临时数组估算117072 B、optimizer tensors1752–1760 B为分项范围，不能相加当作完整进程峰值，也不是持久learner容量。

## 成本与结算

| 口径 | run002 |
| --- | ---: |
| 观测 CPU 小计（截至最终footer前） | 8.625535 秒 |
| 观测墙钟（截至最终footer前） | 10.011705487 秒 |
| 含5秒CPU收尾保留额的CPU上界 | 13.625535 秒 |
| 含10秒墙钟收尾保留额的墙钟上界 | 20.011705487 秒 |
| 科学worker完整wait4 CPU | 7.791221 秒 |
| 新环境交互／GPU | 0／0 |
| 原始返回证据 | 250文件／6875913 B |
| 回传归档后E163总目录（含首次失败及源码/归档） | 11767008 B |
| 同时记录的项目总量 | 2917056136 B |

CPU小计=supervisor自身0.711534秒+三个已回收子进程wait4总计7.914001秒。生命周期CPU预检、停止预检均在同campaign内先于拟合通过；fresh PID1868373由parent1868372启动并显式wait，授权后fork前waited-child基线为0，最终用户/系统CPU各自的wait4对账差约1e-6秒，live handles为空，退出码0。

上述上界在本次120 CPU秒／180墙钟秒内；footer保留额不等于精确实耗。计量范围是新supervisor的内核进程生命周期及其子进程；SSH/父shell、准入、打包回传和本地离线证据审计另记运维，未冒称已被这一时钟精确覆盖。详见 [cost_accounting.json](cost_accounting.json)。

**累计 E163 完整CPU／墙钟仍为 UNKNOWN**：run001在拟合前因计量错误停止，保留原证据与未知成本；run002有效计量不能修补首次缺口。两次单次reservation均已关闭，不计算可复用余额，execution_authorized=false、active_job=null，无自动第三次。

## 身份、证据与审计边界

科学source identity=`0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea`；repair002 run_spec SHA=`d5764b8091aba95b7223ee7cdcb18d752e123d17063bb0c936775753f6e8189d`；编排identity=`b8df209ce3d89df2a4847b2b7e2da380e8d01b819e1f12f8ccde70782ad1353b`。来源为已接受的21项冻结源码/配置及48项prefix输入，foundation单独准确引用。

服务器原证据仅来自`/data/robot/p001/e163/run002`，本地对应 [runs/run002/remote/run002](../../runs/run002/remote/run002)。[镜像清单](../../runs/run002/mirror_manifest.json)逐项核验250个文件的大小和SHA；ZIP SHA=`2e60794044d85a597983ecb3ecd46c43c6d03c3c66c5641bcec0bbead7232503`，远端manifest SHA=`1a19f4567e78759fe1fe0c36e6fca7fc2e1896d1acd3f0a7b241425cb7effa22`。原始字节全部保留，无再次反序列化模型或新增拟合。身份、命令、授权、输入与回传映射见 [provenance](../../runs/run002/provenance.json)。

独立审计仅用标准库：验证镜像/候选字节/原部署哈希/容量、日志计量及32份指标映射，独立重算门槛；不再运行冻结科学数值算法。离线审计器初次有repair002 schema及D闭包调用计数适配错误，原始中间报告保留，修正依据见 [audit.md](audit.md)。这些本地适配修正没有改变冻结算法、输入或远端结果。

## 解释与交回主控

本固定λ=1版本未达到预设必要机制筛查门槛，不能据此推进已提案的在线奖励筛查。观察到部分prefix能减少旧KL并保留新收益，也存在新收益损失、C无正收益和旧KL未改善的情形；这些结果不证明一般函数类无法兼容，不估计reward效果，也不否定整个研究问题。后续消融或算法改动必须由主控基于完整证据另行决定和登记。
"""
    (HERE / "report.md").write_text(report, encoding="utf-8")
    provenance_files = [
        BASE / "preparation/coordinator_start002.json",
        BASE / "preparation/validation/repair002/freeze.json",
        BASE / "preparation/validation/repair002/run_spec.json",
        BASE / "preparation/validation/repair002/transfer_manifest.json",
        BASE / "preparation/validation/repair002/admission_start002.json",
        BASE / "preparation/validation/repair002/transfer_start002.json",
        RUN / "launcher.json", RUN / "launch_attempt.json", RUN / "mirror_manifest.json",
        RUN / "return002_manifest.json", HERE / "audit.json", HERE / "summary.json",
    ]
    write(RUN / "provenance.json", {
        "experiment_id": "E163", "hypothesis_id": "H036", "campaign": "run002",
        "finalized_date_asia_singapore": FINALIZED_DATE,
        "scientific_source_identity_sha256": spec["source_identity_sha256"],
        "git_parent": spec["git_parent"], "frozen_scientific_source": "research/appendix/H036/delivery_v1/source_snapshot",
        "remote_account": "robot", "remote_project_root": "/data/robot/p001", "remote_output_root": "/data/robot/p001/e163/run002",
        "local_mirror": "runs/run002/remote/run002", "raw_files": mirror["file_count"],
        "raw_bytes": mirror["raw_total_bytes"], "archive_sha256": mirror["archive_sha256"],
        "entry_argv": read(RUN / "launcher.json")["entry_argv"],
        "frozen_input_files": spec["input_files"], "foundation": spec["foundation"],
        "environment_evidence": "runs/run002/remote/run002/environment.json",
        "identities": [identity(p) for p in provenance_files],
        "raw_artifact_policy": "Verbatim returned bytes; no checkpoint deserialization/reserialization during evidence return or offline audit.",
    })
    (RUN / "README.md").write_text("""# E163 run002 — COMPLETE_VALID / NO_ADVANCEMENT

The one authorized repair002 attempt completed 32 fits on all 16 fixed prefixes,
with zero new environment interactions. The independent offline audit verified
the complete 250-file byte mirror and recomputed 8/16 passes against the fixed
12/16 gate. See [final report](../../analyses/a002/report.md),
[audit](../../analyses/a002/audit.json), [provenance](provenance.json), and
[mirror manifest](mirror_manifest.json).

Raw files under remote/run002 are preserved byte-for-byte. The return archive
and inventory are retained. SSH scientific launch exited 0; no recorded live
owned handles remain. The old SFTP tool handle expired after download, so its
exit status was not captured; successful complete transfer is established by
archive and all-file size/SHA identities, not by an assumed transport exit code.

Observed run002 CPU subtotal/wall are 8.625535/10.011705487 seconds; explicit
footer reserves produce upper bounds 13.625535/20.011705487 seconds. These are
not exact footer-inclusive measurements. Cumulative E163 complete cost remains
UNKNOWN because run001 accounting was invalid. No additional start is authorized.
""", encoding="utf-8")
    settlement = {
        "experiment_id": "E163", "campaign": "run002", "finalized_date_asia_singapore": FINALIZED_DATE,
        "state": "CLOSED_COMPLETE_VALID_NO_ADVANCEMENT", "reservation_id": "stage0_campaign002",
        "audit_status": audit["audit_status"], "gate_status": "NO_ADVANCEMENT", "passes": 8,
        "denominator": 16, "required_passes": 12, "execution_authorized": False, "active_job": None,
        "fit_calls": 32, "environment_interactions": 0, "cost_record": "analyses/a002/cost_accounting.json",
        "run001_unchanged": True, "cumulative_complete_cost": "UNKNOWN",
        "reusable_balance": "NONE_CALCULATED", "further_retry_or_stageA_authorized": False,
    }
    write(BASE / "preparation/validation/run002_settlement.json", settlement)
    budget_path = BASE / "budget.json"
    budget = read(budget_path)
    assert budget["execution_authorized"] is False and budget["active_job"] is None
    assert budget["attempts"][1]["id"] == "run002"
    budget["state"] = "COMPLETED_NO_ADVANCEMENT"
    budget["finalized_date_asia_singapore"] = FINALIZED_DATE
    budget["next_decision"] = "Return final valid 8/16 NO_ADVANCEMENT evidence to coordinator; fixed version stops. No third attempt, new seeds, lambda sweep or Stage A authorization."
    budget["spent"]["measurement_status"] = "Cumulative complete E163 CPU/wall UNKNOWN due run001. Run002 has independent valid observed subtotal and explicit footer upper bounds; legacy raw fields below remain run001 only."
    budget["spent"]["new_artifact_cache_temp_bytes"] = observation["e163_bytes_after"]
    attempt = budget["attempts"][1]
    attempt.update({"state": "COMPLETE_VALID_NO_ADVANCEMENT", "measurement_status": cost["measurement_status"],
                    "cost_record": "analyses/a002/cost_accounting.json", "audit_record": "analyses/a002/audit.json",
                    "gate_status": "NO_ADVANCEMENT", "gate_passes": 8, "finalized_date_asia_singapore": FINALIZED_DATE})
    for reservation in budget["reservations"]:
        if reservation["id"] == "stage0_campaign002":
            reservation["state"] = "CLOSED_COMPLETED_NO_AUTOMATIC_REUSE"
    budget["active_reservation_id"] = None
    budget["last_completed_reservation_id"] = "stage0_campaign002"
    budget["cost_records"] = {"run001": "analyses/a001/cost_accounting.json", "run002": "analyses/a002/cost_accounting.json"}
    budget["cost_record_scope"] = "Legacy cost_record points to preserved run001; cost_records separates both attempts. Cumulative complete CPU/wall remain UNKNOWN, not the sum of reported raw counters or authorized reservations."
    budget["last_job"]["evidence_state"] = "COMPLETE_MIRROR_BYTE_VERIFIED_AND_INDEPENDENTLY_AUDITED"
    write(budget_path, budget)
    readme = BASE / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("- 当前状态：**RUN002_AUTHORIZED_PENDING_REMOTE_ADMISSION / NOT_EVALUATED**；run001为ACCOUNTING_INVALID，0次真实拟合。",
                        "- 当前状态：**COMPLETED_NO_ADVANCEMENT**；run002完整有效，32次拟合、8/16通过，未达预设12/16；run001保留ACCOUNTING_INVALID。")
    text = text.replace("- 首次CPU生命周期预检拒绝，科学worker未启动；尚无H036机制结果。首次额度已退役；用户已明确追加一次120CPU秒/180墙钟秒/0交互预算，由消融实验（1）执行已验收的repair002，无自动后续重试。",
                        "- [最终报告](analyses/a002/report.md)、[独立审计](analyses/a002/audit.json)、[逐prefix表](analyses/a002/per_prefix.csv)已完成。两次单次额度均关闭；累计完整CPU/墙钟因run001仍UNKNOWN，无自动后续重试或Stage A授权。")
    text = text.replace("run002为修复预留，不存在实际结果", "[run002](runs/run002/README.md)：250文件原始镜像、32次拟合与完整评分，字节核验通过")
    text = text.replace("科学门槛尚未评价 |", "另见[a002有效结果](analyses/a002/report.md)，8/16、NO_ADVANCEMENT |")
    text += f"\n## {FINALIZED_DATE} run002 最终交付\n\n固定32次拟合完整结束，独立审计COMPLETE_VALID；8/16通过，低于预设12/16，结果NO_ADVANCEMENT。完整[报告](analyses/a002/report.md)、[配对表](analyses/a002/per_prefix.csv)、[逐fit诊断](analyses/a002/per_fit.csv)、[成本](analyses/a002/cost_accounting.json)及[结算](preparation/validation/run002_settlement.json)已保存。原始证据250文件、6875913 B逐字节核验；不改冻结源码或历史结果。\n\nrun002观测CPU小计8.625535秒、墙钟10.011705487秒，含收尾保留额上界13.625535／20.011705487秒；累计E163完整成本因首次计量缺口仍UNKNOWN。预算执行开关关闭，active_job=null，无自动第三次或后续阶段。全局H记录、研究索引及最终研究决定交主控集成。\n"
    readme.write_text(text, encoding="utf-8")
    with (BASE / "results.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment_id", "hypothesis_id", "campaign", "source_identity_sha256", "fit_calls", "prefixes_scored", "passes", "required_passes", "environment_interactions", "observed_cpu_subtotal_seconds", "observed_wall_before_footer_seconds", "cpu_upper_with_footer_seconds", "wall_upper_with_footer_seconds", "outcome", "interpretation"], delimiter="\t")
        writer.writeheader()
        common = {"experiment_id": "E163", "hypothesis_id": "H036", "source_identity_sha256": spec["source_identity_sha256"], "environment_interactions": 0}
        writer.writerow({**common, "campaign": "run001", "fit_calls": 0, "prefixes_scored": 0, "passes": "NOT_EVALUATED", "required_passes": 12, "outcome": "ACCOUNTING_INVALID", "interpretation": "Stopped before fits; complete cost UNKNOWN; preserved original reservation retired."})
        writer.writerow({**common, "campaign": "run002", "fit_calls": 32, "prefixes_scored": 16, "passes": 8, "required_passes": 12,
                         "observed_cpu_subtotal_seconds": cost["observed_cpu_subtotal_seconds"], "observed_wall_before_footer_seconds": cost["observed_wall_before_footer_seconds"],
                         "cpu_upper_with_footer_seconds": cost["cpu_upper_including_reserved_footer_seconds"], "wall_upper_with_footer_seconds": cost["wall_upper_including_reserved_footer_seconds"],
                         "outcome": "NO_ADVANCEMENT", "interpretation": "Valid fixed Stage0 KL proxy screen missed 12/16 gate; no reward claim or authorized next phase."})
    print(json.dumps({"state": budget["state"], "audit_status": audit["audit_status"], "passes": 8, "cumulative_cost": "UNKNOWN", "active_job": None}))


if __name__ == "__main__":
    main()
