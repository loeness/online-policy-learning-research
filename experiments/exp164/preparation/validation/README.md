# E164 本地执行包

最新状态：**COMPLETE_VALID / EXPLORATORY_ONLY**。唯一远端 campaign 已完成：5 次恢复拟合、10 个新评分包、16 个逻辑配对、0 新交互/GPU；原始 279 文件已完整镜像并通过独立审计。详见 [结果报告](../../analyses/a001/report.md)、[最终审计](../../analyses/a003/audit.json) 与 [成本披露](../../analyses/a001/cost.json)。reservation 已关闭，项目作业和 SSH 已退出，无下一轮授权。

2026-10-23 本地交付时：**READY_FOR_RESOURCE_DECISION**。科学源码与执行编排已完成精确绑定和必要工程检查；尚未取得单次运行资源准入，也没有真实 E164 结果。

[最终交付记录](READY_FOR_RESOURCE_DECISION.json)列出精确 source、orchestration、spec、输入与传输身份，以及 argv、cwd、环境和预算范围。[执行身份](execution_freeze.json)、[本地工件清单](local_preparation_manifest.json)与[静态检查](static_checks.json)可用于主控最终审阅。

runner 的输入映射必须等于科学接口实际读取的三个固定文件，输出初始化失败也会保存状态或写入捕获的 stderr；两项修复与 CPU 构建声明经[定向独立复核](runner_static_review.md)。[独立核验器](verify_results.py)读取固定原始文件，核对轨迹/锁定/评分包/别名/原始计量并独立重算报告。缺少实际运行或外部启动证据时不能判为 COMPLETE_VALID。[人工 JSON 工程检查](engineering_smoke.json)五组通过；首次测试脚本字段假设错误及修正已保留，未改科学实现。9 项科学合成检查与旧 1876 项审计未重跑。

固定设计为 16 个逻辑配对：11 项准确复用，恢复 26204/26206/26209/26212/26213 各一次，最多 5 次拟合和 15 个新完整评分包。全 cohort 先锁定，5 个 V 锚点通过后才评分 UG/UR。报告只比较 UG−V、UR−UG 的旧 KL 与新收益变化，属于机制诊断，不作推进或奖励结论。

规划上限为 60 CPU 秒、120 campaign 墙钟秒、64 MiB 新文件/cache/temp、单 worker/单数值线程、0 GPU/新交互；55 CPU/110 wall 停止，保留 5 CPU/10 wall 与 65536 B 收尾额度。计量从 fresh supervisor 出生开始；父 launch shell、SSH、传输与准入是外部运维，elapsed 另列、未测 CPU 记 UNKNOWN。详见[metering_notes](metering_notes.md)。无自动重试、不继承 E163 额度。

精确传输清单为 40 文件 / 1177074 B，308 项现有引用不整体复制。规划新增峰值为 32710053 B，低于 64 MiB；[空间计划](storage_plan.json)中的远端 free bytes/inodes/project baseline 保持 null，获准后才能刷新。新工作仅在 robot 的 /data/robot/p001/e164，项目总上限 100 GB。

当前 run_spec.json 的 resource_authorized=false，授权模板也为 false，未建 reservation、未 SSH/传输/远端写入/预检/拟合/评分。主控准入后需要刷新限定目录容量、更新动态字段并重新绑定 spec SHA，随后才执行一次有界 campaign 和真实结果审计。*.draft.json、*_initial.json 为保留的历史草案；当前控制文件以 run_spec.json、execution_freeze.json 和最终交付记录为准。
