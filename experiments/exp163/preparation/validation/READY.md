# E163 — READY，等待主控正式开启 campaign

2026-10-22。冻结科学实现独立审查 **PASS**，runner 独立复审 **PASS**。
Linux supervisor 已完成源码审查及语法检查；生命周期/停止路径预检尚未运行，
必须在正式 campaign 内通过后才可拟合。未改科学快照、共享 H/索引或主控预算。

## 精确版本与远端核验

- 科学 source identity：`0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea`。
- 编排 identity：`0cb8bb54683809acd20b8303a7a73ff578759a62875d4627bbe9be778f242168`。
- run spec SHA256：`6e7604856c28ce33218f732eb893c1cc56827cab7eb4e51abee21b994f4491e8`。
- 白名单包 27 项、133490 B，已逐字节传至 `/data/robot/p001/e163` 并核验。
  包含 21 项科学冻结文件、3 份编排代码、gate 映射、run spec 和原科学 manifest。
- 现存 E161 的48个 prefix 文件和指定 foundation 共49文件哈希全部匹配；
  CPython3.12.15 可用，33项依赖版本匹配。未反序列化任何 prefix。
- 传输后项目总量2905929927 B，E163约640799 B（receipt写入前），目标盘可用
  956723200000 B。预留峰值计划109680415 B，小于256 MiB，也满足项目100 GB。
  启动前需重新核对动态容量。

## 精确启动

工作目录 `/data/robot/p001/e163/s001`，先加载既有项目 entry helper，再按
[launch_command.sh](launch_command.sh) 覆盖线程、缓存和临时路径；HOME保持原值。
该脚本只是已冻结命令文本，本轮没有执行。

```text
/data/robot/p001/env/bin/python -I -B /data/robot/p001/e163/o001/supervisor.py --spec /data/robot/p001/e163/o001/run_spec.json --authorization /data/robot/p001/e163/o001/authorization.json
```

正式授权文件目前不存在。主控更新账本并发出绑定上述run spec的正式指令后，
验证方才可按 [authorization.template.json](authorization.template.json) 写入授权文件并启动。
缺失/false/哈希不符时拒绝，不创建run001、不做预检。该拒绝路径本轮也未试运行。

固定16 prefix × λ0/1，共32拟合；0交互、0GPU、1数值线程。每个拟合保存
diagnostics/metrics、精确候选和原样deployment actor/checkpoint，evidence和foundation
按原字节身份引用。独立gate仅映射编码后的fresh-local=0 holdout五字段。

CPU树预检、停止预检、启动、失败、拟合和报表共用120 CPU秒/180墙钟秒。
115 CPU秒/170墙钟秒安全停止，留5 CPU秒/10墙钟秒完成收尾；完整worker生命周期
由wait4核算，supervisor最终写入尾部以保留额界定，明确区别于精确实测总量。
无可靠计量/停止证明则不启动拟合；失败不自动重试，也不增加seed或启动Stage A。

**当前真实拟合、计量预检、环境交互均为0，campaign尚未开始；SSH/SFTP已关闭。**

[机器可读准入状态](readiness001.json) · [冻结清单和命令](execution_freeze001.json) ·
[科学审查](frozen_review.json) · [编排复审](runner_review.json) ·
[远端传输核验](transfer_receipt001.json) · [计量设计](metering_design.md)
