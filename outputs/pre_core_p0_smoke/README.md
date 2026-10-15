# P0 软件 smoke check

这是 1 seed、1 source、256 PPO 步和每目标 2 episodes 的小规模软件检查，实际交互 1,772 步。它不进入正式 P0 的 18-run / 309,351-step 结果。

此运行使用的确切 runner 版本另存于 `archive/source_versions/PRE_CORE_P0_smoke_runner.py`，SHA-256 为 `46b650ec3a87a82bad693ab5a5d331ea8588c31905e8ec64efe23a100099375d`，与本目录 `provenance.json` 中的 runner hash 一致。正式运行前仅增加了评估前后参数 hash 的运行时守卫，并重新通过 6 项聚焦检查；正式源码提交是 `e4515be`。

checkpoint 中的 `smoke_test` 为 true。保留此记录用于软件与来源核查，不将其视为方法效果证据。
