# E165 静态源码交付

此目录交付完整白名单源码快照及端点引用，不附真实checkpoint/actor/evidence副本，不包含本机指令或凭据。

- `manifest.json`：本轮源码身份、完整快照、父Git SHA/dirty差异、输入与检查规格身份；状态必须为`SOURCE_FROZEN_REMOTE_CHECKS_PENDING`，不是runtime通过。
- `anchor_manifest.json`：237项准确旧工件引用，全16case/48prefix身份、C/D/U32及训练预期；配置钉住其SHA。
- `CHECKS_SPEC.json`：远端单次检查节点、预期数量、合成调用包络、实际计数渠道与停止边界。
- `source_snapshot/`：逐项原字节副本；实际运行以该树和被接受manifest身份为准。
- `static_checks.json`、`tracked_source.patch`：静态AST/JSON/清单与受限tracked差异；没有执行研究代码。
- `freeze_static.py`：只进行白名单字节复制、AST解析、清单及Git差异封装，不导入研究模块；拒绝覆盖已存在的冻结manifest。

实际研究检查只由消融实验在主控获准远端执行。本机没有pytest结果或科学验证receipt。先源冻结/独立静态复审，再由主控允许限定准备和CHECKS；检查receipt返回算法方确认后，主控另定DIAGNOSTIC与AUDIT准入。任何失败保留证据，不自动重试或本机代跑。

方法和精准API见父目录的`METHOD_BUDGETED_V1.md`、`HANDOFF_BUDGETED_V1.md`。manifest中的环境约束/资源计划不自行产生SSH或runtime授权。
