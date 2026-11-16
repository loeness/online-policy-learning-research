# H037 远端外层监督器草案

状态：本地静态草案；**不可执行、未准入**。没有连接服务器、传输文件、运行 supervisor、测试、合成检查或研究代码。仅允许后续对 `supervisor.py` 做 AST 静态解析。

## 预期阶段接口

每个阶段由独立的新 Python 进程启动一次：

```text
python supervisor.py CHECKS --spec <CHECKS spec> --authorization <CHECKS authorization>
python supervisor.py SINGLE_RUN --spec <RUN spec> --authorization <RUN authorization> --receipt <CHECKS state> --receipt-sha256 <SHA256>
```

`SINGLE_RUN` 必须绑定成功的 CHECKS 回执。计划中的 spec/auth 字节哈希绑定阶段、单次新预算分配、逐项代码/配置/输入清单、精确 argv、工作目录、输出/cache/temp 路径、CPU/墙钟/字节/持久包预算及 seed。CHECKS 的交互上限固定为 0；SINGLE_RUN 不得超过 H037 提议的 3,584 次。两者仍需主控逐阶段授权，提议值本身不构成授权，不能沿用历史轮预算。

## 当前阻塞

1. **算法实现尚未可作 CHECKS 输入。** 本次独立审查发现 runner 在 H037 `pre_reset` 返回 error 后仍执行 reset/继续；异常不能满足要求的停止语义。int8 编解码、时间切分和非空 cache 恢复也有测试缺口。算法方须先修复、冻结并独立审查；本草案不改算法文件，也不声称这些检查通过。
2. **无法静态证明完整子孙进程预算边界。** 单进程 `RLIMIT_CPU`、父进程墙钟和直接子进程 `wait4` 不能单独证明全树累计 CPU、墙钟超额停止和全部安全回收。监督器不得扫描其他账户进程；目前没有已审查、已授权的进程树隔离/计量适配器。因此 `supervisor.py` 在任何 worker 启动前明确 fail-closed，不含可达的 worker launch。
3. **交互、GPU、线程仍缺硬执行器和冻结回执。** `state_worker.json` 的预期契约包含交互数、GPU 数、worker/数值线程数及持久包清单，但这是事后审计证据，不能单独硬限制交互或 GPU。CHECKS 的 0 交互与 RUN 的至多 3,584 交互需由冻结代码和可审查计数证据支持，GPU/线程需有运行时强制和记录；当前均未建立。
4. **空间与持久包限制尚未能完整强制。** 草案只描述授权项目路径下的容量准入和输出包核验；没有防止依赖写到项目外的文件系统边界，也没有运行期间可强制停止写入的边界。环境变量与自报包清单不足以保证这一点。
5. **预算和路径没有获准。** 算法建议的 CPU、墙钟、空间及持久包数值仍待主控接受；本目录也没有正式的 H037 phase spec、authorization、唯一新 allocation ID、前序回执或远端路径准入。当前不能生成可运行 spec，更不能推断已有预算可复用。

## 对照和边界

E164/E165 的既有接口以 fresh phase supervisor、spec 精确 SHA、逐文件 bytes/SHA 清单、阶段授权、前序 state 回执、明确 CPU/墙钟/输出字节账本、worker 状态结构校验为参考。它们的 wait4/受管进程记录不被本草案当成“任意子孙进程均已受硬约束”的证明。

当前 `supervisor.py` 只保存这些输入绑定、路径/容量检查及结果回执的审查接口；结尾的显式阻塞是有意设计。它不是执行器，不应上传、调用或用于宣称阶段准入。后续要继续，须先解决上述边界并由主控另行冻结预算、输入白名单、授权 spec、命令及独立审查结论。
