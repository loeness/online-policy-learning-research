# E163 repair002 — 本地修复，等待主控审查与资源决定

run001在首个生命周期预检中报 **ACCOUNTING_INVALID**，科学worker和真实拟合均为0。
启动路径以`exec`复用了长期SSH shell的PID；原进程起点及既有子进程计数进入了监督器
计量。旧版本、原始三文件及成本未知字段均已保留，未重跑。

失败结算见 [run001报告](../../../analyses/a001/report.md)、
[原始镜像](../../../runs/run001/README.md) 和
[结算身份](../run001_settlement.json)。完整campaign CPU/墙钟为UNKNOWN；已知预检
wait4 CPU0.122070秒。原CPU小计0.268236、进程年龄墙钟67.527558秒及名义尾部上界
均未改写成有效的精确cost，单次reservation已关闭，未计算可复用余额。

## 修复内容

新入口明确启动独立Python子进程，由父shell记录父子PID并显式wait。监督器完成
授权校验后、任何fork前，记录实际父PID、自身/父进程起点、初始RUSAGE_SELF和
RUSAGE_CHILDREN；已等待子进程CPU必须为0，不靠减掉历史用量掩盖旧PID问题。
自身CPU仍累计解释器启动开销，墙钟仍从新监督器进程起点算起。

原CPU子孙生命周期预检、停止预检、115CPU/170wall安全停止与5CPU/10wall收尾
预留保持不变。资源基线和wait4核对的初始/最终边界明确记录。授权缺失或无效时，
拒绝进入资源采样和真实预检。

新路径是`/data/robot/p001/e163/o002`及`run002`；既有`s001`科学快照继续只读复用。
E161输入仍指向`e161/run001`。gate与映射逐字节不变；科学21文件身份、配置、
16seed×lambda0/1、32fit和12/16门槛不变。旧o001/run001/launch/冻结清单未改。

## 可审查版本

- 科学identity仍为`0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea`。
- 新编排identity为`b8df209ce3d89df2a4847b2b7e2da380e8d01b819e1f12f8ccde70782ad1353b`，
  包括runner、supervisor、gate、mapping和关键launcher五项。
- 候选run spec SHA256为`d5764b8091aba95b7223ee7cdcb18d752e123d17063bb0c936775753f6e8189d`。
- 本地候选包37912B，SHA256为`11a23bab40e945b1d73252b2d75207c99882fc5b83eff2725127dae21044a3c9`。

拟定入口：

```text
sh /data/robot/p001/e163/o002/launch_command.sh
```

完整子进程命令与env见 [launch_command.sh](launch_command.sh) 和 [freeze.json](freeze.json)。
启动脚本使用LF字节，可供后续Linux文件方式调用。本地检查仅为AST解析、字节/清单
一致性与静态复审；没有执行shell、导入监督器、做计量probe或SSH。

候选run spec与false授权模板仅用于主控审查，尚未取得新的运行额度。本地包未传输。
原120CPU/180wall数值只表示未改的方法上限，不是第二次reservation。若主控决定
继续，仍需先确定资源、正式绑定新spec授权、刷新远端容量与身份，再在新campaign
同一预算内验证Linux生命周期/停止路径。不得直接沿旧授权重启。

[最小diff](orchestration.diff) · [独立静态复审](static_review.json) ·
[显式候选传输清单](transfer_manifest.json) · [false授权模板](authorization.template.json)
