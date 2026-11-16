# E165 · H036 零新增新训练代价的有限受限搜索

状态：PLANNED / IMPLEMENTATION_AND_STATIC_REVIEW。日期：2026-10-24（Asia/Singapore）。没有 SSH、传输、研究检查、拟合或新评分执行许可。

归属：[H036](../../research/records/H036.md)。方法：[算法设计](../../research/appendix/H036/NEXT_DESIGN_20261024.md)；主控接受与范围：[合同](../../research/appendix/H036/COORDINATOR_E165_CONTRACT_20261024.md)。原 E163/E164 结果和关闭额度不改。

问题：现有九系数表示、同教师及 C0 新训练KL界内，固定 C→U 二分缩步能否返回原 D 未交付的可行旧保护改进？S1与S0只差约束是否参与停止；另对照原D和新拟合C。全16prefix；0真实fit/恢复；≤336完整训练oracle，≤33完整评分包。新训练、float32、部署编码和留出结果分开。已见holdout探索性复用，不能称独立确认或奖励结论。

资源规划：[budget.json](budget.json)。120CPU秒/240累计受控wall秒/128MiB，分CHECKS、DIAGNOSTIC、AUDIT三段；全部远端，先冻结/审查，再逐段主控精准准入。当前resource/execution均false，无活动job/reservation。

代码父提交：`81de4afd2d703d17c3953facb78fd04b19cb64a9`；工作树已有未提交改动，实际执行身份还须绑定冻结源码/dirty source快照及SHA，HEAD不足以代表本E代码。

输入：PLANNED，E161固定prefix与foundation，E163 C/D，E164 U32，逐项引用清单待冻结。原始输出与正式报告：NOT_RUN。Figure paths：NOT_APPLICABLE，本次配对表足够。执行归属：消融实验（1）；实现归属：算法实现；最终科学综合：主控。

2026-10-28：科学源码已静态冻结，身份 `4e96df16ca207ea961bafaed6b92dfbee07d7cdd55dc7cf986b5fb2b48189f0b`；36项快照逐文件字节/SHA核对无不符，独立窄静态审查通过。远端 CHECKS 仍未运行，验证方准备检查包；无新科学结果或运行准入。

2026-10-28：用户要求手动转达交接。主控已发布[CHECKS仅准备指令](preparation/coordinator_prepare001.json)，限定容量核对和42项白名单传输；独立预算已登记，runtime尚未授权。待用户转达准备结果与最终spec身份后，由主控判定是否启动检查。
