# V2 终点持久性评估补充

在三种子正式在线结果解释前，从15个主V2终点执行同一配对重访：六情境×八个完整episode×保留/清除修正两条件。清空短历史、保留路由库，observe仍更新路由统计，新槽分配清除不再归属于它的修正，策略梯度更新为零。使用 `restore_knowledge_v2.py` 与对应路由工厂还原状态；逐episode reset和action随机流匹配，沿用上一轮修正评估的方案。

输出 `outputs/knowledge_v2_revisit`；仿真交互、时间、源码与输入hash另计，不并入训练累计回报。该评估不按结果选择训练超参数或最好checkpoint。
