# 项目一：研报复现审计与2009—2025年度滚动研究

这是原项目的修订副本。结果以 `reproduction_report.html`、各结果目录的 `summary.csv` 以及 `verification_current.json` 为准；不要沿用旧包27.25%等不同口径指标。

## 使用
Python 3.12。建议新建conda环境；`python -m pip install -r requirements.txt`。CPU版PyTorch可从官方CPU wheel索引安装。实际运行环境见environment_current.txt。

快速复核：
```
python verify_data.py
python -m unittest discover -s tests
python test_timing.py
python run_neural_grid.py --prepare-only
python verify_current.py
```

完整重跑：
```
python reproduce_all.py --workers 2
```
神经训练在CPU上耗时较长；低内存电脑设为1。默认复用已完成的神经模型；若需从头训练，将results_neural_grid文件夹备份并移出本项目后再运行。不要只删除某个年份的预测而保留同名完成标记JSON。sequence_cache和Alpha158缓存可重建，未必随包携带。修改数据或特征后须同时清理对应旧缓存，不能混用旧模型与新输入。

## 结果目录
- results_2015_2025：Alpha158基线/原始振幅/原始振幅残差，2015—2025共11折。
- results_with_reversal：加入振幅和反转后的共同样本LightGBM对照，同样11折。
- results_raw_factors：两个原始因子的独立可用样本IC、分组标签收益与费用后Top50。
- results_neural_grid：两因子×两种网络×11年检查点、逐轮验证记录及年度样本外预测。
- results_replication：共同样本下原始因子、GRU、DBD、两因子组合与Alpha158预测对照。
- results_stacking_2021_2025：严格上游OOS堆叠；5年下游训练+1年验证，首个测试年2021。

## 数据及不能复现的部分
行情冻结到2025-12-31，历史成分来自Qlib社区。成交金额为社区原始Tushare金额（千元×1000），成交笔数为用户Wind导出；并非全部来自Wind。覆盖见coverage_2009_2025.csv，缺失不填零。

缺少小单主动买、卖额：第三分支和三因子DBD-Combine不能运行。全单买入不能替代。缺历史行业、市值、Barra、基准权重、完整ST及精确交易限制，不能完成原研报中性化和约束指增。22项对照见replication_matrix.csv。

普通GRU与DBD使用20×2输入、64隐藏维度、2层、100最大epoch、20早停。Adam学习率0.001、梯度裁剪5、训练期MAD缩放和截断为实现假设，不能说是研报披露参数。本项目5训练+1验证协议、CSI300股票池、t+1→t+21标签和2015—2025测试区间是明确的项目扩展。

## 交易与指标口径
月频、Top50、目标95%股票仓位，t日收盘信号、t+1收盘近似成交；买5bp/卖15bp。持仓随价格漂移。涨跌停为9.5%对称代理；缺价持仓沿用最后价格，未完整处理真实退市清算、排队、容量、手数。结果为研究模拟，不是实盘。

净年化从全程净值计算，最大回撤从完整路径计算，分年另列实际收益及年度回撤。IR相对沪深300价格指数，Sharpe无风险取0。最后21日无完整标签不算IC，但组合估值至2025年底。分组20日标签收益不是可累乘的日频组合收益。

## 修改定位
- research/features.py：修复主动买卖原始因子错误相减，保留显式排序口径。
- research/neural.py：100轮/20轮早停；保存训练期预处理与最佳状态。
- run_neural_grid.py：内存映射输入、年度并行运行、检查点与OOS provenance。
- prepare_verified_inputs.py：跨来源金额/笔数对齐、单位和恒等关系检查。
- run_research.py：完整11折、时间边界、低内存预处理、逐折预测和参数保存。
- research/combine.py：对称逆平方根正交及等权合成；当前只运行两因子。
- run_stacking.py：严格OOS下游A/B/C比较；不以训练内输出补早年。
- verify_current.py：独立重算净值/费用/风险指标、逐模型回放预测。

原包的其他入口属于历史或扩展工具，本次以reproduce_all.py为主入口，不将HMM或新闻实验混入本次研报结果。
