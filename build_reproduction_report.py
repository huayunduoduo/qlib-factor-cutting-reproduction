"""Create a self-contained HTML audit from actual saved results only."""
import base64,io,json,html
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.data import ROOT

NAMES={'E0_baseline':'Alpha158基线','E1_raw_cut':'加入原始因子','E2_residual_cut':'加入原始因子残差','cut_amplitude':'理想振幅','cut_reversal':'理想反转','raw_amplitude':'原始振幅（反向）','raw_reversal':'原始反转（反向）','gru_amplitude':'振幅GRU','dbd_amplitude':'振幅DBD','gru_reversal':'反转GRU','dbd_reversal':'反转DBD','dbd_two_factor':'两因子DBD组合','alpha158_lgb':'Alpha158基线（匹配样本）','A_alpha158':'A：Alpha158','B_plus_two_dbd':'B：加入两因子DBD','C_plus_residual_two_dbd':'C：加入两因子DBD残差'}
def load(folder,name):return pd.read_csv(ROOT/folder/name)
def table(df):
    d=df.copy()
    for c in ['experiment','factor']:
        if c in d:d[c]=d[c].map(lambda x:NAMES.get(x,x))
    percent=['CAGR','MDD','total_return','year_return','benchmark_CAGR','mean_two_sided_turnover','arithmetic_excess_annual']
    for c in d:
        if c in percent:d[c]=d[c].map(lambda x:f'{x:.2%}' if pd.notna(x) else '—')
        elif pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda x:f'{x:.4f}' if pd.notna(x) else '—')
    d=d.rename(columns={'item':'核对项','reference_or_project_requirement':'研报或项目要求','status':'状态','implementation_or_limitation':'实现与限制','file':'对应文件','experiment':'实验','factor':'因子','year':'年份','CAGR':'净年化','MDD':'最大回撤','Sharpe_rf0':'Sharpe（rf=0）','IR':'相对沪深300 IR','total_return':'区间实际收益','year_return':'年度实际收益','RankIC':'日均RankIC','daily_RankIC':'日均RankIC','signed_RankIC':'按选股方向RankIC','original_raw_RankIC':'原始方向RankIC','monthly_RankIC':'月末RankIC','mean_two_sided_turnover':'日均双边换手','stale_position_days':'缺价持仓累计只·日','drawdown_start':'回撤峰值日','drawdown_trough':'回撤谷底','drawdown_recovery':'恢复日','IC_end':'IC末日','days':'交易日数'})
    return '<div class="scroll">'+d.to_html(index=False,border=0,escape=True)+'</div>'
def select(df,cols):return df[[c for c in cols if c in df]]
def plot_nav():
    fig,ax=plt.subplots(2,1,figsize=(11,6.5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    items=[('results_2015_2025','E0_baseline','Alpha158'),('results_raw_factors','cut_amplitude','Raw amplitude'),('results_raw_factors','cut_reversal','Raw reversal'),('results_replication','dbd_two_factor','Two-factor DBD (matched)')]
    for folder,name,label in items:
        p=ROOT/folder/f'{name}_portfolio.csv'
        if not p.exists():continue
        b=pd.read_csv(p,parse_dates=['date']);ax[0].plot(b.date,b.nav,label=label,lw=1.4)
        peak=np.maximum.accumulate(np.r_[1.,b.nav])[1:];ax[1].plot(b.date,b.nav/peak-1,lw=1.2)
    for a in ax:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
    ax[0].set_ylabel('Net NAV');ax[0].legend(fontsize=8,ncol=2);ax[1].set_ylabel('Drawdown');fig.tight_layout()
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);plt.close(fig)
    return '<img alt="费用后净值与回撤" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
def main():
    required=['results_2015_2025','results_with_reversal','results_raw_factors','results_replication','results_stacking_2021_2025']
    for f in required:assert (ROOT/f/'summary.csv').exists(),f
    verify=json.loads((ROOT/'verification_current.json').read_text());assert verify['completed_neural_models']==44
    raw=load('results_raw_factors','summary.csv');base=load('results_2015_2025','summary.csv');joint=load('results_with_reversal','summary.csv');nn=load('results_replication','summary.csv');stack=load('results_stacking_2021_2025','summary.csv')
    primary=['experiment','factor','RankIC','daily_RankIC','signed_RankIC','CAGR','MDD','Sharpe_rf0','IR','mean_two_sided_turnover','stale_position_days']
    body=['''<header><div class="eyebrow">项目一 · 2026-10-27 · 实际运行结果</div><h1>因子切割论与DBD-GRU<br>复现审计及年度滚动评价</h1><p>2009—2025历史数据 · 2015—2025独立年度测试 · 数据支持范围内复现</p></header>
<aside><b>结论边界：</b>本报告是沪深300历史成分范围内的部分研报复现与扩展，不是研报全市场、中性化、三因子融合、Barra约束指增的全面复刻。所有业绩来自包内实际预测和净值；未将旧版27.25%等指标移入本次结果。</aside>
<nav><a href="#scope">复现范围</a><a href="#data">数据</a><a href="#results">实际结果</a><a href="#annual">逐年结果</a><a href="#risk">风险与局限</a><a href="#replay">复核</a></nav>
<section id="scope"><h2>1. 已完成什么，不能复现什么</h2><div class="cards"><div><b>11</b>年度独立测试段</div><div><b>44</b>年度神经模型</div><div><b>2 / 3</b>可运行因子分支</div><div><b>5</b>下游堆叠测试年份</div></div>
<p>两类因子 × 普通GRU / DBD-GRU × 11年 = 44个模型。统一hidden=64、2层、最大100轮、早停20轮、IC损失；每个模型保存最佳检查点、预处理参数、验证损失及逐日样本外预测。没有以两轮冒烟测试替代正式训练。</p>
<p>主动买卖原始因子已修正为低状态组均值，但缺少小单主动买、卖金额而未运行。仅有全单主动买入额不能补齐该分支。三路DBD-Combine因此不可运行；两因子对称正交组合始终单独命名。</p>''']
    body.append(table(pd.read_csv(ROOT/'replication_matrix.csv'))+'</section>')
    body.append('''<section id="data"><h2>2. 数据来源、单位与覆盖</h2><p>行情来自Qlib社区2026-10-26发布包，已比对发布方SHA256；仅保留截至2025-12-31的行情、日历和相关历史成分区间。较早数据只供特征预热。使用当前供应商整理的历史数据，不等于持有每一天的历史数据版本。</p><p>成交金额保留Tushare原币值，单位千元，乘1000转换为人民币元。将其与真实Wind成交笔数相除得到每笔成交金额。金额不是成交笔数，复权成交量也没有用于冒充成交笔数。两供应商拼接是明确的数据来源差异。</p>''')
    audit=json.loads((ROOT/'amount_audit.json').read_text())
    body.append(f'<p>金额恒等关系检查：{audit["identity_checked_rows"]:,}条记录，99%分位相对误差{audit["relative_error_p99"]:.3g}。它验证单位和归一化一致性，不替代对供应商经济口径的独立审计。</p>')
    cov=pd.read_csv(ROOT/'coverage_2009_2025.csv');cov['成交笔数覆盖率']=cov.count_rows/cov.valid_price_rows
    body.append(table(cov)+'<p>早年成交笔数缺口显著，可能带来覆盖偏差。未填零、未插值，也没有把缺失证券说成完整覆盖。原始单因子独立样本结果与共同样本模型对照分别保存。</p></section>')
    body.append('''<section><h2>3. 时间、标签及训练协议</h2><p>首折：2009—2013训练、2014验证、2015测试；末折：2019—2023训练、2024验证、2025测试。训练标签必须在验证段开始之前成熟；验证标签必须在测试段开始之前成熟。模型只用验证集选轮次。</p><pre>信号：t日收盘\n成交：t+1日收盘\n标签：close(t+21) / close(t+1) - 1\n可用数据截止：2025-12-31</pre><p>训练按照研报每周最后交易日取20日序列；本实现神经验证也周抽样，日频生成样本外分数。研报未完整披露Adam学习率、缩放和梯度裁剪等细节，当前采用lr=0.001、训练期median/MAD、±5截断、梯度裁剪5，并将其标注为实现假设。与研报的5年窗口及2010—2025年5月样本范围不完全相同。</p><p>最后21个交易日没有完整20日未来标签，不参加IC；持仓估值仍延伸至2025年12月31日。2026年价格未用于本次标签或净值。</p></section>''')
    body.append('<section id="results"><h2>4. 原始因子：完整2015—2025</h2><p>原始因子按研报经济方向取负后用于高分买入，方向未依据测试结果选择；原始方向RankIC另存。Top50、月初换仓、目标95%投入，买5bp卖15bp。候选名单按前日信号固定；当天无法买入的部分留现金，不根据当天可成交信息补选其他股票。这不是研报约束指增组合。</p>'+table(select(raw,primary))+'</section>')
    body.append('<section><h2>5. Alpha158增量对照：完整11折</h2><h3>振幅可用共同样本</h3>'+table(select(base,primary))+'<h3>振幅＋反转共同样本</h3>'+table(select(joint,primary))+'<p>每张表内部基线和增量组采用共同样本。第二张表因要求成交笔数覆盖，股票集合可能不同，不应把两张表基线差异归因于模型改善。</p></section>')
    body.append('<section><h2>6. GRU、DBD与两因子组合</h2><p>下表采用各方法共同的证券—日期交集，包括原始因子、普通GRU、DBD和Alpha158预测。独立可用样本结果见available_ic_summary.csv。神经模型训练目标为正IC，因此其输出与原始因子报告的负方向不能不加说明地直接比较符号。</p>'+table(select(nn,primary+['monthly_RankIC','IC_end']))+'</section>')
    body.append('<section><h2>7. 严格上游样本外堆叠：2021—2025</h2><p>上游神经得分从2015年才成为样本外预测。下游再使用完整5年训练＋1年验证，首次独立测试自然是2021年。未用神经网络训练内拟合值填补2009—2014；不能宣称已完成2015—2025三路DBD堆叠。</p>'+table(select(stack,primary))+'</section>')
    body.append('<section><h2>8. 配对增量统计</h2><p>采用同日RankIC差值的移动块bootstrap（循环取块，块长21交易日）。区间跨0表示当前检验未给出稳定正增量证据；它不证明模型等效。重叠标签、模型选择与历史开发过程仍会影响统计解释。</p>')
    for f in ['results_replication','results_stacking_2021_2025']:
        d=json.loads((ROOT/f/'paired_tests.json').read_text());body.append('<h3>'+html.escape(f)+'</h3>'+table(pd.DataFrame([{'comparison':k,**v} for k,v in d.items()])))
    if (ROOT/'bootstrap_sensitivity.csv').exists():body.append('<details><summary>全部配对检验及21/42/63日块长敏感性（2000次）</summary>'+table(pd.read_csv(ROOT/'bootstrap_sensitivity.csv'))+'</details>')
    body.append('</section><section><h2>9. 净值与回撤路径</h2><p>不同曲线的可用股票覆盖可能不同，曲线用于观察风险路径；严格增量结论以前述共同样本表为准。</p>'+plot_nav()+'</section>')
    body.append('<section id="annual"><h2>10. 全部分年度结果</h2><p>年度实际收益与年化收益分开保存；整体CAGR由拼接净值计算，不能平均年度年化。年度回撤遗漏跨年过程，因此下方同时保留全程峰值、谷底和恢复日。</p>')
    for folder,title in [('results_2015_2025','振幅增量11折'),('results_with_reversal','双原始因子增量11折'),('results_raw_factors','原始因子'),('results_replication','神经因子共同样本'),('results_stacking_2021_2025','严格堆叠5折')]:
        years=load(folder,'yearly_portfolio.csv');body.append('<details><summary>'+title+'：逐年收益、回撤、IR</summary>'+table(select(years,['experiment','factor','year','total_return','year_return','CAGR','MDD','Sharpe_rf0','IR','stale_position_days']))+'</details>')
        body.append('<details><summary>'+title+'：逐年RankIC</summary>'+table(load(folder,'yearly_ic.csv'))+'</details>')
    body.append('<h3>全程回撤日期</h3>'+table(select(nn,['experiment','MDD','drawdown_start','drawdown_trough','drawdown_recovery']))+'</section>')
    body.append('''<section id="risk"><h2>11. 必须同时阅读的限制</h2><ul><li>缺历史行业、市值与Barra暴露，未实施研报中性化与约束优化。</li><li>无完整历史ST、上市日期、每日板块涨跌停价，交易使用9.5%对称代理，不能等同精确可成交回测。</li><li>缺价持仓沿用最后已知价格估值；未完整模拟退市清算、停牌恢复、排队成交、手数与容量。累计只·日不是发生缺价的交易日数。</li><li>股票复权收益与沪深300价格指数存在分红口径差异；IR是对该基准的日收益差计算，Sharpe假设无风险收益为0。</li><li>十分组输出是月末信号的20日远期收益均值，不是可交易多空净值，更不能将重叠20日标签按天累乘。</li><li>单随机种子2026；未做多种子参数量匹配，普通GRU与双分支GRU参数量不同。观察到差异不应全部归因于差分结构。</li><li>全段历史曾用于开发与比较，不称为完全未观察的最终留出集，更不称为实盘业绩。</li></ul></section>''')
    body.append('<section id="replay"><h2>12. 如何复核与重跑</h2><p>安装requirements_reproduction.txt中的依赖。在Windows conda环境中进入解压目录执行：</p><pre>python verify_data.py\npython -m unittest discover -s tests\npython test_timing.py\npython run_neural_grid.py --prepare-only\npython verify_current.py\n\n# 从数据重跑全部研究（CPU训练耗时较长）\npython reproduce_all.py --workers 2</pre><p>神经脚本默认复用已有完整检查点。若要从头训练，将results_neural_grid文件夹另存后移出项目目录，再执行重跑。低内存电脑使用--workers 1。sequence_cache与Alpha158缓存可由数据重建。</p><p>核对路径：预测文件→逐日组合收益→净值→分年表与总表。verification_current.json独立重算了费用、净值、CAGR、最大回撤，并重新加载全部44个神经检查点预测首个测试截面。</p>')
    body.append(f'<p>验证完成：{len(verify["portfolio_checks"])}组组合指标；{len(verify["neural_checkpoint_replay"])}个模型检查点回放。</p></section>')
    body.append('''<section><h2>13. 原始参考</h2><p><a href="https://www.fxbaogao.com/detail/4967322">开源证券：因子切割论与深度学习的结合应用（2025-07-26）</a>；<a href="https://finance.sina.cn/2025-07-27/detail-infhwpkc6944259.d.html">作者团队正文及表3、表4、附录</a>；<a href="https://github.com/chenditc/investment_data">Qlib社区数据与归一化源代码</a>；<a href="https://tushare.pro/document/2?doc_id=27">Tushare日线字段单位</a>。</p></section>''')
    style='''body{margin:0;background:#f3f5f8;color:#172337;font:15px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px}header{background:#142b4a;color:white;padding:44px;border-radius:14px}h1{font-size:36px;line-height:1.3;margin:12px 0}h2{font-size:23px;border-bottom:1px solid #dce3ed;padding-bottom:10px}h3{font-size:17px}section,aside{background:white;border-radius:12px;padding:26px;margin:20px 0;box-shadow:0 2px 10px #132c4810}aside{border-left:5px solid #d49b27;background:#fff9e9}nav{display:flex;gap:20px;flex-wrap:wrap;margin:20px}a{color:#286cc0}.eyebrow{color:#b9d1ed;letter-spacing:2px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards div{background:#edf4fb;padding:18px;border-radius:8px}.cards b{display:block;font-size:30px;color:#1f568b}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{background:#eaf0f7;text-align:left}th,td{padding:9px 11px;border-bottom:1px solid #e1e6ed}tr:nth-child(even){background:#f8fafc}pre{white-space:pre-wrap;background:#eef2f7;padding:18px;border-radius:8px}img{width:100%;height:auto}summary{cursor:pointer;font-weight:600;padding:12px;background:#eef3f8}details{margin:14px 0}li{margin:8px 0}@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}header{padding:22px}h1{font-size:28px}main{padding:15px}section{padding:18px}}@media print{body{background:white}main{max-width:none}header{background:#eaf0f7;color:#172337}section{box-shadow:none;break-inside:auto}nav{display:none}.scroll{overflow:visible}table{white-space:normal}details{display:block}}'''
    (ROOT/'reproduction_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>项目一研报复现与滚动评价</title><style>'+style+'</style><main>'+''.join(body)+'</main></html>',encoding='utf-8')
    print('WROTE reproduction_report.html')
if __name__=='__main__':main()
