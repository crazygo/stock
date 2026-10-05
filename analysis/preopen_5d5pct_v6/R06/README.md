# 富途历史信息增量 R06

本轮将实际取得的四族历史字段加入原五日+5%任务；旧R00–R05、v5及十月资格保留。预登记见PROTOCOL.md，接口结案见API_AUDIT.md，训练与完整失败条件见REPORT.md / REVIEW.md。零新增付费，不接下单，不上传R2。

```bash
python3 analysis/preopen_5d5pct_v6/R06/probe_futu.py
python3 analysis/preopen_5d5pct_v6/R06/probe_extra.py
python3 analysis/preopen_5d5pct_v6/R06/probe_subscription.py
python3 analysis/preopen_5d5pct_v6/R06/capture_focus.py
python3 analysis/preopen_5d5pct_v6/R06/collect_history.py
python3 analysis/preopen_5d5pct_v6/R06/quality.py
python3 analysis/preopen_5d5pct_v6/R06/test_features.py
python3 analysis/preopen_5d5pct_v6/R06/features.py
python3 analysis/preopen_5d5pct_v6/R06/training_coverage.py
python3 analysis/preopen_5d5pct_v6/R06/train.py --fit --workers 2
python3 analysis/preopen_5d5pct_v6/R06/finish.py
python3 analysis/preopen_5d5pct_v6/recommend.py --round R06 --top 3
python3 analysis/preopen_5d5pct_v6/server.py --port 8771
```

运行环境：现有Python、Futu OpenD 127.0.0.1:11111、关闭协议加密。采集串行、分页有上限，独立worker有240秒上限；既有完成文件按SHA和元数据保留。原始响应JSON/Parquet、价格缓存、pickle、运行日志全部本地忽略，不进Git。统一推理优先数据格式模型并核对SHA。

复算依赖原R03 124股面板及冻结价格/公司行动缓存；面板父SHA登记在coverage.json。每块训练、候选校准、第一名校准、阈值选择与评价按标签成熟隔离。重做必须建立新的实验版本或完整归档本地cache/models，不能在已有月度meta下更换面板；默认运行只恢复未完成任务。

八输入T0/TC/TF/TO/TV/TS/TA/TL各训练五月至九月五个月，候选校准与额外第一名校准各自统计。概率分箱及Brier在calibration_audit.json；40模型复算在verification.json；T0/H3等价与源日期边界在control_verification.json；最新月八模型的原生等价在portable_verification.json；新增字段在各训练块的实际覆盖在training_coverage.json。

资金流只实取最近一年，归一化分母须是真实同日完整常规盘turnover且价格与冻结OHLCV一致；缺失保留。历史标的期权统计分两个≤364日窗口与分页；OI不前移。卖空成交不等于净空头持仓。宏观、财报实际/预期及评级虽然能返回数字，原始公布与修订版本未认证者明确排除。

历史日级输入统一统计日后两官方交易日才进入开发特征，并固定五日延迟敏感性；这些延迟是假设，不能认证历史PIT。股票池同样是当前成员快照回溯。全部历史已暴露，最终独立效果仍须真实冻结后的新数据。

独立wireframe页面：http://127.0.0.1:8771/futu_data.html。日期、两年mini、全日5m路径、至少6px、禁止鼠标/触控缩放、成功/失败/未知及完整五日目标阴影沿用契约。统一命令给新模型前三候选、概率估计、目标价、截止、历史门槛与当前空门槛、弃权原因；最近完整日参考不冒充当前信号。
