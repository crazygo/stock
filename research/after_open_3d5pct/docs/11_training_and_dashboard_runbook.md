# 11 · 训练与看板启动手册

本文件只对应旧 `manual_three_cutoffs_v1` 看板的当次交付。后续每小时协议已另行预登记并完成受控训练；其启动、一次运行和成熟评估命令见 [15 · 每小时一次运行手册](15_hourly_once_runbook.md)。

本手册分开“只读查看既有结果”和“未来启动新训练”。本次看板交付只执行前者；启动新模型训练必须有新的预登记实验单和输出目录。

## 只读查看本次看板

从仓库根目录检查工作树和现有服务，保护无关 WIP：

```bash
git status --short --branch
lsof -nP -iTCP:8768 -sTCP:LISTEN
```

先核对 `research/after_open_3d5pct/runs/market_dual_track_pilot_20260925/` 的 `config.json`、`manifest.json`、`predictions.parquet`、`trials.jsonl` 等文件完整。使用研究虚拟环境运行只读导出，输出目录必须不存在：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.export_dashboard_v1 \
  --source research/after_open_3d5pct/runs/market_dual_track_pilot_20260925 \
  --output research/after_open_3d5pct/runs/research_dashboard_policy_v1_20260925 \
  --threshold 0.35
```

成功后打开 [本地研究台](http://127.0.0.1:8768/research/after_open_3d5pct/dashboard/index.html)。仓库现有 `serve_preopen_dashboard.py` 在 8768 服务仓库根目录；**先查进程，不要为看板重启/重复启动它**。若没有服务且确需临时查看，可从仓库根目录运行 `python3 serve_preopen_dashboard.py`，不要占用已有端口。若选择其他新输出目录，在地址附加 `?run=<新输出目录名>`。导出文件位于 `runs/`，本地大制品和行情不进 Git。

检查 `EXPORT_MANIFEST.json` 的输出文件 SHA；页面内的来源和源级剔除审计来自 `overview.json`。源行情与首轮 manifest 不一致时导出直接失败，应调查数据修订，不能关闭哈希检查去拼接新旧源。上次失败生成的半成品目录只可在核对路径属于此次导出后删除，再用同一命令或换新目录重试。

## 未来新训练：准备与冻结

1. 阅读工程 `AGENTS.md`、README 和 01–10；在 `templates/experiment.md` **先**登记新 run ID、数据快照、已暴露区间、特征/标签版本、时点、fold、候选超参、资源上限、校准、阈值选择与独立评估门槛。新点子先去 12，不自动进入本轮。
2. `git status --short --branch`，核对源码/配置改动、依赖版本。执行 README 的合成测试与小规模工程检查。数据 manifest 显式列出本地行情、日历、股票池、公司行动 SHA；训练期间不联网、不调用交易接口、不自动推 R2。
3. 历史当前名单回溯和 `available_at=bar_end+1 秒` 只能探索；正式独立验证前补 PIT 股票池、真实到达时延、夜盘覆盖和新的前向时期。所有已看过的 2026 月份继续标 exposed。
4. 复制 `configs/pilot_v2.json` 到**新配置文件**并填新实验 ID/新输出目录。目标/标签修改则升版本；输入改变则升 feature schema；不能让旧配置或旧 run 的语义漂移。

## 未来新训练：启动、日志、失败

首轮固定配置训练入口供复现和新实验模板参考；不要覆盖首轮目录：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_pilot_v2 \
  --config research/after_open_3d5pct/configs/<new-config>.json \
  --output research/after_open_3d5pct/runs/<new-run-id>
```

`<...>` 是占位符，不可原样复制。完整正式训练还须先落实 07 的有限超参/时间校准/独立验收规则；此入口仍是 pilot，不能仅因跑通便发布模型概率。新日志代码会记录 LightGBM 内层验证逐轮曲线及累计秒数、TCN 每 epoch loss/Brier/当轮与累计秒数；外层指标在训练结束后单独评分，不能反向选择最佳轮次。旧 2026-09-25 run 没有这些新增日志。

运行中核查 `trials.jsonl`：每个 fold × 模型 × 输入组都应有 completed/failed 及原因，成功项应有关联模型文件；`metrics.csv`、`predictions.parquet`、`manifest.json` 的数量和哈希要一致。失败 trial 保留，不删掉重试痕迹。缺失行情/标签、内层样本不足、GPU 内存或依赖错误须记录后停在新 run；修复后开新 ID，禁止覆盖旧运行伪装成同一试验。模型复现至少核对配置/源文件 SHA、种子、依赖、fold、逐条预测和指标，容忍平台数值误差须先声明。

训练完成后先复核因果时间、标签成熟、覆盖分母、拆股与缺失处理，再导出新看板。若改变策略阈值，只重跑独立 policy 并另存目录；不能据外层评分挑一个最漂亮阈值后说它通过了独立验证。阈值、校准和退出规则应在未见独立评估之前冻结，并在未来纸面数据上核验。
