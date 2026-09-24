# 连续上涨区间：路径形态分解与条件转移概率研究

对应父任务"基于序列预测后续为连续上涨区间的概率"。
任务理解与范围界定见 `scope.md`。文件夹名 `uptrend_path_research` 是占位名。

## 现在的进度

**M0 已完成**：分段算法 + 交互调试看板 + 25 项单元测试 + JS/Python 交叉校验。

```bash
# 跑测试（25 项）
cd analysis/uptrend_path_research && python3 -m unittest test_segments

# 重建调试看板
cd /Users/admin/Code/stock && PYTHONPATH=. python3 analysis/uptrend_path_research/build_debug_dashboard.py

# 校验看板 JS 与 Python 参考实现逐段一致（8 组参数）
PYTHONPATH=. python3 analysis/uptrend_path_research/crosscheck_js_py.py
```

看板地址：http://127.0.0.1:8768/analysis/uptrend_path_research/debug.html
（服务：仓库根目录 `python3 serve_preopen_dashboard.py`）

## 文件

| 文件 | 内容 |
|---|---|
| `scope.md` | 任务理解、区间定义、待定参数、明确不做的事 |
| `segments.py` | 分段算法参考实现（Python） |
| `test_segments.py` | 25 项单元测试，期望值全部手工推导后验证 |
| `debug_template.html` / `debug.html` | 交互调试看板，分段在浏览器里实时跑 |
| `crosscheck_js_py.py` | 校验 JS 与 Python 分段结果逐段一致 |
| `build_debug_dashboard.py` | 生成 debug.html |

## 三个参数，三种分工（不要混）

| 参数 | 角色 | 会不会移动支点 |
|---|---|---|
| `delta` | **唯一结构参数**。回撤容忍度 θ = δ·ATR，决定"这一轮上涨什么时候算结束" | ✅ 会 |
| `min_amp` | 纯资格标记。段结束后才看"这段够不够大" | ❌ 不会 |
| `min_bars` | 纯资格标记。时长不足只打标 | ❌ 不会 |

**目标涨幅（如 8%）不是本模块的参数**，它属于预测标签（三重障碍），不参与分段。

为什么不合并小段：删一个段就要删它的一个边界支点，而相邻支点同类（都是峰或都是谷），
交替性立刻破掉，合并出的段不是单调段。要合并就把 δ 调大——等价且诚实。

## 两套分段，两种用途

| | 标签分段 | 因果分段 |
|---|---|---|
| ATR 口径 | 全样本中位数 | 滚动 20 根 |
| 能否当时可知 | ❌ 含未来信息 | ✅ 每个支点都是"反转确认"时才产生 |
| 用途 | 生成因变量 | 生成特征 |

因果分段有一条硬保证：**支点序列随时间单调只增不减，不会改写历史**
（`test_pivots_grow_monotonically_with_trailing_atr`）。这条是特征可用性的前提。

## 已知数据事实

- 小时线 `time_key` 是**美东 bar 结束时刻**（已用日线 OHLCV 反解验证）。
- SNOW / SPCX / TSM 在 point-in-time 资格门槛下没有合格股票日，实际有效股票池 18 只。
- 18 只 × 约 4,500 根小时 bar = 81,513 根，已全部嵌入 debug.html（7.6 MB，可离线打开）。

## 下一步（M1）

δ 三轴敏感性、边界例与被拒例抽样、删失规则定稿、人工核验。

## 当前冻结规格

`spec_v1.md`（含修订 A）。要点：

- 主档位 **+5%**，4/6/7/8% 为敏感性对照；档位已冻结，看完 M4 结果不得再改。
- 观测点 = 空仓状态的候选入场点；实盘为 `FLAT → BUY → EXIT(+5%) → FLAT` 顺序状态机。
- 入场价 = 下一根可交易常规时段 bar 的 Open（基准模拟入场价，非成交保证）。
- 主标签 = 3 个交易日内最高价是否触及 +5%；收盘价为敏感性对照。
- M4 第一版只用可解释规则，不上 ML；Holm 校正 + 配对时间块 bootstrap。
- 第一轮不输出概率与可靠度分数。

## 已跑出的结果

`run_baselines.py` → `baselines.json`（Q1 五档基准率、episode 对账、障碍先后顺序）。
