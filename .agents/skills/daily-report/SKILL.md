---
name: daily-report
description: Standardized daily report methodology, guidelines, and format for AI coding agents. Enforces hierarchical task tree in YAML, explicit outcome accounting, in-depth critique and pitfall reflections, actionable unfinished task roadmaps, and fixed Option A (Tree & Outcomes Cockpit) wireframe UI standard.
---

# 日报规范与方法论指南 (Daily Report Methodology & Skill)

> 本规范面向所有在本项目中独立或协同工作的 AI Coding Agents 与开发者。日报不仅是工作进度的备忘录，更是交付物审计、逻辑纠偏复盘与人机认知对齐的最高级别事实契约。

---

## 一、 核心指导思想 (Guiding Principles)

1. **真实性高于修饰性 (Truth Over Polish)**
   - 严禁任何形式的报喜不报忧或模糊措辞。
   - 成功的实证要给出量化证据；被推翻的假设、失败的代码、被用户批评的缺陷要完整呈现，绝不遮掩。
2. **因果复盘与底层思考 (Events + Thinking Behind)**
   - 记录“踩坑与被批评”时，**不仅要客观陈述发生了什么事实，更要深度剖析用户反馈背后的真正目的与量化/业务逻辑**。
   - 用户批评的往往不是表面语法，而是底层的业务契约（例如交易因果性、未来函数穿透、实盘可落地性等）。必须提炼出可被其他 Agent 复用的系统性避坑规则。
3. **结构化与机械可追溯 (Machine-Readable Structure)**
   - 任务必须使用结构化 YAML 明确父子层级关系，并分配全局唯一任务编号（如 `T-01`, `T-01-01`）。
   - 每一个任务产出必须能通过任务编号追溯至具体的代码文件、数据集、指标或文档。
4. **状态与边界明确 (Clear Task States & Boundaries)**
   - 未竟任务必须清晰声明状态：`进行中 (IN_PROGRESS)`、`暂停 (PAUSED)`、`阻塞 (BLOCKED)` 或 `已废弃 (ABANDONED)`，并阐明下一阶段的具体处置与边界。

---

## 二、 命名与存储规范 (Naming & Path Conventions)

- **存储目录**：统一归档在项目根目录下的 [`reports/daily/`](file:///Users/admin/Code/stock/reports/daily/)。
- **文件命名格式**：
  - Markdown 版本：`{yyyy-mm-dd}-{agent-cli}-{agent-name}.md`
  - HTML Wireframe 版本：`{yyyy-mm-dd}-{agent-cli}-{agent-name}-wireframe.html`
- **命名示例**：
  - `2026-09-29-agy-main.md` （2026年9月29日，Antigravity 主 Agent Markdown 日报）
  - `2026-09-29-agy-main-wireframe.html` （2026年9月29日，Antigravity 主 Agent HTML 原型日报）

---

## 三、 日报标准模板与内容结构 (Standard Content Structure)

每个 Agent 撰写日报必须严格遵循以下四大核心模块：

### 模块 1：任务层级树 (Task Hierarchy in YAML)
- **要求**：使用清晰的 YAML 树形结构表达今天承接的全部任务，区分父任务与子任务，且每个任务都必须具有全局唯一编号（`id`）、任务描述（`name`）、以及状态（`status`: `DONE` / `IN_PROGRESS` / `PAUSED`）。

### 模块 2：任务成果明细 (Task Outcomes & Deliverables)
- **要求**：与模块 1 的任务编号一一对应。每一项写明具体的交付产出，必须附带：
  - 核心量化结论与实证数据；
  - 交付文件路径（点击可达的绝对链接形式：`[file.py](file:///path/to/file.py)`）；
  - 关键指标对比。

### 模块 3：踩坑复盘与用户反馈深度思考 (Pitfalls & Reprimands Reflection)
- **要求**：这是日报的核心灵魂部分。对于每一个踩坑或被用户纠正的问题，按三层结构剖析：
  1. **发生的事实 (The Event)**：发生了什么代码 Bug、认知偏差或设计缺陷？
  2. **用户批评与纠偏背后的深层目的 (Underlying Intent & Thinking)**：用户为什么指出这个问题？背后的量化实战逻辑、商业逻辑或交易本质是什么？
  3. **系统性规则提炼 (Systemic Takeaway)**：如何将本次教训沉淀为全局永久规则，避免其他 Agent 重复犯错？

### 模块 4：未竟任务规划与状态追踪 (Unfinished Tasks & Roadmap)
- **要求**：明确列出未完成任务，明确区分状态（进行中 vs 暂停），给出规划：
  - **进行中任务 (IN_PROGRESS)**：下一步具体怎么做、预计交付物；
  - **暂停任务 (PAUSED)**：为什么暂停（如触及用户边界约束、等待外部条件成熟等）。

---

## 四、 界面与 Wireframe 交互交付规范（固化方案 A：任务树审计台）

> ⚠️ **强制规约**：当用户或任务要求提供 HTML / Wireframe 版日报时，**严禁使用单栏流水账或自由发散布局，必须统一采用经过用户核准的【方案 A：任务树审计台 (Tree & Outcomes Cockpit)】**。

```
┌────────────────────────────────────────────────────────────────────────┐
│ [WIREFRAME] Daily Report · {Date}-{CLI}-{Agent}   [方案 A 任务树审计台]│
├────────────────────────────────────────────────────────────────────────┤
│ [KPI 概览条] 今日任务总量 | 已交付完成率 | 核心产出指标 | 重大纠偏反思数│
├───────────────────────────────────┬────────────────────────────────────┤
│ 🌲 任务层级树 (YAML Hierarchy)    │ 📦 任务交付成果清单 (Task Outcomes)│
│ tasks:                            │ • T-01: 交付物路径 + 核心量化证据  │
│   - id: "T-01"                    │ • T-02: 交付物路径 + 核心量化证据  │
│     subtasks: ...                 │ • T-03: 交付物路径 + 核心量化证据  │
├───────────────────────────────────┴────────────────────────────────────┤
│ 🚨 踩坑复盘与用户反馈深度思考 (三层剖析: 事实 ➔ 用户深层目的 ➔ 系统规则)│
├────────────────────────────────────────────────────────────────────────┤
│ 🗺️ 未竟任务规划与边界追踪 (进行中 IN_PROGRESS vs 严格暂停 PAUSED)      │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 为什么固化方案 A？
1. **左右对照，证据闭环**：
   左栏以标准的 YAML 代码块呈现机器可读的任务树与状态；右栏以表格形式逐项对应任务编号、量化成果与点击可达的文件链接，彻底杜绝“有任务无产出”或“有产出无归属”的悬空漏洞。
2. **高信息密度，杜绝滚动疲劳**：
   左右分栏充分利用宽屏视口，审计者无需在页面上来回滚动寻找任务与产出的对应关系。
3. **主次分明，认知沉淀置底**：
   将任务成果放在第一屏（左树右表），保障验收效率；将 3 层踩坑剖析（事实 ➜ 用户深层目的 ➜ 系统性规则沉淀）作为底层大板块展开，既不喧宾夺主，又保留了极高的风控复盘价值。

### 2. 界面视觉规范 (Low-Fidelity Contract)
- **纯灰阶色彩**：仅使用 `#111827`、`#4b5563`、`#d1d5db`、`#f3f4f6`、`#ffffff`，不引入任何品牌色调或渐变；
- **系统排版**：采用默认系统字体族，数字、代码与任务编号使用 `ui-monospace, Menlo, Consolas` 等宽字体；
- **零外部网络依赖**：单文件自包含（Inline CSS），不加载任何外部 CDN 资源，确保本地离线秒开。

---

## 五、 范式检查清单 (Agent Self-Check Before Submission)

提交日报前，Agent 必须在后台静默完成以下检查：
- [ ] 任务是否具有父子层级且用标准的 YAML 块声明？
- [ ] 任务编号是否在成果部分一一对应？
- [ ] 踩坑部分是否深入挖掘了用户批评的底层逻辑，而非流水账应付？
- [ ] 未做完的任务是否明确标记了【进行中】或【暂停】，并说明了原因？
- [ ] 若生成 HTML Wireframe，是否**严格采用方案 A（任务树审计台）**，且无冗余的多方案切换器？
- [ ] 文件命名是否符合 `{yyyy-mm-dd}-{agent-cli}-{agent-name}.md` 与 `-wireframe.html`？
- [ ] 文件内所有的文件路径是否均使用标准的文件协议链接？
