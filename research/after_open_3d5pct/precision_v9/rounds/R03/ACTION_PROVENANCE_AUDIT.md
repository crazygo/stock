# R03 公司行动来源核对

2026-09-27，只读核对 v8 冻结快照，未发出新网络请求，也未改公司行动数据。读取 `runs/focus_v8_20260927/source_provenance.json` 的 109 个行动文件登记，分别定位原文件、冻结文件和登记 SHA-256。

109/109 的原文件哈希、冻结文件哈希和 v8 登记哈希相等；109/109 均能在相应下载器的 `get_rehab` 审计中找到成功且行数相同的记录。原 102 个来源使用仓库 `market_data/manifests/request_audit.jsonl`；新增 7 个光通信来源使用 `research/strategy_group_lab/runs/optics_3round_20260926/market_data/manifests/request_audit.jsonl`。因此，只查共享 warehouse 会误报新增证券的缺失。

其中 23 个冻结文件为空表，均有 provider 成功返回零行的证据；例如 LITE 的返回发生于 2026-09-24T19:09:36.520019Z。空表不是抓取失败，也不能仅凭成功状态证明外部供应商历史事件穷尽。审计记录本身未保存每次响应内容哈希，故“同股票成功行数匹配”和“当前原始文件哈希匹配冻结来源”是两层证据，不声称拥有逐响应的强绑定。

R03 继续使用这套冻结行动数据以保持对照，记录 `event_list_only_historical_coverage_uncertified` 边界。新 builder 的实际拆股排除覆盖全部有效输入回看窗口；如跨期价格或数据质量审计另发现疑似公司行动，应保留该行异常并单独补证，不依赖修改复权价格或删除失败标签来提高命中率。
