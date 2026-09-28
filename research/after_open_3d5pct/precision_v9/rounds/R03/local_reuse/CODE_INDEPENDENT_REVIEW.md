# R03 本地复用代码独立审查

2026-09-27。依据已采用 `REGISTRATION_DRAFT.md`（SHA `0b60e51b5ea57a968480ea3e12b53228398435197b05d6abfc717f695200fd56`）和 `ADOPTION.md`（SHA `2a29d34ee6d869dab9442fc70848cb86d9b8f8cc785158ede3dc4e3cd010d824`）。本审查仅新本地复用代码及prepare分派；不触碰另一采集任务、锁、等待器、旧run或并行数值工作。

状态：**初审问题已修复；稳定源码的独立离线来源/兼容门禁验证通过，可采用此代码版本。真实采集、545来源验收与全池性能未验证；启动仍由根核验现等待器状态后决定。** 根已显式接受P1重复重放的有界成本偏差，本轮不另开缓存优化；不能称“一次重放”或全池性能已实现。

## 已反馈的具体问题

| 项目 | 初始WIP问题 | 必需修复/验证 |
|---|---|---|
| R1 旧v1空表门禁 | 新tier例外同时用于recovery_v1，旧空月可把tier改local_verified_provider后跳过raw零响应验证 | 新来源例外仅限新version；旧v1伪装tier＋空audit/缺raw必须拒绝 |
| R2 原完成片优先 | 新prepare未强制原complete片使用原链接；新local fixture替换原完整月竟预期通过 | 原complete必须合法继承；即使local数据相同也拒绝替换来源；resume同样检查 |
| R3 执行依赖冻结 | 新identity只列5文件，遗漏实际normalize_kline_dataframe及其他执行依赖 | 登记/快照/门禁覆盖完整标准化链及新prepare等；篡改依赖应拒绝 |
| R4 失败证据保留 | freeze失败时删除staging | 保留失败来源与错误状态，后继run处理，不能删除失败证据 |
| R5 分页与receipt | 只聚合成功页，未拒绝终页后事件、倒序attempt、naive时间；dropna证券集合漏过null code | 逐事件有限状态顺序、有效带时区时间、每行证券一致；定向反例拒绝 |
| R6 导入审计 | committed验证未核新part audit的bundle、manifest、receipt及imported_at | audit=[]或篡改实际来源/时间不得接受 |
| R7 R2兜底 | 新run中R2异常直接终止，原恢复器记录异常后可走Futu | 保持已登记三级fallback并保留失败审计；不隐式改变采集失败策略 |
| R8 实际5m时间栅格 | 仅parse时间与范围，额外off-grid记录可能不破坏RTH缺条统计 | 原始time_key严格源格式、秒为0且分钟整除5，带偏移字符串/09:36等必须拒绝 |
| P1 重放成本 | 请求组在freeze、逐月命中与prepare重复整段normalize，五个月典型约14次 | 正确性优先；若不优化，明确实测与“只复制一次、重复重放”偏差，不虚称只重放一次 |

## 稳定后离线验证范围

本范围在执行前固定，不扩大采集或模型计划：运行现有 `test_local_reuse`、`test_acquire_recovery`、`test_data`；独立补查旧v1伪装tier、原完整片替换、part audit篡改、失败冻结证据、错误分页/无时区receipt/null code。使用合成fixture，禁止R2/OpenD/真实锁；测试临时文件限定本目录独立临时子目录。验证前后核被审源码SHA，防测试与交付版本不一致。旧代码不能为通过测试被放宽。

若定向检查出现新问题，先反馈再修复，保留初次失败记录；不通过修改既有登记或隐藏失败来签通过。真实545来源齐备、空覆盖及prepare配对验收仍属于未来运行阶段。

## 最终独立执行结果

执行于2026-09-27 13:30:39–13:30:41 UTC；详细结果 [INDEPENDENT_VALIDATION.json](INDEPENDENT_VALIDATION.json)，原测试输出 [INDEPENDENT_VALIDATION.log](INDEPENDENT_VALIDATION.log)。套件28项通过，耗时2.412秒；加3项独立控制流检查总计2.623秒。受审执行源码前后SHA全同。

- 旧recovery_v1空月改tier为local_verified_provider、audit清空并删除raw后，仍被`empty provider reply not audited`拒绝。
- 新入口mock R2异常时保留R2错误审计，确实调用一次合成Futu页stub完成该合成月；没有真实R2/OpenD请求。
- 新入口遇PendingSource时返回incomplete_unresolved，R2对象与Futu页stub调用均为0；没有用网络掩盖未提交来源。

全部测试置于禁止真实OpenQuoteContext、R2和flock的保护下。两项run级检查仅把锁文件换为本目录临时fixture内的fake.lock并mock锁函数；未打开真实共享锁。测试临时写区为本目录`independent_fixture_tmp`，fixture已自行清理。审查未修改代码，未扫描外部全库新状态，未操作22976或其他任务。

R1–R8在稳定实现中均闭合：旧v1空tier例外隔离；新version原完成片强制继承；十项执行源码及完整normalizer链登记/快照/复核；失败staging保留；分页/attempt/实际receipt/每行证券与时间栅格严格核验；part导入audit与bundle/receipt一致；R2正常兜底。合并/修订来源仍安全拒绝，未放宽为自动选新值；真实修订出现后需另行来源审查。

新来源成功零月与同一多月请求中的零月可通过专属门禁，保持unknown覆盖与rth_complete=false；metadata-only仍不能冒充provider-empty。原2025五空＋2026非空的短历史分支仍保留，2025/2026全空仍拒绝。这不证明当前HONA/SPCX十片已有provider空证据。

## 稳定代码指纹

| 文件 | SHA-256 |
|---|---|
| prepare.py | `83094884e6465ffa2310c930664e813107fadfe1ef55a0c58c931ee9993d3d01` |
| acquire_local_reuse.py | `aeb8f29ea72d6d510f062166806dc5b2ecf0a3ad649f6e1fe016d76a326b60a6` |
| local_reuse_source.py | `f25a9a7adc6d713fb1809f3b747f86f814a63d830f8fc8b4c09084fb03455b9a` |
| test_local_reuse.py | `35f7d8d0ad483b036af333a6ef03c5be155bed71d23b7fe3855d0dfc799008e2` |
| 原acquire_recovery.py，未改 | `6b4e056492dc05dad7a6a56b358dc6dbcb463ee91b7c6e621bd32a831f735627` |

其余执行依赖逐项SHA在INDEPENDENT_VALIDATION.json；Sol交付说明CODE_REVIEW.md SHA为`211b929b678ec411c1eaa7320256b1d1bf669b1f7577f2f927a365328a4c58f1`。

## 交接与保留边界

独立结论支持根继续完成启动前核验，不能替代该核验。根已明确接受P1重复重放边界，下一步按ADOPTION顺序确认并结束自己仍在等待的22976、保留旧初始化证据，随后另记新run身份与启动记录；若22976已经采集，按实际checkpoint另审，不能按未请求假设中断。

新源每个请求bundle只复制一次，但五个月请求仍可能在freeze/逐月使用/prepare中约14次整段normalize。未做全池实测，不承诺本地阶段时长，不以2.623秒fixture性能外推真实数据性能。实际计时是**首次网络阶段开始后的180分钟墙钟上限，含其间插入的Local验真时间**，不是纯累计网络时间；首次网络前的Local阶段不在该计时内。本地验证耗时另记。根接受这一更保守的计时边界，并另设20分钟观察检查点；后者是根的运行观察，不是程序硬超时。单worker和no_upload保持；全545数据状态、十空月证据、配对2026与实际manifest仍是后续工作，未运行训练、未改变目标或宣布90%通过。
