# 真实数据评测与复现

以下命令使用训练导出的 checkpoint。数据根目录只放公开发行物、用户已获授权的工业数据及其登记表。输出目录按 checkpoint 和协议分开。每条结果记录输入或登记表 SHA-256。预测前固定候选集、根因定义、时间范围和校准阈值。

~~~bash
python -m pip install -e . statsmodels matplotlib tqdm
python -m shifted_attention.evaluation prepare --data benchmarks --assets tsad chamber msds
python -m shifted_attention.evaluation timercd --checkpoint best.pt --data benchmarks --output results/timercd
python -m shifted_attention.evaluation tsb --checkpoint best.pt --data benchmarks --output results/tsb
python -m shifted_attention.evaluation rca --checkpoint best.pt --data benchmarks --ledger benchmarks/rca/chamber.json --output results/chamber
python -m shifted_attention.evaluation rca --checkpoint best.pt --data benchmarks --ledger benchmarks/rca/msds.json --output results/msds --threshold "$SOURCE_THRESHOLD"
python -m shifted_attention.evaluation rca --checkpoint best.pt --data benchmarks --ledger benchmarks/rca/msds-direct-service-logs.json --output results/msds-services --threshold "$SOURCE_THRESHOLD"
~~~

## TimeRCD 与 TSB-AD

TimeRCD 表 1 与附录 D 表 8–9 给出 16 个来源。按作者提供的 TSB-AD 两个 ZIP，从文件名中的来源精确选择。读取官方压缩包目录逐项核对后得到以下数目。适配器发现任一数目不符即停止。TimeRCD 和 TSB-AD 官方原始评测程序均按 CSV 去掉缺测行，保存的预测包含保留行对应的原始行号。[TimeRCD 全文](https://proceedings.mlr.press/v306/lan26b.html) · [TimeRCD 固定评测实现](https://github.com/thu-sail-lab/Time-RCD/tree/768bb4e2b019280bd47eb116ef976ca1b8360b0f) · [TSB-AD 发行物](https://github.com/TheDatumOrg/TSB-AD)

| 面板 | 来源及序列数 | 总数 |
|---|---|---:|
| 单变量 | UCR 228、NAB 28、YAHOO 259、IOPS 17、MGAB 9、SED 3、Stock 20、TODS 15、NEK 9、Power 1、WSD 111 | 700 |
| 多变量 | MSL 16、PSM 1、SMAP 27、SMD 22、SWaT 2 | 68 |

**TimeRCD 测试协议。** 从文件名解析训练前缀长度，在保留行序列上使用前缀拟合鲁棒归一化，仅对后缀计分。使用模型在每个滑窗输出的异常 logit，重叠处平均有效 logit 后取 sigmoid。无证据行记零并计入分母。默认采用论文固定的 5,000 行上下文。步幅 2,500 和重叠 logit 聚合是本模型的推理选择，可通过参数改动并随结果存档。对比本模型预注册的 2,048/1,024 协议可另传 --width 2048 --stride 1024、使用独立输出目录。

直接调用 TimeRCD 固定版本的四个核心指标函数。VUS-PR 采用 250 个阈值和作者 runner 的周期选择，Standard-F1、Affiliation-F、F1_T 均沿用发布实现的阈值扫描。阈值扫描读取测试真值，结果仅供论文基准比较。连续告警评估使用独立校准集阈值。另外保存作者 get_metrics 所报告的 AUC/VUS ROC 与 PR、标准 precision/recall、PA-F1、Affiliation 和 F1_T 的 precision/recall。summary.json 包含逐来源均值、每来源等权均值和逐序列加权均值。输出记录协议、上下文与库版本提交。指标失败会中断，先行保存逐点分数供诊断。

**TSB-AD 官方面板。** 固定 350 条 U-Eva 和 180 条 M-Eva，同时验证与 48/20 条 Tuning 清单无交集。对全记录计分，使用首列计算滑动周期，直接调用固定 SHA 的官方 get_metrics。模型许可使用文件指定的参考前缀拟合归一化，零长度前缀使用仅历史统计。默认 2,048/1,024，各 U/M 内序列等权。TimeRCD 与 TSB 各自沿用作者的指标实现，尤其是传给周期估计的形状。[TSB-AD 固定实现与清单](https://github.com/TheDatumOrg/TSB-AD/tree/303dffbfc11612995b17b06633f8c9d0ea0e38e5)

发布数据上的标签审计覆盖全部 768 条序列。有 169 条的指定训练前缀含阳性，有 6 条的测试后缀仅含负类（NEK 1 条，YAHOO 5 条）。每条仍生成完整后缀预测。主表对单类后缀标为 `all-negative` 或 `all-positive`，各指标提供有限分母；另附固定上游库在这些输入上的实际返回值或异常，以及按该库约定汇总的独立结果。表中 `reference_positive_rows` 仅用于披露对原始前缀标签的事后核验。TimeRCD 默认 `--metric-engine fast`，使用发布程序的 affiliation 网格与 800 个 F1_T 分位，`--metric-engine default` 切至其另一套公开阈值扫描。源码及官方清单从固定提交的文件树提取，重新计算并核对 30 个 Git blob 对象指纹。仅安装该部分可使用 `prepare --assets metrics`。

## 根因定位文献与计分单位

### 可执行的真实数据对照

**原版模型的显存与复核。** 两套原始多变量实现会为每层构造形状为 [B,H,L,L] 的通道偏置及注意力分数，其中 L = 通道数 × ceil(窗口行数 / 16)。实际执行把独立的多变量窗口按物理 batch=1 投递；论文或 TSB 配置中的 64 继续作为原始批大小记录。L 至多 2048 时使用作者原始注意力。更长的窗口在一次计算中保留完整 K/V 与完整掩码，依次处理不超过 128 个 Query。每行计算全部 L 个键的通道偏置、同一 softmax 和加权和，再应用原输出层；同一 score 临时张量上限 512 MiB。可用 --author-query-rows 调整查询分块。每文件的新结果在 inference_execution 记录实际物理批大小、各窗完整键数与查询块大小。旧版已完成的密集参考结果可沿用。

在真实 SMD 数据和原作者多变量权重上，取 237 行、38 通道，形成 3 个独立 96 行窗口，其中最后一窗含补齐。比较作者原始批处理、原始注意力单窗和每块 37 行 Query 的单窗输出。论文 API 及 TSB 移植在 CPU FP32 上三组逐点概率的最大绝对误差均为 0。两套各 8 层、118 个命名参数在替换计算路径后保持张量存储与版本一致。可设置 AUTHOR_BASELINE_ASSETS 指向准备好的数据目录，执行 pytest tests/test_author_attention.py；设置 AUTHOR_BASELINE_DEVICE=cuda 可在目标 GPU 上复核数值误差。

TSB-AD 的多变量移植将八层二值注意力偏置参数命名为 binary_attention_bias.embedding.weight，而公开的多变量权重将其命名为 binary_attention_bias.emd.weight。准备 TSB 原版对照时只对这八个键建立一一对应关系，在本地保存派生 state_dict，再逐张量使用 torch.equal 核对与原件完全一致。TSB 的严格加载继续校验完整的键与形状。目录中的 tsb-multi-strict.json 及输出登记保留原件指纹、派生件指纹和八个映射。

~~~bash
python -m pip install einops jaxtyping
python -m shifted_attention.evaluation prepare --data benchmarks --assets author
python -m shifted_attention.evaluation timercd --predictor author --data benchmarks --output results/original-paper
python -m shifted_attention.evaluation tsb --predictor author --data benchmarks --output results/original-tsb
python -m shifted_attention.evaluation rca --predictor severity --data benchmarks --ledger benchmarks/rca/msds.json --output results/severity-msds
python -m shifted_attention.evaluation rca --predictor severity --data benchmarks --ledger benchmarks/rca/chamber.json --output results/severity-chamber
python -m shifted_attention.evaluation tsb --predictor severity --data benchmarks --output results/severity-tsb
~~~

TimeRCD 的 16 来源基线调用作者公开的 TimeRCDDetector，默认测试后缀、5000 行窗口、batch 64。官方 TSB-AD 面板调用固定提交的 run_Time_RCD dispatcher，默认完整记录、15000 行窗口、batch 64。两者使用作者原有的整段归一化、非重叠窗、补齐和双类 softmax。作者权重固定到 Hugging Face revision 0880420070a524efe14d7e163b34dcca90d2be1b，按输入宽度选 U/M，完整 LFS SHA-256 见 assets.py 和每系列登记。分数进入与本模型相同的文件清单、指标引擎、逐系列分母和汇总。需要比较上下文时可显式传 --width，独立保存结果。[作者模型](https://github.com/thu-sail-lab/Time-RCD/tree/372bb980426b2f67007311c6f3165ab789c79bef) · [固定权重](https://huggingface.co/thu-sail-lab/Time-RCD/tree/0880420070a524efe14d7e163b34dcca90d2be1b) · [TSB dispatcher](https://github.com/TheDatumOrg/TSB-AD/blob/303dffbfc11612995b17b06633f8c9d0ea0e38e5/TSB_AD/model_wrapper.py)

统计严重度基线共享同样的通道归一化、事件区间和缺测规则。取每个传感器的最大绝对标准化偏离 z，限制在 30，映射到 z/(1+z) 供排序与有界指标调用。逐点检测取通道最大值，并平均覆盖窗的有效分数。该值表示标准化偏离程度，候选外结果只报其分布。若要做报警后定位，使用此基线在独立源校准集上确定的阈值。原版 TimeRCD 发布逐点异常分数，其公共接口没有直接根因排名。

AERCA 作者仓库固定版本的 39 项 Git 树中没有公开模型权重，GitHub Releases 也没有发行附件。其 SWaT 和 MSDS 对比涉及目标系统正常数据训练；文中 SWaT 为 51 变量序列、MSDS 为十指标派生标签。本文严格测量事件和零样本模型的训练访问、候选、时间边界均须独立登记。现有入口可评 SWaT 的完整直接测量目标及 MSDS 的派生代理，两者分别列出。复现 AERCA 原文数值还需可对齐的 SWaT 业务记录、训练访问和原论文的异常窗口定义。[作者仓库](https://github.com/hanxiao0607/AERCA/tree/c3dd49d88568d77c3e1ecb9a40472cd77a79b930)

| 一手来源 | 原论文使用的关联测试与指标 | 本仓库采用的口径 |
|---|---|---|
| Han 等，AERCA，ICLR 2025，§4 与附录 A.2.3 | 4 个合成系统及 SWaT、MSDS。每序列直接作用变量作集合，表 3 报 AC@1/3/5/10 及 Avg@10。附录另在变量×时刻候选上定义 AC*。 | 对明确真值的事件报告 AC@k 和 Avg@k。MSDS 十指标标签单独标为代理。当前根因头输出通道级排名，因此跨时间位置 AC* 不进入该头的对比。 [全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/6fde96479648d71e4fd9724374bf76eb-Paper-Conference.pdf) · [AERCA 标注与预处理](https://github.com/hanxiao0607/AERCA/tree/c3dd49d88568d77c3e1ecb9a40472cd77a79b930/datasets) |
| Tuli 等，TranAD，PVLDB 2022，表 4 | 单独在 SMD、MSDS 比较诊断。作者逐异常时刻计算 HR@100%/150% 和 NDCG@100%/150%，深度为四舍六入五成双的 round(p×阳性数)。 | 另报带 event- 前缀的事件级 HR/NDCG，采用同类截断深度。与逐时刻原始表格的分母和访问窗口不同。 [全文](https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf) · [原指标代码](https://github.com/imperial-qore/TranAD/blob/main/src/diagnosis.py) |
| Ikram 等，RCD，NeurIPS 2022，§5–6 | 结构方程合成数据及 Sock-shop CPU/内存注入，top-k recall 统计真实注入目标是否出现在前 k 个候选。 | 单根因 Hit@1、MRR 与 Recall@k 可解释为直接定位排名；多根因另报标准集合 Recall@k，分母为全部真实根因个数。 [全文](https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf) |
| Gamella 等，Nature Machine Intelligence 2025 | Causal Chambers 作者给出可控干预、测量和物理因果变量定义，有可核验的设定参数。 | 光隧道的测量装置归属是本文代理面板，风洞作根因在测量面板外的检查。 [论文](https://www.nature.com/articles/s42256-024-00964-x) · [光隧道发布说明](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_interventions_standard_v1) |

对每事件先用 512/128 窗、末尾补窗推理，按合格通道取有效窗口的最大根因概率。缺证据候选得零并保留。相同分数以 SHA256(release|run|channel_id) 的字典序破同分，AP 对相同分数整体积分。主表为 Hit@1、MRR、Recall@1/3 和 AP。AC@k 的分母是 min(k, |G|)，集合 Recall@k 的分母是 |G|。多根因按已知根因个数分层，退化、未知、候选外事件记录分母。表中代理真值按其独立单位报告。给定事件定位与先检测再定位分别输出，后者最大连续阈值区间与真值配对，IoU 至少 0.1，先最大化匹配数，再最大化总 IoU。漏检定位计零。记录 precision/recall、每千行和可用时钟的每小时误报数。阈值由训练脚本用源域校准数据确定。

**原始数据可用性。** prepare 默认取得 MSDS v1.1 的 concurrent 压缩包以及 AERCA 十列 labels.csv，复现 AERCA 指定的五主机顺序及其按索引标签删除原始行的具体行为，得到 325400 个合并时间戳。丢弃前 10% 后对半切分，测试区有 146430 行测量。标签来自 TranAD，以其原始 preprocess.py 对测量取 [::5]、对标签取 [::1]，恰好对齐 29286 个有序测试时间戳。标签文件与 AERCA 发布的文件共享同一个 Git blob be5b52a6d4c4f169267ff30b3bb83f8b50719cc9。AERCA 加载器再次对标签稀疏抽样会丢失对应关系，本面板按标签发布方的原始采样约定取值。[原始预处理](https://github.com/imperial-qore/TranAD/blob/7ffb98d0c18189cc3d9ab732b4cb0278200a0af0/preprocess.py)聚合连续阳性区间，使用段内各列 OR 作为指标标注代理。源域训练模型始终冻结。另外从 Rally 实际执行记录和独立服务日志构造直接服务面板，参见下文“MSDS 原生服务注入”。两个面板独立计分。[MSDS v1.1](https://zenodo.org/records/3549604)

Causal Chambers 的两个发行物用发布方 MD5 核验。光隧道保留九项实测量与 uniform_reference，针对 angle_1、angle_2、current 各两个实验，使用完整的 512 行非重叠窗；不足一整窗的实验保留一窗。先聚合每实验内的窗，再使六实验等权。风洞仅用十一项实测量，不向模型提供执行器命令或 intervention 列。另报最大传感器概率超过 0.5 的频率。[风洞发布说明](https://github.com/juangamella/causal-chamber/tree/main/datasets/wt_intake_impulse_v1)

复核 Causal Chambers 原始物理面板时，光隧道正常参考有 10000 行；六个干预文件各 1000 行，对应三个电压参考参数 v_c、v_angle_1、v_angle_2。参考值均为 5；各文件仅其对应参考参数移至约 2.56 或 1.10。目标字段是仪表的参考电压配置，测得的 current、angle_1、angle_2 用作仪表归属标签。预注册每次实验使用一段完整 512 行窗，尾部 488 行独立留存于原始输入。五个风洞实验中施加脉冲的是进风机负载命令 load_in，发布变量字典将 intervention 定义为指令后首次测量标志。11 个实测量中没有该命令，因此输出候选外断言统计。[光隧道变量说明](https://github.com/juangamella/causal-chamber/blob/c2b307036446b8d3ca173e9f41ff465b7111819a/datasets/lt_interventions_standard_v1/variables.csv) · [风洞协议](https://github.com/juangamella/causal-chamber/tree/c2b307036446b8d3ca173e9f41ff465b7111819a/datasets/wt_intake_impulse_v1)

## 导入 SWaT、WADI 与 MSDS 原生事件

### MSDS 原生服务注入

`prepare --assets msds` 生成十指标代理和直接服务两个面板。`prepare --assets msds-services` 单独生成后者。构建器验证 Zenodo concurrent 压缩包的发布 MD5，输出可核验的 31 项执行记录、29 项聚合事件、时钟锚点和输入来源摘要到 benchmarks/msds-direct-services；通用 industrial importer 输出 benchmarks/rca/msds-direct-service-logs.json 与相应 NPZ。此数据来自同一次并发实验，按一个独立运行登记。[原始数据](https://zenodo.org/records/3549604) · [作者数据说明](https://github.com/snedelkoski/multi-source-observability-dataset) · [作者论文](https://jorge-cardoso.github.io/rd/Papers/CP-2020-093-Multi-source_Distributed_System_Data.pdf)

**直接目标。** 三份实际 Rally 报告的 hooks/results 给出每次成功执行的 started_at、finished_at、返回码和容器回显。按容器归属得到 Nova 7 次、Glance 12 次、Neutron 12 次。取实际执行区间的时间并集，保留完全落在区间内的一秒计数箱。两对 Nova/Neutron 操作在时间上重叠，聚合后有 27 个单服务和 2 个双服务事件。事件边界表示直接干预动作的执行时段，恢复期症状可延续至时段之后。31 条实际回显均缺少被操作的物理主机 ID，真实目标单位在本面板固定为 OpenStack 服务。十指标面板另行保留每主机 CPU/内存标注。

**独立观测。** 只读取独立聚合日志的 @timestamp 与 Logger 两列。按发布记录中的全部八个具名 OpenStack 服务，将每秒日志条数作为输入。169230 行中 168 行缺少合格 Logger。输入为 21384×8，4079 秒内八项计数全零；零表示日志集中该秒没有收到对应服务消息。计数程序在读取任何干预记录之前执行。独立审计扫描了原日志文本中与实验驱动回显有关的签名，命中数为 0。报告字段、日志文本、告警等级、主机名和请求 ID 均不进入模型。该面板的观测是从实际服务日志派生的计数，属于直接服务定位的外部评测。

**时钟。** 三份报告保存 Unix epoch 执行时刻，全部 169230 行日志的 @timestamp 带 +01:00。按各自原始字段转换至 UTC，得到 2019-11-25 14:00:17 至 19:56:40 的日志计数轴，覆盖全部干预。通过 Rally 迭代的 trace ID 连到同一资源、方法、主机的独立 API 完成日志，12 个跨时间抽取的锚点的最大绝对差为 0.003141 秒。发行包提示文件的手写日期顺序自相矛盾，项目说明中的 CEST 文字与 CSV 的显式偏移亦有出入，重建采用可交叉验证的记录级时钟。

SWaT 选择 2015 年 12 月 Normal_v1、Attack_v0、同版 List_of_attacks_Final。WADI 选择 WADI.A2 的 WADI_14days_new.csv、WADI_attackdataLABLE.csv 以及同版直接注入目标说明。向 iTrust 获取授权数据后，冻结 spec JSON 和 events JSON。标注不全时保留原文引用，以 unknown 或 outside 标记。SWaT/WADI 的列选择需显式区分测量、执行器、目标外变量。为严格传感器面板只输入实测变量，确保每一名 eligible 目标均被包含。按原始事件表核对直接目标；同单位重叠事件若可穷尽归因先审定目标并集再登记为一个事件，其余重叠时间事件标注 unknown。程序验证列和时间边界、文件哈希、来源证据，合并重复时标的连续测量取中位数并保存源行到分组行映射。无法确认重复行是否同采集周期的记录应单独登记。WADI 标签列需显式映射 -1 至攻击、1 至正常，导入时输出点标注与事件表分歧行数。[SWaT 发行信息](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/) · [WADI 发行信息](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/)

调用 industrial --spec spec.json --data benchmarks。spec 的 features 是「候选 ID → 源数据精确列名」字典；若做原生实体定位，candidate_ids 列实体，entity_members 显式将每实体映射到 features 的键。reference 和 recording 支持用户的 pandas read_csv/read_excel 参数，如 skiprows/header。没有可靠正常参考时省略 reference。recording.time_columns、time_format、timezone 以相同发行版为准。日期时间保留原文，事件边界由同版来源核验后填写。下列为字段模板，尖括号均须替换。

~~~json
{"dataset":"SWaT","release":"<release ID>","id":"<recording>","group":"<independent run>",
 "truth_unit":"direct-measured-sensor","source_url":"<release documentation>",
 "features":{"<id>":"<source column>"},"candidate_ids":["<id>"],
 "recording":{"path":"<attack.xlsx>","read_options":{"header":1},"time_columns":["<timestamp>"],
              "time_format":"<strptime format>","timezone":"Asia/Singapore",
              "label_column":"<attack flag>","label_mapping":{"Normal":0,"Attack":1}},
 "reference":{"path":"<normal.xlsx>","read_options":{"header":1}},
 "events_json":"events.json"}
~~~

~~~json
[{"id":"<source event ID>","start_time":"<ISO 8601>","end_time":"<ISO 8601>",
  "end_inclusive":true,"status":"eligible","targets":["<id>"],
  "evidence_ref":"<release report, page/row and direct attacked tag>"}]
~~~

导入结果写入 benchmarks/rca/<recording>.json。进入直接传感器排名表的事件须有穷尽映射的非空目标集，且至少有一个合格负例。各登记表可独立运行。TSB-AD 的 SWaT 文件提供逐点异常标签，直接传感器评估使用同版事件目标表。
