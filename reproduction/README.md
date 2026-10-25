---
pretty_name: GXMZU Wuming Zhuang ASR 5h
license: cc-by-nc-4.0
task_categories:
  - automatic-speech-recognition
size_categories:
  - 1K<n<10K
---

# GXMZU Wuming Zhuang ASR 5h — Reproduction Package v1.0.0

**署名 / Attribution：GXMZU**  
**数据许可 / Data license：CC BY-NC 4.0（署名、仅限非商业使用）**  
本文件夹是准备公开的复现包，尚未由打包程序上传到任何外部平台。

本包包含实际5h训练子集的2,585条音频与壮文标签、22个匿名来源说话人标识、原顺序、原训练参数、正式运行的源码快照、可迁移训练入口、环境记录和校验工具。数据是团队的武鸣壮语给定句子朗读材料。未对年龄、性别、语言背景、采集具体地点或人工标注审核流程作未经记录的推断。

## 波形与数据范围

- `train/audio/*.wav`：16kHz单声道IEEE FLOAT32 WAV。每一条均与原实验使用的16kHz缓存波形逐点相等；打包时另从原音频按同版ffmpeg规则重新解码核对。
- 原容器清单标称4.997919960504小时。公开音频实际时长由帧数计算，两者存在少量容器时长差异；manifest分别保留`source_duration_seconds`和`duration`，训练准备使用前者以保留原筛选/统计口径。
- 原始训练样本顺序不变。公开文件名、路径和说话人标识重新匿名化，WAV只含fmt/fact/data必要块，没有title/artist/comment等原始录音元数据。声音本身保留。
- 壮文`text`保持源标签；训练/计分时才执行原NFC与空白折叠。没有根据模型输出修改标签。
- 这是**训练子集**。本包不包含原4,120条dev、1,980条test、其他训练档位、训练好的模型或优化器状态。
- 22是来源标识数，尚未据人口资料逐个确认独立真人数。匿名来源别名/原TTS命名线索见`SOURCE_GROUP_NOTES.md`；不按命名擅自合并或删除，保持论文5h成员。

`train/metadata.csv`可用于AudioFolder类数据加载：`file_name,text,speaker_id,duration,id`。CSV内路径相对`train/`；`train/manifest.jsonl`另含帧数、文件SHA、PCM SHA与原统计时长。

## 立即进行离线核验

```bash
python scripts/verify_package.py
# 安装numpy和soundfile后，可再核实际解码PCM
python scripts/verify_package.py --full-audio
```

第一次命令核全部文件SHA、2,585条覆盖、元数据和顺序；第二次再逐条读FLOAT32波形核PCM身份。验证不使用GPU、不访问外网、不训练模型。

## 原训练配方

| 设置 | 固定值 |
|---|---|
| 初始化 | openai/whisper-large-v3，revision `06f233fe06e710322aca913c1bc4249a0d71fce1`，fresh |
| seed / 增强 | 42 / A0无增强 |
| GPU / 数值精度 | 原3×NVIDIA L20；FP32参数、BF16 autocast |
| 每rank batch / 梯度累积 / global batch | 2 / 3 / 18 |
| 最大更新 / warmup | 1,149 / 57 |
| 优化器 | AdamW；LR1e-5、weight_decay0.01、默认betas(0.9,0.999)、eps1e-8 |
| LR调度 | 原LambdaLR：warmup后线性下降，完整定义保留源码 |
| gradient clipping / checkpointing | max_norm1.0 / enabled、use_reentrant=False |
| worker / eval batch | 每rank1 / 1 |
| monitor / full-dev间隔 | 46 / 144步 |
| monitor选取 | 原dev顺序均匀linspace512条 |
| 选模 / 早停 | 完整dev CER严格改善；至少431步、patience3 |
| 采样 | 每epoch打乱，DistributedSampler seed42，drop_last，不补齐尾部 |
| 生成 | greedy，无语言ID的[50258,50360,50364]前缀，无采样、无时间戳 |
| 原观测best / terminal | 576 / 1008步；1008为实际早停，不将预算改为1008或训练固定到576 |

详细设置见`config/training.json`。源码控制固定3rank/global18；改变GPU拓扑或batch的方案属于另一设置，不承诺得到同一训练轨迹。

## 环境与公共模型

原run明确记录PyTorch2.6.0+cu124、NVIDIA L20；生成和源码依赖的Transformers4.46.3也被现有运行环境和审计记录支持。`environment/runtime_observed.json`记录打包时实际环境，未在历史run单独落盘的Python/NumPy等版本明确标为“当前观察”，不冒充原训练时完整锁文件。

建议Python3.12及CUDA12.4兼容环境：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-observed.txt
# 精确revision与每个公共初始化文件SHA单独验证；需下载约6GB上游模型
python scripts/fetch_model.py
```

上游Whisper模型不在包中，其条款按上游模型仓库处理。数据的CC BY-NC许可不重新许可第三方模型或第三方库。

## 准备训练与执行

先在**包目录之外**创建新的工作目录，不破坏不可变的公开数据：

```bash
# 无dev也能检查本包，并生成原顺序的训练清单/精确FLOAT32缓存链接
python scripts/prepare_training.py --workdir ../wz5h_prepared
```

这个命令打印原配方的命令和环境变量，但不会训练。它输出的`prepared.json`在没有dev时明确标为未具备原dev评价；不会以训练数据伪造验证集。

要复现原早停/选模与报告分数，必须另行取得原dev。其JSONL需要`audio,sha256,speaker_candidate,duration,text`，音频文件可用，SHA和参考/顺序符合`config/evaluation_identity.json`的完整签名。当前5h公开范围不自动授权发布dev/test。

```bash
# 使用另一个全新workdir。此命令验证dev、公共模型与本包后才启动3rank训练
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/prepare_training.py \
  --workdir ../wz5h_reproduce_seed42 \
  --dev-manifest /path/to/original_dev.jsonl \
  --execute
```

`code/`是原正式训练源码的最小可迁移版本：只把5h已私有清单哈希门替换为“本包先完整校验后生成的清单SHA”。模型初始化、增强A0、采样、优化、恢复、生成和计分数值函数未变，差异见`provenance/portable_trainer.diff`及`source_migration.json`。`source/`保留原源码。原始5h训练/波形身份为审计来源，匿名后的JSONL文件字节因路径/ID变化不同；不伪造其SHA与旧私有清单相同。

## 指标与能够复现的范围

参考和预测NFC＋空白折叠，保留大小写、标点、数字；CER去空白；token-WER使用正字法空白token。总编辑数除以总参考单位数，不平均逐句比率。`scripts/score_predictions.py`提供原评分入口，输入需含reference/prediction的JSONL。

| 原5h模型 | 评价集 | CER | token-WER |
|---|---|---:|---:|
| best576 | dev4120 | 12.270988% | 54.574076% |
| final1008 | dev4120 | 12.611491% | 53.671142% |
| best576 | test1980 | 17.052176% | 59.587408% |

本包可核验训练样本、精确波形、原顺序及配方。**仅有这5h，无法独立复核上述未公开dev/test分数，也不能复现原完整早停选择。** 即使另行取得相同评价材料，GPU/kernel与运行环境仍可能造成非逐bit的训练差异。不会声称只需seed42即可保证相同分数。

## 许可、引用与来源

©2026 GXMZU。数据/壮文标签按CC BY-NC4.0提供：需要署名、注明修改，仅限非商业使用；完整许可在`LICENSE`。署名建议“GXMZU, Wuming Zhuang ASR 5h, v1.0.0”，引用见`CITATION.cff`。公开仓库URL、DOI与具体文章书目信息尚未产生，不虚构。

采集时间/地点、物理录音设备、标签原始形成和人工听音审核流程仍有待团队完善；本包提供有依据的技术处理记录，不能当作所有标签已由语言学专家逐条认证。

打包过程逐条核对原训练清单、原音频SHA、实际解码/缓存波形、元数据与公开波形；原姓名映射和源路径保留私有审计目录。`BUILD_VALIDATION.json`、`provenance/`、`CHECKSUMS.sha256`记录实际覆盖、源码身份和校验结果。发布包没有真实来源文件名、服务器路径、原说话人姓名映射、dev/test或训练权重。

本包随附的本项目数据、标签、复现脚本和说明按署名中的许可声明提供；第三方模型和依赖库保留各自原有许可证，不被本包的数据许可覆盖。
