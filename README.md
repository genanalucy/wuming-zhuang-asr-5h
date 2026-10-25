# GXMZU Wuming Zhuang ASR 5h

武鸣壮语语音识别5小时训练子集及原实验复现材料。**署名：GXMZU；许可：CC BY-NC 4.0（署名、非商业使用）。**

This release provides the exact 5h training subset, Zhuang transcripts, training recipe, source snapshots, and verification tools used in the corresponding Whisper experiment.

## 下载 / Download

- **[下载完整音频与复现包 v1.0.0（965 MB）](https://github.com/genanalucy/wuming-zhuang-asr-5h/releases/download/v1.0.0/wuming-zhuang-asr-5h-repro-v1.0.0.tar.gz)**
- [Release 页面](https://github.com/genanalucy/wuming-zhuang-asr-5h/releases/tag/v1.0.0)
- [完整数据与复现说明](reproduction/README.md) · [训练参数](reproduction/config/training.json) · [许可](LICENSE) · [来源标识说明](SOURCE_GROUP_NOTES.md)

| 项目 | 内容 |
|---|---|
| 样本数 | 2,585条训练音频及壮文标签 |
| 时长 | 原清单4.997919960504h；公开WAV按帧数4.995889704861h |
| 音频格式 | 16kHz、单声道、IEEE FLOAT32 WAV |
| 波形身份 | 逐点等于原训练实际使用的16kHz缓存输入 |
| 来源标识 | 22个匿名来源标识，尚未确认等同于22名独立真人 |
| 标签与顺序 | 保留实际5h训练成员、原标签、原行顺序 |
| 压缩包大小 | 965,115,141 bytes（约965 MB / 920 MiB） |
| 不包含 | 原dev/test、其他训练档位、模型权重、姓名映射或原私有路径 |

完整音频放在Release附件中。Git仓库中的`reproduction/`提供可浏览的说明、参数、源码及清单副本，**不含音频**。请下载并解压完整Release包后运行核验和准备脚本。

```bash
curl -L --fail -o wuming-zhuang-asr-5h-repro-v1.0.0.tar.gz \
  https://github.com/genanalucy/wuming-zhuang-asr-5h/releases/download/v1.0.0/wuming-zhuang-asr-5h-repro-v1.0.0.tar.gz
shasum -a 256 wuming-zhuang-asr-5h-repro-v1.0.0.tar.gz
tar -xzf wuming-zhuang-asr-5h-repro-v1.0.0.tar.gz
cd wuming-zhuang-asr-5h-repro-v1.0.0
python scripts/verify_package.py
```

期望SHA256：

```text
9330cdda0a7a784ac4edb6766473008a7a3803b3c3ce9b5d3aae94a954347965
```

## 原训练设置与复现范围

Whisper-large-v3，固定公共revision `06f233fe06e710322aca913c1bc4249a0d71fce1`；fresh初始化；seed42；A0无增强；3rank；每rank batch2、梯度累积3、global batch18；AdamW LR1e-5；最大1,149步；原best576、实际早停1008。完整选模、早停、数值精度和采样规则见[原配方](reproduction/config/training.json)。

**本次公开范围是5h训练集。**原dev/test未随包发布，因此仅凭本包不能独立复核原评价分数或完整重现原dev早停/选模。相同seed也不保证跨GPU、kernel和软件环境逐bit相同。准备脚本默认只准备、不训练；执行训练需另行提供通过原固定身份验证的dev。

匿名化覆盖文件名、来源标识及录音元数据；声音本身保留。来源别名与合成相关命名线索见[SOURCE_GROUP_NOTES.md](SOURCE_GROUP_NOTES.md)，不将来源标识数直接当作已确认真人数。采集时间、具体地点、设备、人口资料与人工审核流程中尚无证据的字段未补造。

`reproduction/README.md`及压缩包内文档是打包时冻结的来源快照，其中“尚未上传”等描述指打包时状态；当前发布地址以本仓库和Release为准。冻结压缩包的字节与校验哈希保持不变。

## 许可与引用

©2026 **GXMZU**。数据、壮文标签及本项目随附复现材料按[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)提供：署名、注明修改、仅限非商业使用。第三方模型和依赖保留各自许可。本包不附带上游模型权重。

建议引用：**GXMZU. GXMZU Wuming Zhuang ASR 5h, v1.0.0.** 机器可读信息见[CITATION.cff](CITATION.cff)。

发布版本包含原成员/标签/顺序核对、全波形位身份、完整archive文件校验和CPU复现入口检查。构建核验记录位于`reproduction/BUILD_VALIDATION.json`与`reproduction/provenance/`。
