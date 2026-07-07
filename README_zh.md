<p align="center">
  <img src="assets/banner.png" width="45%" alt="RxnID Banner" />
</p>

<h1 align="center"><img src="assets/logo.png" width="36" style="vertical-align: middle;" /> RxnID</h1>

**Molecular Identifier Visual Prompt and Verifiable Reinforcement Learning for Chemical Reaction Diagram Parsing**

[项目主页](https://chuangwang123.github.io/RxnID/) | [arXiv](https://arxiv.org/abs/2603.15011) | [模型](https://huggingface.co/songjhPKU/RxnID) | [Mid-Mapper](https://huggingface.co/songjhPKU/Mid-Mapper) | 数据集：待更新 | License: CC BY-NC 4.0

RxnID 面向化学反应图解析，将反应图片解析为结构化 reaction JSON。核心思路是 **Identifier as Visual Prompting (IdtVP)**：用分子 identifier（如 `1a`、`2b` 或虚拟 identifier）作为视觉锚点；同时用 **Re3-DAPO** 通过可验证的 reaction-level reward 做强化学习优化。

<p align="center">
  <img src="assets/fig3.png" width="100%" alt="RxnID Framework" />
</p>

## 最新动态

- [07/2026] 整理 ECCV 2026 开源仓库代码。
- [03/2026] 论文已发布到 arXiv: [2603.15011](https://arxiv.org/abs/2603.15011)。

## 亮点

- **IdtVP**：用分子 identifier 作为视觉提示，增强 VLM 对反应结构的理解。
- **Mid-Mapper**：从分子检测框中识别 identifier，并为缺失 identifier 的分子渲染虚拟 IDT。
- **Re3-DAPO**：使用 Soft/Hybrid reaction-level reward 的 RLVR 训练。
- **ScannedRxn**：面向历史扫描文献反应图的鲁棒性 benchmark。
- **完整脚本**：包含推理 JSONL 构造、swift 输出转换、IdtVP 评测、verl 训练脚本。

## 主要结果

下表为 ECCV 2026 论文中的 F1 结果。H-F1 表示 Hybrid Match F1，S-F1 表示 Soft Match F1。

| 方法 | 策略 | RxnScribe H/S | RxnCaption-15k H/S | ScannedRxn H/S |
|---|---:|---:|---:|---:|
| RxnID + RL | IdtVP | 75.0 / 85.9 | 64.4 / 74.5 | 56.3 / 76.2 |
| RxnID | IdtVP | 74.6 / 85.6 | 61.2 / 72.8 | 54.5 / 74.4 |
| RxnCaption + RL | BIVP | 75.9 / 86.9 | 64.1 / 73.6 | 45.1 / 59.2 |
| RxnCaption | BIVP | 72.2 / 86.2 | 59.8 / 70.4 | 51.0 / 69.5 |

## 仓库结构

```text
RxnID/
├── README.md / README_zh.md
├── LICENSE
├── requirements.txt
│
├── rxnid/
│   ├── build_inference_jsonl.py   # 构造 IdtVP 推理 JSONL
│   ├── prompt.py                  # IdtVP prompt 模板
│   ├── identifier.py              # Mid-Mapper 兼容入口
│   ├── evaluate_idtvp.py          # IdtVP JSON 的 Soft/Hybrid 评测
│   ├── annotate_bivp.py           # BIVP baseline 打框工具
│   ├── evaluate_bivp.py           # BIVP/RxnCaption bbox 格式评测
│   ├── mid_mapper/                # identifier 识别和 IDT 渲染流程
│   └── rl/reward.py               # verl 使用的 Re3-DAPO reward
│
├── tools/
├── scripts/
├── demo/
└── docs/
```

## 快速开始

```bash
git clone https://github.com/opendatalab/RxnID
cd RxnID
pip install -r requirements.txt
```

运行推理：

```bash
bash scripts/run_inference.sh \
    --image_dir /path/to/reaction_images \
    --idt_file /path/to/image_idts.json \
    --model songjhPKU/RxnID \
    --output_dir outputs/inference
```

`image_idts.json` 格式如下：

```json
{
  "example.png": ["1a", "2b", "3c"]
}
```

运行评测：

```bash
bash scripts/run_eval.sh \
    --gt_file data/val_gt.json \
    --pred_file outputs/inference/prediction.json \
    --output_dir outputs/eval
```

快速 demo：

```bash
MODEL=songjhPKU/RxnID bash demo/run_demo.sh
```

## Mid-Mapper Identifier 流程

运行 identifier 识别、缺失 IDT 分配/渲染，并生成 IdtVP SFT JSONL：

```bash
bash scripts/run_mid_mapper.sh \
    --image_dir /path/to/raw_images \
    --json_in /path/to/bivp_mapped.json \
    --model_path songjhPKU/Mid-Mapper \
    --num_splits 4 \
    --output_dir outputs/mid_mapper
```

Mid-Mapper 模型默认使用 `songjhPKU/Mid-Mapper`，也可以通过 `--model_path /local/path` 覆盖。可以用 `--dry_run` 做不加载模型的流程检查。完整说明见 [docs/MID_MAPPER.md](docs/MID_MAPPER.md)。

## 训练

转换 verl parquet：

```bash
bash scripts/prepare_data.sh \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir data/parquet
```

启动 Re3-DAPO：

```bash
bash scripts/run_rl_train.sh \
    --verl_root /path/to/verl \
    --train_file data/parquet/train.parquet \
    --val_file data/parquet/val.parquet \
    --model /path/to/sft-checkpoint \
    --output_dir outputs/rl
```

详见 [docs/TRAINING.md](docs/TRAINING.md)。

## 模型权重

RxnID 主模型见 [docs/MODEL.md](docs/MODEL.md)。Mid-Mapper identifier 识别模型是单独的模型 artifact，发布位置为 [songjhPKU/Mid-Mapper](https://huggingface.co/songjhPKU/Mid-Mapper)。

## MolYOLO

MolYOLO 用于 BIVP baseline 和 Mid-Mapper 流程中的分子检测。本仓库不内置 MolYOLO 代码或权重，相关部分直接指向 [RxnCaption](https://github.com/opendatalab/RxnCaption) / [MolYOLO](https://github.com/songjhPKU/MolYOLO) 的发布说明。

## 引用

```bibtex
@misc{song2026molecularidentifiervisualprompt,
      title={Molecular Identifier Visual Prompt and Verifiable Reinforcement Learning for Chemical Reaction Diagram Parsing},
      author={Jiahe Song and Chuang Wang and Yinfan Wang and Hao Zheng and Rui Nie and Bowen Jiang and Xingjian Wei and Junyuan Gao and Yubin Wang and Bin Wang and Lijun Wu and Jiang Wu and Qian Yu and Conghui He},
      year={2026},
      eprint={2603.15011},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.15011},
}
```

## License

本项目使用 CC BY-NC 4.0 license，见 [LICENSE](LICENSE)。
