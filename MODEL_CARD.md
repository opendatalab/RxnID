---
license: cc-by-nc-4.0
language:
- en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- chemistry
- reaction-diagram-parsing
- vision-language-model
- rxnid
- idtvp
---

# RxnID

RxnID is the model checkpoint for **Molecular Identifier Visual Prompt and Verifiable Reinforcement Learning for Chemical Reaction Diagram Parsing**.

- Project page: https://chuangwang123.github.io/RxnID/
- Paper: https://arxiv.org/abs/2603.15011
- Code: https://github.com/opendatalab/RxnID
- License: CC BY-NC 4.0

## Usage

This checkpoint is intended to be used with the RxnID codebase:

```bash
git clone https://github.com/opendatalab/RxnID
cd RxnID
pip install -r requirements.txt

bash scripts/run_inference.sh \
    --image_dir /path/to/reaction_images \
    --idt_file /path/to/image_idts.json \
    --model songjhPKU/RxnID \
    --output_dir outputs/inference
```

## Citation

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

