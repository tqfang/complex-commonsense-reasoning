# Complex Reasoning over Logical Queries on Commonsense Knowledge Graphs

This is the official code and data repository for the ACL 2024 paper: [Complex Reasoning over Logical Queries on Commonsense Knowledge Graphs](https://arxiv.org/abs/2403.07398).

### 1. Dataset

The human-annotated evaluataion set and the distantly-supervised instruction tuning dataset can be found at [data](https://huggingface.co/datasets/tqfang229/COM2-commonsense).

### 2. Experiments

- OpenAI LLM prompting experiments

Under the `prompting` folder.

- Zero-shot QA Model

Under the `zero_shot_csqa` folder.

Processed evaluated Commonsense QA datasets can be found at [here](https://github.com/HKUST-KnowComp/CAR/blob/main/tasks/)

- Generative commonsense inference

Under the `comet` folder.


### 3. Citing this work

```
@article{fang2024complex,
  title={Complex Reasoning over Logical Queries on Commonsense Knowledge Graphs},
  author={Fang, Tianqing and Chen, Zeming and Song, Yangqiu and Bosselut, Antoine},
  journal={ACL},
  year={2024}
}
```