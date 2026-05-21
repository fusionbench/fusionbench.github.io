# FusionBench: A comprehensive Benchmark for QA under Ambiguity and Heterogeneity

<div align="center">
<!-- <a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a> -->
<a href="https://fusionbench.github.io/" target="_blank"><img src="https://img.shields.io/badge/Leaderboard-blue"></a>
<a href="https://huggingface.co/datasets/cccwe/FusionBench/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/Amireon/FusionBench" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://opensource.org/license/mit"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
</div>

## 🧾 Overview

`FusionBench` is a dataset of ambiguous, heterogeneous QA examples with carefully annotated interpretations and evidence sets, enabling fine-grained evaluation beyond single-answer accuracy.

Here is the dataset statistics.

| Question Type | #Questions | #Answers | Text(%) | Table(%) | Triples(%) |
|---------------|-------------|-----------|----------|-----------|-------------|
| Multi-answer  | 3,473       | 12,521    | 75.96%   | 71.43%    | 76.51%      |
| Single-answer | 3,706       | 3,706     | 99.24%   | 60.06%    | 61.31%      |



## 📐 Metrics

Precision and recall are computed by considering both the correctness of the generated answer and its associated disambiguated entity.

- **Answer Precision / Recall (AP/AR):**  
  Precision and recall computed at the token level with respect to all gold-standard answers. This measures how accurately the model generates correct answer content.
- **Entity Precision / Recall (EP/ER):**  
  Precision and recall computed at the token level with respect to all gold-standard disambiguated entities. This evaluates the accuracy of entity grounding in the responses.
- **Entity-Answer Precision / Recall (EAP/EAR):**  
  Joint metrics that combine both answer and entity evaluation. EAP/EAR requires both the answer and its linked entity to be correct, providing a stricter measure of end-to-end performance.

## 🚀 How to use

## Download Dataset

You can download the dataset from [HuggingFace](https://huggingface.co/datasets/cccwe/FusionBench) ,  [Google Drive](https://drive.google.com/drive/folders/1DWKhFhXdS42dO7mnE6XjpXYt_nSxI0ud?usp=drive_link) or [BNU Drive](https://pan.bnu.edu.cn/v/link/view/3fee1f1cf09f4c97a40e7edac81d3154).

Then put the original question-answer `JSON` files in directory `data/`.

### Environment Setup
1.Install SDK in your `python` environment by:

```bash
pip install -r requirements.txt
```

2.Configure LLMs provider in `env.yaml`. Here is a demo.

```yaml
llm_providers:
  openai:
    api_key: "OpenAI API Key"
    base_url: "https://api.openai.com/v1/"
    models:
      - "gpt-4o"
      - "gpt-4o-mini"

  openai2:
    api_key: "OpenAI API Key"
    base_url: "https://api.openai.com/v1/"
    models:
      - "gpt-4o"
      - "gpt-4o-mini"
```

You can add more LLMs providers in `env.yaml` following the demo.

### Generation
Then you could directly execute the command line by following instructions and the answers will be stored in a `jsonl` file:

```bash
python generate.py --task evidece --qtype multi --model gpt-4o
```

Here is the parameter table.

| Argument          | Type / Action        | Required | Default    | Choices / Description                                                                 |
|-------------------|----------------------|----------|------------|----------------------------------------------------------------------------------------|
| `--task`          | `str`                | Yes      | —          | `"evidence"` or `"rag"`: Specifies the task context.                                   |
| `--qtype`         | `str`                | Yes      | —          | `"multi"` or `"single"`: Indicates the question type.                                  |
| `--model`         | `str`                | Yes      | —          | Model name from `get_available_models()`: Specifies the model to use.                  |
| `--restore_path`  | `str`                | No       | `None`     | Path to a previous save directory for resuming progress.                               |
| `--use_align`     | `store_true` (flag)  | No       | `False`    | In OpenBook mode, whether to use grouped context. RAG mode only supports ungrouped context. |
| `--use_think`     | `store_true` (flag)  | No       | `False`    | Whether to enable the reasoning model's "thinking" mode.                                |
| `--use_pipeline`  | `store_true` (flag)  | No       | `False`    | Whether to enable the pipeline mode.                                                   |

> **Notes**:
> - Arguments with `action="store_true"` are boolean flags: present → `True`, absent → `False`.

### Evaluation

When all answers have been successfully generated, you can compute the evaluation metrics using the following command:  

```bash
python evaluate.py --data_path /path/to/predictions.jsonl
```

Here is the parameter table for the evaluation script:

| Argument        | Type / Action        | Required | Default    | Description                                                                 |
|-----------------|----------------------|----------|------------|-----------------------------------------------------------------------------|
| `--data_path`   | `str`                | No       | `None`     | Path to a single result file for which metrics will be computed.            |
| `--data_dir`    | `str`                | No       | `None`     | Root directory containing result files (non-recursive; subdirectories are ignored). If both `--data_path` and `--data_dir` are provided, only `--data_path` is valid and used. |
| `--verbose`     | `store_true` (flag)  | No       | `False`    | Enable verbose output (e.g., print example predictions for debugging).      |

> **Note**:  
> - Only one of `--data_path` or `--data_dir` should be used at a time. If both are specified, the script prioritizes `--data_path`.  
> - The evaluation assumes that the result files contain model-generated answers aligned with ground-truth labels in a supported format.
