# FusionBench

<p align="center">
    <a href="https://arxiv.org/abs/2408.09174">📚 Paper</a>
    &nbsp;&nbsp;
    <a href="https://fusionbench.github.io/">🏆  Leaderboard</a> 
    &nbsp;&nbsp;
    <a href="https://huggingface.co/datasets/cccwe/FusionBench">🤗 FusionBench</a> 
</p>

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

### Environment Setup
1.Install SDK in your `python` environment by:

```bash
pip install -r requirements.txt
```

2.Configure LLMs service in `llm_service.py`. Here is a demo.

```json
MODEL_PROVIDER_CONFIG = {
    "openai": {
        "api_key": "XXXXXXXXXXX",
        "base_url": "https://api.openai.com/v1/",
        "models": ["gpt-4o", "gpt-4o-mini"],
    },
    ...
}
```

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

Evaluation results will be saved in a CSV file. Then you can just submit the CSV file to our communication email: `zcwang@bnu.edu.cn`.

Here is the parameter table for the evaluation script:

| Argument        | Type / Action        | Required | Default    | Description                                                                 |
|-----------------|----------------------|----------|------------|-----------------------------------------------------------------------------|
| `--data_path`   | `str`                | No       | `None`     | Path to a single result file for which metrics will be computed.            |
| `--data_dir`    | `str`                | No       | `None`     | Root directory containing result files (non-recursive; subdirectories are ignored). If both `--data_path` and `--data_dir` are provided, only `--data_path` is valid and used. |
| `--verbose`     | `store_true` (flag)  | No       | `False`    | Enable verbose output (e.g., print example predictions for debugging).      |

> **Note**:  
>
> - Only one of `--data_path` or `--data_dir` should be used at a time. If both are specified, the script prioritizes `--data_path`.  
> - The evaluation assumes that the result files contain model-generated answers aligned with ground-truth labels in a supported format.



## 📄 Citation

If you find our work helpful, please use the following citations.

```
```
