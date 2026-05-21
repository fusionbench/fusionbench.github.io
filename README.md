# FusionBench


## 🧾 Overview

`FusionBench` is a dataset of ambiguous, heterogeneous QA examples with carefully annotated interpretations and evidence sets, enabling fine-grained evaluation beyond single-answer accuracy.

You can find data here: https://drive.google.com/drive/folders/1BLXqTXMReLpprp6hWpzebExc7mxhu8DC?usp=sharing

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
