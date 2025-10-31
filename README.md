#  Language Model Distillation

<img src="https://arxiv.org/html/2402.13116v3/x2.png" width="600">

## Overview
Model Distillation transfers the “knowledge” of a large **foundation model** (teacher) into a smaller **student model**, so the student can mimic high-level behavior with a fraction of the cost, memory, and energy.

In this project, a **405B-parameter LLaMA 3.1** model acts as the teacher to annotate sentiment data, and a **125 M-parameter RoBERTa-base** model learns from that labeled data — achieving comparable performance at **0.03 % of the size**.

> “This process is akin to transferring the knowledge of a highly skilled teacher to a student…”  
> — *A Survey on Knowledge Distillation of Large Language Models (ArXiv 2402.13116)*

---

## 🎯 Objective
Fine-tune a lightweight language model on synthetic annotations produced by a large teacher LLM, demonstrating how domain-specific tasks can be distilled efficiently.

---
![](https://github.com/prakhar105/llama-to-roberta-distillation/blob/main/img.png)
##  Architecture

```
┌──────────────────────────────┐
│        TEACHER MODEL         │
│  LLaMA 3.1 (405B params)     │
└──────────────┬───────────────┘
       │  (Generates labels)
       ▼
┌──────────────────────────────┐
│   Annotated Dataset (CSV)    │
│  tweet → sentiment (+ label) │
└──────────────┬───────────────┘
       │  (Used for distillation)
       ▼
┌──────────────────────────────┐
│        STUDENT MODEL         │
│   RoBERTa-base (125 M)       │
│  Fine-tuned → Sentiment LLM  │
└──────────────────────────────┘
```

---

##  Dataset

- **Source:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)  
- **Purpose:** Provide raw tweets for teacher LLM annotation.  
- **Output:** CSV files containing `id`, `text`, `label`, and `label_text`.

Example snippet:
```python
from datasets import load_dataset
ds = load_dataset("mteb/tweet_sentiment_extraction")
```

---

##  Workflow

1. **Teacher Annotation**
   - Prompts LLaMA 3.1 (405B) to label tweets as *positive*, *negative*, or *neutral*.
   - Saves annotations into structured CSV chunks.

2. **Student Training**
   - Loads RoBERTa-base from Hugging Face.
   - Fine-tunes on teacher-generated labels.
   - Evaluates accuracy on validation split.

3. **Comparison & Metrics**
   - Measures accuracy and F1-score of student vs teacher.
   - Demonstrates compression-efficiency trade-off.

---

##  Environment Setup

```bash
conda create -n distill python=3.10
conda activate distill

pip install torch transformers datasets tqdm accelerate
```

Optional for visualization:
```bash
pip install matplotlib seaborn
```

---

##  Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/llama-to-roberta-distillation.git
   cd llama-to-roberta-distillation
   ```

2. Open and execute:
   ```bash
   jupyter notebook model_distillation.ipynb
   ```

3. The notebook will:
   - Load dataset  
   - Generate synthetic labels  
   - Train RoBERTa student  
   - Display training logs and evaluation metrics  

---

##  Results

| Model          | Parameters | Accuracy | Relative Size |
|----------------|-------------|-----------|----------------|
| LLaMA 3.1 405B | 405 B       | ~ 65.49 %    | 100 %          |
| RoBERTa-base (distilled) | 125 M | ~ 63.38 % | **0.03 %**     |

> The distilled RoBERTa achieves near-teacher performance while being dramatically smaller and cheaper to deploy.

---

##  Extensions
- Multi-task distillation (classification + generation)
- Feature-map / attention-pattern alignment
- Distillation to quantized / LoRA-adapted models
- Integration with Hugging Face PEFT for efficiency

---

##  References
- [A Survey on Knowledge Distillation of Large Language Models (ArXiv 2402.13116)](https://arxiv.org/pdf/2402.13116)
- [Meta-LLaMA 3.1 (405B)](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B)
- [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base)

---

##  License
MIT License — feel free to use, modify, and extend.

---

##  Author
**Prakhar Awasthi**  
AI Engineer · Researcher · ML Practitioner
