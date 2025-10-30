# EverMemModel


[![arXiv](https://img.shields.io/badge/arXiv-2510.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2510.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **EverMemModel: End-to-End Memory Model for Ultra-Long Contexts Question Answering with Dual-Stream Sparse Attention**.

## 📝 Abstract

Large Language Models (LLMs) struggle in knowledge-intensive domains that require deep, specialized knowledge. While Retrieval-Augmented Generation (RAG) is a common solution, its decoupled retrieve-then-read pipeline suffers from misaligned objectives and is prone to performance degradation from distractor documents. We propose **EverMemModel**, a unified, end-to-end trainable architecture that treats a massive document corpus as an extended context. The core of our model is the novel **Dual-Stream Sparse Attention (DSA)** mechanism, which efficiently processes the entire document set, identifies relevant information, and generates the final answer in a single, jointly-optimized process. Through a specialized three-stage training and inference procedure, EverMemModel can handle memory contexts on the scale of **100 million tokens**. Our model achieves state-of-the-art results on both retrieval (NQ320k) and question-answering (MS MARCO) benchmarks, significantly outperforming traditional RAG pipelines and powerful long-context models.

## ✨ Key Contributions

-   **End-to-End Memory Model**: We propose **EverMemModel**, a unified architecture that seamlessly integrates retrieval and generation, moving beyond the limitations of decoupled RAG systems.
-   **Dual-Stream Sparse Attention (DSA)**: We design a novel and efficient attention mechanism that enables scalable processing of massive document corpora by separating intra-document processing (Memory Stream) from query-aware retrieval and generation (Generation Stream).
-   **State-of-the-Art Performance**: EverMemModel achieves SOTA performance on both the NQ320k retrieval benchmark and the MS MARCO question-answering task.
-   **Massive-Scale Context**: Thanks to its efficient architecture and a three-stage inference process, EverMemModel is one of the first models capable of handling contexts up to **100M tokens**.

## 🏗️ Architecture: Dual-Stream Sparse Attention (DSA)

The core of EverMemModel is the DSA mechanism, which replaces standard self-attention with two specialized, parameter-sharing streams.
[dsa3.drawio.pdf](https://github.com/user-attachments/files/23229685/dsa3.drawio.pdf)

<p align="center">
  <img src="https://i.imgur.com/w2t6X6S.png" width="800">
</p>
<p align="center">
  <em><b>Figure 1</b>: The Dual-Stream Sparse Attention (DSA) mechanism. The Memory Stream processes all documents in parallel. The Generation Stream's router selects relevant documents (e.g., Doc 2 & 4), and their compressed representations are concatenated with the question to generate the final answer.</em>
</p>

1.  **Memory Stream**: Processes each document independently using standard intra-document self-attention. This preserves the internal semantic structure of each document.
2.  **Generation Stream**: Performs dynamic retrieval and answer generation. Its router component is key to its efficiency.

[DSA_router_2.drawio.pdf](https://github.com/user-attachments/files/23229609/DSA_router_2.drawio.pdf)

<p align="center">
  <img src="https://github.com/user-attachments/files/23229609/DSA_router_2.drawio.pdf" width="700">
</p>
<p align="center">
  <em><b>Figure 2</b>: The architecture of the router. It chunks and pools document representations, calculates relevance scores against the question, selects the top-k documents, and concatenates their compressed representations to form the final context.</em>
</p>

## 🚀 Results

### Retrieval Performance (NQ320k)

EverMemModel sets a new state of the art on generative retrieval. The best result is in **bold**, second-best is <u>underlined</u>, and third-best is in *italics*.

| Method | NQ320K (Full text) | | NQ320K (Unseen) | |
| :--- | :---: | :---: | :---: | :---: |
| | **R@1** | **R@10** | **R@1** | **R@10** |
| **_Sparse retrieval_** | | | | |
| BM25 (Robertson & Zaragoza, 2009b) | 29.7 | 60.3 | 32.3 | 61.9 |
| DocT5Query (Cheriton, 2019) | 38.0 | 69.3 | 48.5 | 72.9 |
| **_Dense retrieval_** | | | | |
| DPR (Karpukhin et al., 2020b) | 50.2 | 77.7 | 50.0 | 74.2 |
| ANCE (Xiong et al., 2021) | 50.2 | 78.5 | 52.0 | 75.9 |
| GTR-Base (Ni et al., 2021) | 56.0 | 84.4 | 61.9 | 83.2 |
| Sentence-T5 (Ni et al., 2022) | 53.6 | 83.0 | 56.5 | 79.5 |
| HCE-J (Chen et al., 2025) | *71.2* | **93.9** | - | - |
| Qwen3-Embedding-0.6B (Zhang et al., 2025) | 54.0 | 82.6 | 54.8 | 80.8 |
| Qwen3-Embedding-4B (Zhang et al., 2025) | 62.6 | 89.2 | <u>62.6</u> | **86.7** |
| **_Generative retrieval_** | | | | |
| DSI-QG (Zhuang et al., 2022) | 63.1 | 80.7 | 45.9 | 65.8 |
| NCI (Wang et al., 2022) | 66.4 | 85.7 | 54.5 | 75.9 |
| GenRet (Sun et al., 2023) | 68.1 | 88.8 | *62.5* | <u>83.6</u> |
| Self Retrieval (Tang et al., 2024) | <u>73.3</u> | <u>92.6</u> | - | - |
| **Ours (EverMemModel)** | **75.5** | *90.6* | **66.5** | *83.5* |

### Question Answering Performance (MS MARCO)

EverMemModel significantly outperforms both strong RAG baselines and large-context models.

| Dataset | Docs | Qwen3RAG-QA | | | Gemini-2.5-Flash | EverMemModel (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| | | **R@1** | **R@5** | **R@10** | | |
| MS MARCO (0.8M Tokens) | 8,389 | 2.235 | 2.535 | 2.548 | 2.710 | **3.812** |
| MS MARCO (7.1M Tokens) | 75,574| 2.225 | 2.521 | 2.759 | N/A† | **2.774** |

† *Input exceeds the model's maximum context length.*

Notably, EverMemModel achieves this with an average adaptive recall of just **2.5 documents**, showcasing its superior efficiency and precision compared to fixed-size RAG retrieval.
<!--
## 🛠️ Setup & Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/EverMemModel.git](https://github.com/your-username/EverMemModel.git)
    cd EverMemModel
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Training & Inference

### Training

The model is trained using our three-stage curriculum (continuous pre-training, normal-sample finetuning, hard-sample finetuning). To launch training, run:

```bash
# Example command for Stage 2: Normal-Sample Finetuning
python train.py \
    --model_name_or_path qwen/Qwen3-4B-Instruct \
    --dataset_name ms_marco \
    --stage normal_sample_finetuning \
    --output_dir ./checkpoints/longmem_msmarco_stage2
-->
