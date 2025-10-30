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


<p align="center">
  <img src="https://github.com/user-attachments/files/23229685/dsa3.drawio.pdf" width="800">
</p>
<p align="center">
  <b>Figure 1</b>: The Dual-Stream Sparse Attention (DSA) mechanism. The Memory Stream processes all documents in parallel. The Generation Stream's router selects relevant documents (e.g., Doc 2 & 4), and their compressed representations are concatenated with the question to generate the final answer.
</p>

1.  **Memory Stream**: Processes each document independently using standard intra-document self-attention. This preserves the internal semantic structure of each document.
2.  **Generation Stream**: Performs dynamic retrieval and answer generation. Its router component is key to its efficiency.


<p align="center">
  <img src="https://github.com/user-attachments/files/23229609/DSA_router_2.drawio.pdf" width="700">
</p>
<p align="center">
  <b>Figure 2</b>: The architecture of the router. It chunks and pools document representations, calculates relevance scores against the question, selects the top-k documents, and concatenates their compressed representations to form the final context.
</p>

## 🚀 Results

### Retrieval Performance (NQ320k)

EverMemModel sets a new state of the art on retrieval task. The best result is in **bold**.

| Method | NQ320K (Full text) | NQ320K (Unseen) |
| :--- | :---: | :---: |
| | **R@1** | **R@1** |
| **_Sparse retrieval_** | | |
| BM25 (Robertson & Zaragoza, 2009b) | 29.7 | 32.3 |
| DocT5Query (Nogueira et al., 2019) | 38.0 | 48.5 |
| **_Dense retrieval_** | | |
| DPR (Karpukhin et al., 2020b) | 50.2 | 50.0 |
| ANCE (Xiong et al., 2021) | 50.2 | 52.0 |
| GTR-Base (Ni et al., 2021) | 56.0 | 61.9 |
| Sentence-T5 (Ni et al., 2022) | 53.6 | 56.5 |
| HCE-J (Chen et al., 2025) | 71.2 | - |
| Qwen3-Embedding-0.6B (Zhang et al., 2025) | 54.0 | 54.8 |
| Qwen3-Embedding-4B (Zhang et al., 2025) | 62.6 | 62.6 |
| **_Generative retrieval_** | | |
| DSI-QG (Zhuang et al., 2022) | 63.1 | 45.9 |
| NCI (Wang et al., 2022) | 66.4 | 54.5 |
| GenRet (Sun et al., 2023) | 68.1 | 62.5 |
| Self Retrieval (Tang et al., 2024) | 73.3 | - |
| **Ours (EverMemModel)** | **75.5** | **66.5** |

<!--
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
| Qwen3-Embedding-4B (Zhang et al., 2025) | 62.6 | 89.2 | 62.6 | **86.7** |
| **_Generative retrieval_** | | | | |
| DSI-QG (Zhuang et al., 2022) | 63.1 | 80.7 | 45.9 | 65.8 |
| NCI (Wang et al., 2022) | 66.4 | 85.7 | 54.5 | 75.9 |
| GenRet (Sun et al., 2023) | 68.1 | 88.8 | *62.5* | 83.6 |
| Self Retrieval (Tang et al., 2024) | 73.3 | 92.6 | - | - |
| **Ours (EverMemModel)** | **75.5** | *90.6* | **66.5** | *83.5* |

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="2" align="center">NQ320K (Full text)</th>
      <th colspan="2" align="center">NQ320K (Unseen)</th>
    </tr>
    <tr>
      <th align="center"><strong>R@1</strong></th>
      <th align="center"><strong>R@10</strong></th>
      <th align="center"><strong>R@1</strong></th>
      <th align="center"><strong>R@10</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5"><strong>Sparse retrieval</strong></td>
    </tr>
    <tr>
      <td align="left">BM25 (Robertson & Zaragoza, 2009b)</td>
      <td align="center">29.7</td>
      <td align="center">60.3</td>
      <td align="center">32.3</td>
      <td align="center">61.9</td>
    </tr>
    <tr>
      <td align="left">DocT5Query (Cheriton, 2019)</td>
      <td align="center">38.0</td>
      <td align="center">69.3</td>
      <td align="center">48.5</td>
      <td align="center">72.9</td>
    </tr>
    <tr>
      <td colspan="5"><strong>Dense retrieval</strong></td>
    </tr>
    <tr>
      <td align="left">DPR (Karpukhin et al., 2020b)</td>
      <td align="center">50.2</td>
      <td align="center">77.7</td>
      <td align="center">50.0</td>
      <td align="center">74.2</td>
    </tr>
    <tr>
      <td align="left">ANCE (Xiong et al., 2021)</td>
      <td align="center">50.2</td>
      <td align="center">78.5</td>
      <td align="center">52.0</td>
      <td align="center">75.9</td>
    </tr>
    <tr>
      <td align="left">GTR-Base (Ni et al., 2021)</td>
      <td align="center">56.0</td>
      <td align="center">84.4</td>
      <td align="center">61.9</td>
      <td align="center">83.2</td>
    </tr>
    <tr>
      <td align="left">Sentence-T5 (Ni et al., 2022)</td>
      <td align="center">53.6</td>
      <td align="center">83.0</td>
      <td align="center">56.5</td>
      <td align="center">79.5</td>
    </tr>
    <tr>
      <td align="left">HCE-J (Chen et al., 2025)</td>
      <td align="center">71.2</td>
      <td align="center"><strong>93.9</strong></td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left">Qwen3-Embedding-0.6B (Zhang et al., 2025)</td>
      <td align="center">54.0</td>
      <td align="center">82.6</td>
      <td align="center">54.8</td>
      <td align="center">80.8</td>
    </tr>
    <tr>
      <td align="left">Qwen3-Embedding-4B (Zhang et al., 2025)</td>
      <td align="center">62.6</td>
      <td align="center">89.2</td>
      <td align="center">62.6</td>
      <td align="center"><strong>86.7</strong></td>
    </tr>
    <tr>
      <td colspan="5"><strong>Generative retrieval</strong></td>
    </tr>
    <tr>
      <td align="left">DSI-QG (Zhuang et al., 2022)</td>
      <td align="center">63.1</td>
      <td align="center">80.7</td>
      <td align="center">45.9</td>
      <td align="center">65.8</td>
    </tr>
    <tr>
      <td align="left">NCI (Wang et al., 2022)</td>
      <td align="center">66.4</td>
      <td align="center">85.7</td>
      <td align="center">54.5</td>
      <td align="center">75.9</td>
    </tr>
    <tr>
      <td align="left">GenRet (Sun et al., 2023)</td>
      <td align="center">68.1</td>
      <td align="center">88.8</td>
      <td align="center">62.5</td>
      <td align="center">83.6</td>
    </tr>
    <tr>
      <td align="left">Self Retrieval (Tang et al., 2024)</td>
      <td align="center">73.3</td>
      <td align="center">92.6</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="left"><strong>Ours (EverMemModel)</strong></td>
      <td align="center"><strong>75.5</strong></td>
      <td align="center">90.6</td>
      <td align="center"><strong>66.5</strong></td>
      <td align="center">83.5</td>
    </tr>
  </tbody>
</table>
-->

### Question Answering Performance (MS MARCO)

EverMemModel significantly outperforms both strong RAG baselines and large-context models.
<!--

| Dataset | Docs | Qwen3RAG-QA | | | Gemini-2.5-Flash | EverMemModel (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| | | **R@1** | **R@5** | **R@10** | | |
| MS MARCO (0.8M Tokens) | 8,389 | 2.235 | 2.535 | 2.548 | 2.710 | **3.812** |
| MS MARCO (7.1M Tokens) | 75,574| 2.225 | 2.521 | 2.759 | N/A† | **2.774** |

† *Input exceeds the model's maximum context length.*

-->
<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Dataset</th>
      <th rowspan="2" align="center">Docs</th>
      <th colspan="3" align="center">Qwen3RAG-QA</th>
      <th rowspan="2" align="center">Gemini-2.5-Flash</th>
      <th rowspan="2" align="center">EverMemModel (Ours)</th>
    </tr>
    <tr>
      <th align="center"><strong>R@1</strong></th>
      <th align="center"><strong>R@5</strong></th>
      <th align="center"><strong>R@10</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">MS MARCO (0.8M Tokens)</td>
      <td align="center">8,389</td>
      <td align="center">2.235</td>
      <td align="center">2.535</td>
      <td align="center">2.548</td>
      <td align="center">2.710</td>
      <td align="center"><strong>3.812</strong></td>
    </tr>
    <tr>
      <td align="left">MS MARCO (7.1M Tokens)</td>
      <td align="center">75,574</td>
      <td align="center">2.225</td>
      <td align="center">2.521</td>
      <td align="center">2.759</td>
      <td align="center">N/A†</td>
      <td align="center"><strong>2.774</strong></td>
    </tr>
  </tbody>
</table>

† Input exceeds the model's maximum context length.
<!--
Notably, EverMemModel achieves this with an average adaptive recall of just **2.5 documents**, showcasing its superior efficiency and precision compared to fixed-size RAG retrieval.

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
