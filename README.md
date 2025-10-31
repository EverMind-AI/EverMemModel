# EverMemModel


[![arXiv](https://img.shields.io/badge/arXiv-2510.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2510.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **EverMemModel**.

## 📝 Abstract

Large Language Models (LLMs) struggle in knowledge-intensive domains that require deep, specialized knowledge. While Retrieval-Augmented Generation (RAG) is a common solution, its decoupled retrieve-then-read pipeline suffers from misaligned objectives and is prone to performance degradation from distractor documents. We propose **EverMemModel**, a unified, **end-to-end** trainable memory model. EverMemModel can handle memory contexts on the scale of **100 million tokens**. Our model achieves state-of-the-art results on both retrieval and question-answering benchmarks, significantly outperforming traditional RAG pipelines and long-context models.

## ✨ Key Contributions

-   **End-to-End Memory Model**: We propose **EverMemModel**, a unified architecture that seamlessly integrates retrieval and generation, moving beyond the limitations of decoupled RAG systems.
-   **State-of-the-Art Performance**: EverMemModel achieves SOTA performance on both the NQ320k retrieval benchmark and the MS MARCO question-answering task.
-   **Massive-Scale Context**: Thanks to its efficient architecture, EverMemModel is one of the first models capable of handling contexts up to **100M tokens**.



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


### Question Answering Performance (MS MARCO)

EverMemModel significantly outperforms both strong RAG baselines and large-context models.
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

## 🌐 Our Homepage

[**Click here to visit EverMind AI's official website**](https://www.evermind-ai.com/lander)
