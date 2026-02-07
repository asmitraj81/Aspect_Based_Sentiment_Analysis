# K-Tran ABSA: Syntax-Aware Aspect-Based Sentiment Analysis

K-Tran (Knowledge-Transfer) is a sophisticated NLP model designed for **Aspect-Based Sentiment Analysis (ABSA)**. Unlike standard sentiment analysis which classifies an entire sentence, this model identifies specific "aspects" (e.g., *food*, *service*, *price*) and assigns a sentiment to each individually.

This project implements a hybrid architecture combining the contextual power of **RoBERTa** with a custom **Syntax-Aware Transformer Encoder**.

## 🧠 How it Works: The Explanation

The core innovation of this project is the **K-Tran Encoder Layer**, which uses linguistic structure to improve machine learning performance.

### 1. Dependency Parsing (Syntax Matrix)
Most models treat a sentence as a flat sequence of words. This model uses **SpaCy** to perform dependency parsing. It identifies which words are grammatically linked (e.g., an adjective describing a specific noun). This is stored in a **Syntax Matrix**.



### 2. Aspect-Aware Attention
In the custom K-Tran layers, the **Syntax Matrix** is used as a "bias." During the Self-Attention mechanism, the model is mathematically "nudged" to pay more attention to words that are grammatically related. 
* **Example:** In *"The pizza we had yesterday at the park was great,"* the model uses the syntax bias to link "great" directly to "pizza," ignoring the noise of "yesterday" or "park."

### 3. Joint Task Learning
The model performs two tasks at once:
* **ATE (Aspect Term Extraction):** Identifying the tokens that make up an aspect (using B-I-O tagging).
* **Sentiment Classification:** Categorizing the sentiment of those aspects into Positive, Neutral, or Negative.

---

## 🚀 Key Features
* **Backbone:** `roberta-base` for high-performance embeddings.
* **Custom Layers:** 3-layer K-Tran Encoder with 8-head Aspect-Aware Attention.
* **Optimization:** Mixed Precision Training (`autocast`) and Gradient Accumulation for efficient GPU usage.
* **Interactive UI:** Integrated **Gradio** interface for real-time testing.

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.8+
* NVIDIA GPU (CUDA) recommended.

### Installation
```bash
pip install transformers torch scikit-learn lxml spacy gradio
python -m spacy download en_core_web_sm