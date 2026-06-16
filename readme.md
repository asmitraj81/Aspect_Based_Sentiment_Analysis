# 🚀 Sentiment Analysis of Comments Received Through Social Media Platforms

## 📌 Overview

This project presents a Syntax-Aware Aspect-Based Sentiment Analysis (ABSA) system for analyzing user comments collected from social media platforms.

Unlike traditional sentiment analysis, which predicts a single sentiment for an entire sentence, this system identifies specific aspects mentioned in a sentence and determines their corresponding sentiment (Positive, Negative, or Neutral).

The model leverages RoBERTa for contextual language understanding and incorporates syntactic dependency information to improve aspect-level sentiment prediction.

---

## 🎯 Objectives

- Extract aspect terms from social media comments.
- Classify sentiment associated with each aspect.
- Improve contextual understanding using dependency parsing.
- Support fine-grained sentiment analysis for real-world applications.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Text Cleaning
- Tokenization
- Dependency Parsing using SpaCy
- Label Encoding

### 2. Feature Extraction
- Contextual embeddings generated using RoBERTa.
- Syntax-aware attention bias created from dependency relations.

### 3. Multi-Task Learning
The model jointly performs:

- Aspect Term Extraction (ATE)
- Aspect Sentiment Classification (ASC)

### 4. Model Architecture
- RoBERTa Base Encoder
- Syntax-Aware Transformer Layers
- Multi-Head Attention
- Classification Head

---

## 📊 Dataset

The project utilizes:

- `absa_30000.csv`
- `Restaurants_Train.xml`
- `Restaurants_Test_Gold.xml`

Dataset contains social media and review-based sentences labeled with aspect terms and sentiment categories.

### Sentiment Classes

- Positive
- Negative
- Neutral

---

## 🛠️ Tech Stack

### Languages
- Python

### Libraries & Frameworks
- PyTorch
- Transformers (Hugging Face)
- SpaCy
- Scikit-Learn
- Pandas
- NumPy
- Gradio

---

## 📈 Results

The proposed model demonstrated:

- Improved Macro F1 Score
- Better Aspect Extraction Performance
- Strong Sentiment Classification Accuracy
- Reduced Training Loss Across Epochs

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Weighted F1 Score
- ATE F1 Score

---

## 📂 Project Structure

```text
.
├── app.py
├── absa_30000.csv
├── Restaurants_Train.xml
├── Restaurants_Test_Gold.xml
├── README.md
└── requirements.txt
```

## ▶️ Installation

```bash
git clone https://github.com/asmitraj81/Aspect_Based_Sentiment_Analysis.git

cd Aspect_Based_Sentiment_Analysis

pip install -r requirements.txt
```

## ▶️ Run Project

```bash
python app.py
```

---

## 📚 Research Contribution

This work was developed as part of the research paper:

**"Sentiment Analysis of Comments Received Through Social Media Platforms"**

The study focuses on fine-grained sentiment understanding using Aspect-Based Sentiment Analysis techniques and syntax-aware transformer architectures.

---

## 👨‍💻 Authors

- Asmit Raj




## 📧 Contact

**Asmit Raj**

Machine Learning | NLP | Data Science

GitHub: https://github.com/asmitraj81
LinkedIn: https://linkedin.com/in/asmitraj81

---

⭐ If you find this project useful, consider giving it a star.
