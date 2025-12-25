# 🤖 LLM Chatbot Battle: User Preference Prediction

## 📌 Project Overview

Large Language Models (LLMs) are widely deployed in chatbot systems, yet not all generated responses are equally preferred by users. Understanding and predicting **user preference between competing chatbot responses** is critical for improving conversational quality and user satisfaction.

This project focuses on **predicting human preferences** between two LLM-generated responses given the same prompt. Using **real human-labeled data from the Chatbot Arena**, we benchmark a wide spectrum of models—ranging from **classical machine learning** to **deep learning** and **transformer-based architectures**—to identify the most effective approach for preference prediction.


## 📊 Dataset Description

The dataset is sourced from the **Chatbot Arena** competition, where users compare responses from two LLMs and select their preferred answer or declare a tie.

### Files
- `train.csv`: Labeled comparison data (80% train / 20% validation)
- `test.csv`: Unlabeled test data for prediction
- `example_submission.csv`: Submission format guide

### Dataset Characteristics
- **Shape:** (57,477, 9)
- **Target Labels:**  
  - Model A wins  
  - Model B wins  
  - Tie

### Features
| Feature | Description |
|------|------------|
| `id` | Unique row identifier |
| `model_a`, `model_b` | Identity of competing models |
| `prompt` | User input prompt |
| `response_a`, `response_b` | Model responses |
| `winner_model_a`, `winner_model_b`, `winner_tie` | One-hot encoded ground truth |

## 🧠 Methodology

### 🔍 Problem Formulation
- **Task:** 3-class classification  
- **Label Mapping:**  
  - A wins → `0`  
  - B wins → `1`  
  - Tie → `2`

### 🧪 Model Categories

#### 1️⃣ Classical Machine Learning
- Logistic Regression
- LightGBM  
- Feature Engineering:
  - TF-IDF representations
  - Prompt–response cosine similarity
  - Response length and token statistics

#### 2️⃣ Neural Networks
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Networks (CNN)
- Input:
  - Pretrained embeddings (e.g., BERT / SBERT)
- Techniques:
  - Standardization
  - Early stopping
  - Regularization

#### 3️⃣ Transformer-Based Models
- BERT / RoBERTa / DeBERTa
- Gemma (1B, 2B)
- Input:
  - `[prompt] [SEP] response_a [SEP] response_b`
- Strategies:
  - Fine-tuning with small learning rates
  - Quantization and LoRA adapters (Gemma)


## 📈 Evaluation Metrics

### 🔢 Classification Metrics
- **Accuracy** – Overall correctness
- **Precision** – Correct positive predictions
- **Recall** – Coverage of actual positives
- **F1 Score (Macro)** – Balanced performance across classes
- **Log-Loss** – Probability calibration quality
- **ROC-AUC (OvR)** – Multiclass discrimination ability
- **Confusion Matrix** – Per-class error analysis
- **Training Time** – Computational efficiency

These metrics provide both **performance** and **calibration** insights, essential for preference modeling tasks.


## 🛠️ Technologies Used

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - NumPy, Pandas
  - Scikit-learn
  - PyTorch
  - Matplotlib / Seaborn
  - Hugging Face Transformers


## 🎯 Conclusion

This project presents a comprehensive benchmark of **user preference prediction models** using real human judgments from Chatbot Arena. Results show that **transformer-based models**, particularly BERT-family and Gemma, consistently outperform classical and shallow neural approaches, albeit with higher computational cost.

By providing a systematic comparison across model families, this work establishes a **practical baseline for future preference-learning and alignment research**, bridging traditional ML and modern LLM fine-tuning strategies.
