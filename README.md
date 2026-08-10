# Smart MCQ Solver Challenge

**Student Name:** Shobhit Raj  
**Roll Number:** 24f2008744  
**Course:** BSDA2001P - Introduction to DL and GenAI Project  
**W&B Project:** `24f2008744-t22026`  
**Live Web App:** [https://aashhmwmkw9vcydm4tvav8.streamlit.app/](https://aashhmwmkw9vcydm4tvav8.streamlit.app/)  

---

## 📌 Project Overview

This repository contains my final project submission for the **Smart MCQ Solver Challenge**. The objective of the competition is to build a machine learning system capable of evaluating multiple-choice question prompts and ranking the top three most probable correct answers ($A, B, C, D, E$), evaluated using **Mean Average Precision at 3 (MAP@3)**.

### Key Highlights & Results
* **Validation MAP@3 Score:** `0.9788` (Master Hybrid Ensemble)
* **Kaggle Leaderboard Score:** `0.74355` (Rank 1022 / 1,530 Teams)
* **Target Cutoff:** Exceeded the competition target threshold of `0.73`
* **Deployment:** Interactive web application built with Streamlit and deployed on Streamlit Community Cloud.

---

## ⚙️ Methodology & Model Summary

I experimented with four distinct model architectures and built a weighted hybrid ensemble:

1. **Pretrained Transformer Baseline (`all-MiniLM-L6-v2`):** Zero-shot cosine similarity between prompt and option embeddings (Validation MAP@3: `0.3996`).
2. **Custom PyTorch Deep Neural Network (`CustomMCQDeepNet`):** Multi-layer neural network built from scratch using 1D Batch Normalization, Dropout regularization, and Adam optimizer. Trained on a 15,011-dimensional input vector (Validation MAP@3: `0.9696`).
3. **Histogram Gradient Boosting Classifier:** Tree-based model trained on 11 engineered prompt-option statistical metrics (Validation MAP@3: `0.9496`).
4. **TF-IDF + Logistic Regression:** High-dimensional linear classifier trained on 15,000 word n-grams plus statistical features (Validation MAP@3: `0.9475`).
5. **Master Hybrid Ensemble:** Blended prediction logits from TF-IDF LR (45%), PyTorch DNN (35%), and HistGBC (20%) (Validation MAP@3: **`0.9788`**).

---

## 📊 Performance Comparison

All experiments were tracked using Weights & Biases (W&B Run: `v20-official-criteria-100pct`):

| Model Architecture | Top-1 Accuracy | Macro F1-Score | Validation MAP@3 | Cutoff (>0.73)? |
| :--- | :---: | :---: | :---: | :---: |
| MiniLM Dense Transformer | 0.2475 | 0.2457 | 0.3996 | No |
| TF-IDF + Logistic Regression | 0.9100 | 0.9112 | 0.9475 | Yes |
| HistGradientBoosting | 0.9125 | 0.9149 | 0.9496 | Yes |
| **PyTorch DNN (CustomMCQDeepNet)** | **0.9550** | **0.9571** | **0.9696** | **Yes** |
| **Master Hybrid Ensemble** | **0.9525** | **0.9538** | **`0.9788`** | **Yes (Winner)** |

---

## 📁 Repository Structure

* `app.py` — Streamlit web application script for real-time MCQ option ranking.
* `dl-24f2008744-notebook-t22026.ipynb` — Kaggle training notebook containing feature extraction, model definitions, training loops, and W&B logging.
* `mcq_solver_model.joblib` — Serialized model artifacts (TF-IDF vectorizer, StandardScaler, Logistic Regression).
* `requirements.txt` — Python dependencies for local setup and cloud deployment.
* `README.md` — Project documentation.

---

## 🛠️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/shobhitrajxyz/DL-GenAI-Project.git
cd DL-GenAI-Project
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to test the interactive solver!
