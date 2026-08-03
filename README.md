

```markdown
# 🧠 Smart MCQ Solver Challenge

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat&logo=weightsandbiases&logoColor=white)

## 📌 Overview
This repository contains the codebase for the **Smart MCQ Solver Challenge**. The objective is to build an intelligent AI/ML pipeline capable of solving complex multiple-choice questions. Given a question prompt and five possible answer choices (A, B, C, D, E), the system evaluates contextual reasoning to predict the top three most probable answers in ranked order. 

The project is evaluated using the **Mean Average Precision at 3 (MAP@3)** metric.

## ✨ Key Features
*   **Modular Preprocessing:** Automated extraction of 12 robust statistical/tabular features (character counts, word lengths, capitalization, punctuation, and prompt-option text overlap).
*   **TF-IDF Vectorization:** Custom text vectorization concatenated with tabular data to create dense representations of the prompts and options.
*   **Deep Learning Baseline:** A custom PyTorch Deep Neural Network (DNN) built from scratch with batch normalization and dropout layers.
*   **Retrieval-Augmented Generation (RAG):** Explores external knowledge retrieval using `FAISS` and dense embeddings (`all-MiniLM-L6-v2`), re-ranked with a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
*   **Zero-Shot Classification:** Experimentation with LLM inference using Hugging Face pipelines (`facebook/bart-large-mnli`).
*   **Master Ensemble:** A high-accuracy probability blending strategy combining Logistic Regression, HistGradientBoosting, and the PyTorch DNN.
*   **Experiment Tracking:** Fully integrated with Weights & Biases (WandB) for logging metrics, hyperparameters, and model performance.

## 📂 Repository Structure
```text
├── data/
│   ├── train.csv                # Training dataset with correct answers
│   ├── test.csv                 # Test dataset for inference
│   └── sample_submission.csv    # Kaggle submission format
├── notebooks/
│   └── dl-inference-pipeline.ipynb  # Main Jupyter Notebook with complete workflow
├── src/
│   ├── preprocess.py            # Feature engineering & TF-IDF extraction
│   ├── models.py                # PyTorch custom DNN and Sklearn models
│   ├── rag_pipeline.py          # FAISS index creation and Cross-Encoder ranking
│   └── ensemble.py              # Blending logic and submission generation
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation

```

## 📊 Dataset Description

The dataset is composed of complex reasoning questions spanning various difficulty levels and domains:

* **id:** Unique identifier for the question.
* **prompt:** The actual question or problem statement.
* **A, B, C, D, E:** The five possible answer choices.
* **answer:** The ground-truth correct option label (Only available in `train.csv`).

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/YourUsername/smart-mcq-solver.git](https://github.com/YourUsername/smart-mcq-solver.git)
cd smart-mcq-solver

```


2. **Create a virtual environment (Optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up Weights & Biases (WandB):**
Ensure you have your WandB API key ready.
```bash
wandb login

```



## 💻 Usage

To run the full end-to-end inference and generate a submission file:

1. Ensure `train.csv` and `test.csv` are placed in the root or `data/` directory.
2. Execute the notebook or run the main pipeline script:
```bash
python src/ensemble.py

```


3. The final predictions will be saved as `submission.csv`, containing the `id` and `prediction` (top 3 choices separated by spaces).

## 📈 Results & Performance

* **Current MAP@3 Score:** `0.74355`
* **Ensemble Weights:**
* 45% TF-IDF + Logistic Regression
* 35% Custom PyTorch DNN
* 20% HistGradientBoosting Classifier



## 🔮 Future Work

* Migrating from TF-IDF feature extraction to fine-tuning Causal Language Models (e.g., Llama-3, Mistral) using QLoRA.
* Integrating the FAISS RAG pipeline directly into the master ensemble to ground the predictions in factual context.
* Implementing `AutoModelForMultipleChoice` (DeBERTa-v3-large) for true semantic logit extraction.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

```

```
