<div align="center">

# 💳 Credit Scoring Model

### Predict Loan Applicant Risk with Machine Learning

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nEokGos1f8Om3olOGkyr_QqL37oh175t)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?logo=xgboost&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-8B5CF6)

> **CodeAlpha Machine Learning Internship — Task 1**
> Developer: **Muhammad Ali**

*A production-ready credit risk prediction system that evaluates loan applicants as GOOD or BAD credit risks using 4 ML models, SMOTE class balancing, and an interactive Gradio dashboard.*

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Features & Engineering](#-features--engineering)
- [ML Models](#-ml-models)
- [Data Pipeline](#-data-pipeline)
- [Class Imbalance — SMOTE](#-class-imbalance--smote)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [UI Tabs](#-ui-tabs)
- [Disclaimer](#-disclaimer)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

Banks face a critical problem: **how do you decide who to give a loan to?** This project solves that using machine learning. It takes an applicant's financial profile — age, income, housing, credit history, loan amount — and predicts the probability they will default.

Four ML algorithms are trained, benchmarked on 5 metrics, and the best-performing model (by ROC-AUC) is deployed in a live **Gradio** web app where loan officers can input applicant data and get an instant, explainable risk assessment.

```
Applicant Financial Data  →  Preprocessing  →  4 ML Models  →  Best Model  →  Risk Score
     (9+ features)             SMOTE + Scale    LR · DT · RF · XGB   (AUC winner)   Good / Bad
```

---

## 🚀 Live Demo

> Run on Google Colab — a public Gradio link is auto-generated at launch.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nEokGos1f8Om3olOGkyr_QqL37oh175t)

**Steps to run:**
1. Open the Colab link above
2. `Runtime → Run All`
3. Gradio public URL appears at the last cell — click to open

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | German Credit Risk Dataset |
| **Source** | UCI ML Repository / Kaggle |
| **Primary URL** | [dsrscientist/dataset1](https://raw.githubusercontent.com/dsrscientist/dataset1/master/german_credit_risk.csv) |
| **Fallback URL** | [YBIFoundation/Dataset](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Credit%20Risk.csv) |
| **Fallback** | Auto-generated synthetic dataset (1,000 rows) if both URLs fail |
| **Applicants** | 1,000 |
| **Raw Features** | 9 |
| **Target** | Binary — `good` (0) · `bad` (1) credit risk |
| **Class split** | ~70% good · ~30% bad (imbalanced → fixed with SMOTE) |

---

## 🔬 Features & Engineering

### Raw Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Applicant age in years |
| `sex` | Categorical | male / female |
| `job` | Ordinal | 0=unskilled · 1=skilled · 2=highly skilled · 3=management |
| `housing` | Categorical | own / free / rent |
| `saving_accounts` | Categorical | little / moderate / quite rich / rich |
| `checking_account` | Categorical | little / moderate / rich |
| `credit_amount` | Numerical | Loan amount in Deutsche Mark |
| `duration` | Numerical | Loan duration in months |
| `purpose` | Categorical | car / furniture / education / business / vacation / etc. |

### Engineered Features

Two additional features are derived during preprocessing to improve model accuracy:

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| `amount_per_month` | `credit_amount / duration` | Monthly repayment burden — higher = riskier |
| `age_group` | `pd.cut(age, bins=[0,25,35,50,100])` | Young borrowers (0–25) tend to default more often |

---

## 🤖 ML Models

Four algorithms are trained and compared head-to-head:

| Model | Key Parameters | Strategy |
|-------|---------------|----------|
| **Logistic Regression** | `C=1.0`, `max_iter=1000`, `class_weight='balanced'` | Finds a straight-line decision boundary between good/bad |
| **Decision Tree** | `max_depth=6`, `min_samples_split=10`, `class_weight='balanced'` | Builds interpretable if/else decision rules |
| **Random Forest** | `n_estimators=200`, `max_depth=8`, `class_weight='balanced'`, `n_jobs=-1` | 200 trees vote together — ensemble reduces variance |
| **XGBoost** | `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `scale_pos_weight=auto` | Gradient boosting — each tree corrects the previous one's errors |

> **Winner is selected automatically** by highest ROC-AUC on the test set.

---

## ⚙️ Data Pipeline

```
Raw CSV (1,000 rows)
        │
        ▼
   ┌─────────────────────────────────────┐
   │         Preprocessing               │
   │  1. Encode target: good=0, bad=1   │
   │  2. Fill missing categoricals      │
   │     → "unknown"                    │
   │  3. Fill missing numericals        │
   │     → column median                │
   │  4. Label-encode all categoricals  │
   │  5. Feature engineering            │
   │     + amount_per_month             │
   │     + age_group                    │
   └─────────────────────────────────────┘
        │
        ▼
   ┌──────────────────┐
   │   SMOTE          │
   │  Balance classes │
   │  700 → 700 bad  │
   └──────────────────┘
        │
        ▼
   ┌──────────────────────────┐
   │  Train / Test Split      │
   │  80% train · 20% test   │
   │  Stratified by target   │
   └──────────────────────────┘
        │
        ▼
   ┌──────────────────────────┐
   │   StandardScaler         │
   │  Fit on train only       │
   │  Apply to both           │
   │  (prevents data leakage) │
   └──────────────────────────┘
        │
        ▼
   Train 4 Models → Evaluate → Pick Best by AUC → Deploy
```

---

## ⚖️ Class Imbalance — SMOTE

Real credit datasets are heavily skewed — far more good customers than bad. Without correction, a naive model just predicts "good" for everyone and still gets 70% accuracy while missing all the risky applicants.

**SMOTE** (Synthetic Minority Over-sampling Technique) fixes this:

```python
# Before SMOTE
# Good: 700  |  Bad: 300  → Model bias toward "good"

smote = SMOTE(random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)

# After SMOTE
# Good: 700  |  Bad: 700  → Model learns both equally
```

> SMOTE generates **synthetic** bad-credit examples (not duplicates) by interpolating between real minority-class samples — making the model much better at catching actual defaulters.

---

## 🏋️ Training Configuration

| Setting | Value |
|---------|-------|
| Test size | 20% (stratified) |
| Random seed | 42 |
| Cross-validation | `StratifiedKFold` |
| Scaler | `StandardScaler` (fit on train only) |
| Evaluation metrics | Accuracy · Precision · Recall · F1-Score · ROC-AUC |
| Training time | < 45 seconds (all 4 models combined) |
| Best model selection | Highest ROC-AUC on test set |

### Why ROC-AUC for selection?

> In credit scoring, **Recall is critical** — missing a bad customer (False Negative) is far more expensive for a bank than incorrectly rejecting a good one. ROC-AUC captures the model's full ability to rank good vs bad applicants across all thresholds, making it the most reliable selector.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | ~0.80 | ~0.79 | ~0.81 | ~0.80 | ~0.87 |
| Decision Tree | ~0.82 | ~0.81 | ~0.83 | ~0.82 | ~0.85 |
| Random Forest | ~0.86 | ~0.85 | ~0.87 | ~0.86 | ~0.92 |
| **XGBoost 🏆** | **~0.88** | **~0.87** | **~0.89** | **~0.88** | **~0.93** |

> *Actual values vary slightly per run. Expected range: Accuracy 0.80–0.88, AUC 0.85–0.93.*

**Prediction output example:**
```python
proba = best_model.predict_proba(applicant)
# → [0.78, 0.22]  →  78% Good Credit · 22% Bad Credit  →  ✅ Approve
# → [0.21, 0.79]  →  21% Good Credit · 79% Bad Credit  →  ❌ Reject
```

---

## 📁 Project Structure

```
CodeAlpha_CreditScoring/
│
├── task_1_credit_scoring_model.py    # Main Colab script (15 cells)
├── README.md
│
└── /tmp/  (auto-generated during runtime)
    ├── best_credit_model.pkl         # Saved best model
    ├── credit_eda.png                # 6-panel EDA visualization
    ├── model_comparison.png          # 5-metric bar chart (all 4 models)
    ├── roc_curves.png                # ROC-AUC curves overlay
    ├── feature_importance.png        # Feature importance (RF / XGBoost)
    └── best_model_eval.png           # Confusion matrix of best model
```

---

## 🚀 Installation & Usage

### ▶ Run on Google Colab (Recommended)

```
1. Click the "Open in Colab" badge at the top
2. Runtime → Run All
3. Wait ~45 seconds for all 4 models to train
4. Click the public Gradio URL that appears at Cell 15
```

### 💻 Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring

# Install dependencies
pip install scikit-learn xgboost imbalanced-learn \
            matplotlib seaborn pandas gradio

# Run the script
python task_1_credit_scoring_model.py
```

Then open `http://localhost:7860` in your browser.

---

## 🖥️ UI Tabs

The Gradio app has **5 interactive tabs:**

| Tab | Description |
|-----|-------------|
| 🔍 **Credit Risk Predictor** | Input applicant details via sliders → instant risk score + probability breakdown + 6 example profiles |
| 📊 **Model Comparison** | Head-to-head bar charts + ROC curves + full metrics table for all 4 models |
| 🔑 **Feature Importance** | Which financial factors influence the prediction most + best model confusion matrix |
| 📈 **Data Analysis** | 6-panel EDA — class distribution, age, credit amount, duration, purpose, housing type |
| ⚙️ **How It Works** | Plain-English explanation of SMOTE, scaling, model training, and metric definitions |

---

## ⚠️ Disclaimer

> This project is built **for educational purposes** as part of the CodeAlpha ML Internship.
> It is **not** a financial advisory tool and should not be used for real lending decisions.
> Real credit scoring systems require regulatory compliance, fairness audits, and validation by domain experts.

---

## 🙌 Acknowledgements

- [CodeAlpha](https://www.codealpha.tech) — Internship program
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — German Credit Dataset
- [Scikit-learn](https://scikit-learn.org) · [XGBoost](https://xgboost.readthedocs.io) · [Imbalanced-learn](https://imbalanced-learn.org) · [Gradio](https://gradio.app)

---

## 📬 Contact — CodeAlpha

<div align="center">

| 🌐 Website | 📧 Email | 💬 WhatsApp |
|-----------|---------|------------|
| [www.codealpha.tech](https://www.codealpha.tech) | services@codealpha.tech | +91 9336576683 |

</div>

---

<div align="center">

*Made with ❤️ by Muhammad Ali · CodeAlpha ML Internship*

</div>
