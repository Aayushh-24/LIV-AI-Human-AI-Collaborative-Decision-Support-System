# 🔬 LIV-AI: Human-AI Collaborative Decision Support System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

LIV-AI is an enterprise-grade clinical decision support framework designed to predict liver disease risk using patient biochemical markers. Moving beyond traditional "black-box" machine learning, this system integrates **Explainable AI (XAI)**, **Uncertainty Quantification**, and a **Human-in-the-Loop (HITL) Active Learning** pipeline.

It empowers hepatologists with transparent, mathematically grounded diagnostic reasoning while allowing the model to continuously learn from expert clinician feedback.

---

## ✨ Key Features

* **High-Recall Predictive Engine:** Utilizes an optimized Random Forest ensemble achieving an 88.1% recall on the Indian Liver Patient Dataset (ILPD), strictly minimizing life-threatening false negatives.
* **Dual-Engine Explainability (XAI):** * **SHAP (Game Theory):** Generates global feature importance and local waterfall plots for exact marginal feature attribution.
  * **LIME (Local Perturbation):** Builds localized surrogate models to map precise decision boundaries for individual patients.
* **Uncertainty Quantification:** Calculates 95% Confidence Intervals via ensemble tree variance, outputting a dynamic "Reliability Score" to prevent algorithmic overconfidence.
* **HITL Active Learning Loop:** Features a built-in SQLite feedback repository. Clinicians can override erroneous AI predictions, dynamically triggering a retraining pipeline to mitigate concept drift and bias over time.
* **Automated Clinical Reporting:** Generates downloadable, professional medical PDF reports summarizing vitals, risk probability, and statistical confidence.

---

## 🛠️ Tech Stack

* **Frontend / UI:** Streamlit (with custom CSS/Glassmorphism)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Explainability:** SHAP (`TreeExplainer`), LIME (`LimeTabularExplainer`)
* **Database (Active Learning):** SQLite3
* **Data Visualization:** Matplotlib, Seaborn
* **Reporting:** FPDF

---

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/LIV-AI.git](https://github.com/yourusername/LIV-AI.git)
cd LIV-AI