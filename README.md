<!-- PREMIUM README -->

<div align="center">

# 🎓 Student Attendance Predictive System

### 🤖 ML-Powered Attendance Prediction • 📊 Risk Analysis • ⚡ Real-Time Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit\&logoColor=white)](https://studentattendancepredictor.streamlit.app)
[![Supabase](https://img.shields.io/badge/Supabase-Connected-3ECF8E?logo=supabase\&logoColor=white)](https://supabase.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-0076B6)](https://xgboost.readthedocs.io)

### 🔗 [Live Demo](https://studentattendancepredictor.streamlit.app)

</div>

---

## 🚀 Overview

This project is a **real-world academic analytics system** that predicts student attendance using Machine Learning.

It helps institutions:

* Identify **at-risk students early**
* Monitor attendance trends in real-time
* Take **data-driven actions** to improve academic performance

👉 Built as an **end-to-end ML pipeline + live dashboard system**

---

## 🧠 Problem Statement

Low attendance directly impacts student performance and outcomes.

Traditional systems:

* Only show past attendance ❌
* Do not provide predictive insights ❌

👉 This project solves that by enabling **predictive + prescriptive analytics**

---

## 💡 Solution

The system:

* Collects and processes attendance data
* Applies **ML models to predict future attendance**
* Generates **gap reports & risk levels**
* Visualizes insights via a **live Streamlit dashboard**

---

## ✨ Key Features

| Feature                       | Description                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------- |
| 🤖 **5 ML Models**            | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| ⚖️ **SMOTE Balancing**        | Handles class imbalance (84% Present / 16% Absent)                            |
| 🔧 **14 Engineered Features** | Includes lag features with no data leakage                                    |
| 🧠 **SHAP Explainability**    | Global + local model interpretation                                           |
| 📊 **Attendance Gap Report**  | Risk classification: Safe / Caution / At Risk / Detained                      |
| 🔮 **Next Class Prediction**  | Predicts attendance with probability score                                    |
| 📡 **Live Backend**           | Supabase real-time integration                                                |
| 📝 **Teacher Input System**   | Direct attendance entry via dashboard                                         |

---

## 🏗️ System Architecture

```
Student Data → Feature Engineering → ML Models → Predictions → Dashboard (Streamlit)
                                      ↓
                              Supabase Database
```

---

## 🏆 Model Performance

| Model                  | F1-Score   | Train F1 | Overfit Gap |
| ---------------------- | ---------- | -------- | ----------- |
| **GradientBoosting** ⭐ | **0.7810** | 0.819    | +0.038      |
| XGBoost                | 0.7632     | 0.778    | +0.015      |
| DecisionTree           | 0.6418     | 0.648    | +0.006      |
| RandomForest           | 0.6214     | 0.721    | +0.099      |
| LogisticRegression     | 0.3913     | 0.394    | baseline    |

👉 **Gradient Boosting selected as final model (best generalization)**

---

## 📸 Dashboard Preview

(Add your screenshots here — already good 👍)

---

## 📊 Workflow

```
Data Collection → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

---

## 🧩 Project Structure

```
student-attendance-predictive-system/
├── app.py
├── main.py
├── ml/
├── model/
├── scripts/
├── supabase/
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Ankur0307/student-attendance-predictor.git
cd student-attendance-predictor

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 🖥️ CLI Usage

```bash
python main.py --mode train
python main.py --mode gap --student ST1001 --remaining 20
python main.py --mode predict --student ST1001
python main.py --mode explain
```

---

## 🔧 Tech Stack

| Layer          | Tools                        |
| -------------- | ---------------------------- |
| ML             | scikit-learn, XGBoost, SMOTE |
| Explainability | SHAP                         |
| Backend        | Supabase                     |
| Dashboard      | Streamlit                    |
| Data           | pandas, numpy                |
| Visualisation  | matplotlib, seaborn          |

---

## 💼 Real-World Impact

This system can be used by:

* 🎓 Schools & colleges to track attendance trends
* ⚠️ Identify students at risk before it’s too late
* 📈 Improve overall academic performance

---

## 🌟 Why This Project Stands Out

* End-to-end ML system (not just a notebook)
* Real-time dashboard + backend integration
* Explainable AI using SHAP
* Production-ready architecture

---

<div align="center">

⭐ If you like this project, give it a star!

</div>
