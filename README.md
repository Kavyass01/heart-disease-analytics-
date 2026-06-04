# ❤️ Heart Disease Analytics & Prediction Platform

## 📌 Overview

Heart Disease Analytics & Prediction Platform is a comprehensive healthcare analytics application built using **Python**, **Streamlit**, **Machine Learning**, and **Interactive Data Visualization**.

The platform helps healthcare professionals, researchers, and students:

* Analyze heart disease datasets
* Discover hidden patterns and risk factors
* Visualize patient health metrics
* Train machine learning models
* Predict heart disease risk
* Generate explainable AI insights

The application provides an intuitive and modern dashboard with advanced analytics, interactive charts, and predictive intelligence.

---

# 🚀 Features

## 📊 Executive Dashboard

* Dataset Overview
* Patient Statistics
* Disease Prevalence Metrics
* Interactive KPI Cards
* Health Trend Monitoring

---

## 🔍 Data Exploration

* Dataset Preview
* Missing Value Analysis
* Feature Statistics
* Data Distribution Analysis
* Correlation Analysis

---

## 📈 Advanced Analytics

* Age Distribution Analysis
* Gender-wise Disease Comparison
* Cholesterol Risk Analysis
* Chest Pain Type Analysis
* Heart Rate Performance Metrics
* Exercise-Induced Angina Insights

---

## 🤖 Machine Learning Models

The platform supports:

### Logistic Regression

* Fast baseline model
* Easy interpretation

### Random Forest

* High accuracy
* Feature importance extraction

### XGBoost

* Advanced boosting algorithm
* Superior predictive performance

---

## 🧠 Explainable AI

Using SHAP (SHapley Additive Explanations):

* Feature Importance
* Local Explanations
* Global Explanations
* Model Transparency

---

## 🩺 Patient Risk Predictor

Input patient details:

* Age
* Sex
* Blood Pressure
* Cholesterol
* Heart Rate
* Chest Pain Type
* ECG Results

Receive:

* Risk Prediction
* Probability Score
* Clinical Insights

---

## 📄 Report Generation

Generate:

* Analytics Reports
* Model Evaluation Reports
* Patient Prediction Reports

---

# 🏗️ Project Architecture

```text
heart-disease-analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── heart.csv
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Data_Explorer.py
│   ├── 3_Advanced_Analytics.py
│   ├── 4_ML_Model.py
│   ├── 5_Patient_Predictor.py
│
├── models/
│   ├── train_model.py
│   ├── model.pkl
│
├── assets/
│   ├── logo.png
│   ├── style.css
│
├── utils/
│   ├── data_loader.py
│   ├── charts.py
│   ├── insights.py
│
└── reports/
```

---

# 📂 Dataset Description

The dataset contains medical attributes used for predicting heart disease.

| Feature  | Description             |
| -------- | ----------------------- |
| age      | Age of patient          |
| sex      | Gender                  |
| cp       | Chest Pain Type         |
| trtbps   | Resting Blood Pressure  |
| chol     | Cholesterol             |
| fbs      | Fasting Blood Sugar     |
| restecg  | Resting ECG             |
| thalachh | Maximum Heart Rate      |
| exng     | Exercise-Induced Angina |
| oldpeak  | ST Depression           |
| slp      | Slope of Peak Exercise  |
| caa      | Number of Major Vessels |
| thall    | Thalassemia             |
| output   | Target Variable         |

Target:

```text
0 = No Heart Disease
1 = Heart Disease
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/heart-disease-analytics.git
```

```bash
cd heart-disease-analytics
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Application starts at:

```text
http://localhost:8501
```

---

# 📊 Visualizations Included

## Dashboard

* KPI Cards
* Disease Distribution
* Patient Overview

## Statistical Analysis

* Histograms
* Box Plots
* Violin Plots
* Density Plots

## Correlation Analysis

* Heatmaps
* Correlation Matrix

## Comparative Analytics

* Gender vs Disease
* Age vs Disease
* Cholesterol vs Disease

## Machine Learning

* Confusion Matrix
* ROC Curve
* Precision Recall Curve
* Feature Importance

---

# 🧠 Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Evaluation
      ↓
Prediction
      ↓
Explainability
```

---

# 📈 Model Evaluation Metrics

The application evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

# 🔥 Sample Results

Typical Random Forest performance:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 85-92% |
| Precision | 84-90% |
| Recall    | 83-91% |
| F1 Score  | 84-90% |
| ROC-AUC   | 88-95% |

*Results may vary depending on train-test split.*

---

# 🛠️ Technology Stack

## Frontend

* Streamlit
* HTML
* CSS

## Backend

* Python

## Machine Learning

* Scikit-Learn
* XGBoost

## Visualization

* Plotly
* Matplotlib
* Seaborn

## Explainable AI

* SHAP

---

# 🌐 Streamlit Deployment

## Push Project to GitHub

```bash
git init

git add .

git commit -m "Initial commit"

git branch -M main

git remote add origin https://github.com/yourusername/heart-disease-analytics.git

git push -u origin main
```

---

## Deploy on Streamlit Cloud

1. Login to Streamlit Cloud
2. Connect GitHub account
3. Select repository
4. Choose:

```text
Main File:
app.py
```

5. Click Deploy

---

# 📌 Future Enhancements

* Deep Learning Models
* AutoML Integration
* Real-Time Health Monitoring
* FastAPI Deployment
* Docker Support
* CI/CD Pipeline
* Multi-Disease Prediction
* Clinical Recommendation Engine
* LLM-Based Medical Insights

---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork Repository
2. Create Feature Branch
3. Commit Changes
4. Push Branch
5. Open Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed for Healthcare Analytics, Machine Learning Research, and Educational Purposes.

---

⭐ If you found this project useful, please give it a star on GitHub.
