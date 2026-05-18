# AdaBoost Classifier - Disease Prediction App

A Machine Learning web application built using **Streamlit** and **AdaBoost Classifier** to predict disease risk based on patient health indicators and medical information.

## Live Demo

https://adaboost-classification1.streamlit.app/

---

# Project Overview

This project demonstrates a complete **Machine Learning Classification workflow** including:

- Data Collection
- Data Cleaning
- Outlier Detection and Treatment
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

The application predicts whether a patient has:

- High Disease Risk
- Low Disease Risk

based on medical features and health conditions.

AdaBoost is an ensemble boosting algorithm that improves classification performance by combining multiple weak learners into a strong classifier. :contentReference[oaicite:0]{index=0}

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

# Machine Learning Algorithm

## AdaBoost Classifier

AdaBoost (Adaptive Boosting) is a supervised ensemble learning algorithm used mainly for classification problems.

The algorithm works by:
- Training multiple weak classifiers sequentially
- Increasing focus on previously misclassified samples
- Combining all weak learners into one strong classifier

Scikit-learn describes AdaBoostClassifier as a meta-estimator that repeatedly adjusts weights toward difficult classification cases. :contentReference[oaicite:1]{index=1}

In this project:
- `1` → Disease Detected
- `0` → No Disease

---

# AdaBoost Formula

Core AdaBoost concept:

:contentReference[oaicite:2]{index=2}

Where:
- \(h_m(x)\) = weak learner
- \(\alpha_m\) = learner weight
- \(M\) = number of estimators

---

# Dataset Information

The dataset contains:

- 5000 rows
- 8 columns
- Balanced classes

## Features

| Feature | Description |
|---|---|
| Age | Patient age |
| BloodPressure | Blood pressure level |
| Cholesterol | Cholesterol level |
| Glucose | Blood glucose level |
| BMI | Body Mass Index |
| HeartRate | Heart rate |
| Smoking | Smoking habit |
| Disease | Target variable |

---

# Project Workflow

## 1. Data Preprocessing

- Removed duplicate rows
- Checked missing values
- Statistical analysis using `describe()`

---

## 2. Outlier Detection using IQR

Outliers were detected using the IQR (Interquartile Range) method.

Formula:

:contentReference[oaicite:3]{index=3}

Lower Bound:

:contentReference[oaicite:4]{index=4}

Upper Bound:

:contentReference[oaicite:5]{index=5}

---

## 3. Outlier Treatment

Detected outliers were treated by replacing extreme values using lower and upper bounds with NumPy operations.

---

## 4. Train-Test Split

Dataset split:
- 80% Training
- 20% Testing

---

## 5. Model Training

Used:

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
