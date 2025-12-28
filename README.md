# 🎓 Student Dropout & Academic Success Prediction

## 📌 Overview
This repository contains a **university-level Applied AI & Machine Learning project** focused on **education analytics**.  
The objective of the project is to predict **student academic outcomes** and **dropout risk** using supervised and unsupervised machine learning techniques.

The project is based on the **UCI Machine Learning Repository dataset: _Predict Students Dropout and Academic Success_** and was developed as part of an **academic university coursework**.

---

## 🎯 Problem Statement
To predict whether a student will:
- **Dropout**
- **Continue Enrollment**
- **Graduate**

based on academic performance, demographic attributes, and socio-economic factors, and to identify students who are at risk for early academic intervention.

---

## 📂 Dataset
- **Source:** UCI Machine Learning Repository  
- **Dataset Name:** Predict Students Dropout and Academic Success  
- **Target Classes:** `Dropout`, `Enrolled`, `Graduate`

> ⚠️ The dataset is **not included** in this repository and must be downloaded separately from UCI due to licensing restrictions.

---

## 🧠 Machine Learning Techniques Used

### 🔹 Learning Types
- **Supervised Learning**
  - Multiclass Classification
  - Binary Classification (Dropout vs Non-Dropout)
- **Unsupervised Learning**
  - K-Means Clustering for exploratory analysis

---

## 🛠️ Feature Engineering & Preprocessing
- Automatic separation of:
  - **Categorical features** (≤ 20 unique values)
  - **Numerical features**
- Preprocessing pipeline:
  - One-Hot Encoding for categorical variables
  - Standard Scaling for numerical variables
- Implemented using **ColumnTransformer**
- Final feature space: **~161 processed features**

---

## 🤖 Models Implemented

### 1️⃣ Multiclass Logistic Regression
- Predicts: `Dropout`, `Enrolled`, `Graduate`
- Accuracy ≈ **73%**
- Provides probabilistic and interpretable predictions

### 2️⃣ Multiclass Decision Tree
- Handles non-linear feature interactions
- Controlled using `max_depth`
- Accuracy ≈ **66%**

### 3️⃣ Binary Logistic Regression (Main Model)
- Binary target:
  - `Dropout` vs `Non-Dropout`
- Best performing model
- **Accuracy:** ~88%  
- **ROC-AUC:** ~0.90  
- Suitable for early dropout risk detection

---

## 📊 Unsupervised Analysis
- **K-Means Clustering** (K = 2, K = 3)
- **Principal Component Analysis (PCA)** for 2D visualization
- Used to identify natural groupings and risk patterns among students

---

## 📈 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve and AUC Score (Binary Classification)

---

## 📉 Visualizations
- PCA cluster plots
- Confusion matrices (Multiclass & Binary)
- ROC curve for binary logistic regression
- Feature importance plots

---

## 🗂️ Project Structure
