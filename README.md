# Breast Cancer Classification using Support Vector Machines (SVM)

## Project Overview

This project implements a binary classification model to predict breast cancer diagnosis (Malignant vs Benign) using Support Vector Machines (SVM). The dataset used is the Breast Cancer Wisconsin dataset from Kaggle.

---

## Steps Performed

### 1. Data Loading and Preparation
- Loaded the dataset from a CSV file.
- Removed unnecessary columns (`id`).
- Encoded diagnosis labels: `M` → 1 (Malignant), `B` → 0 (Benign).
- Scaled features using `StandardScaler` for better model performance.
- Split the dataset into training and testing sets.

### 2. Training SVM Models
- Trained an SVM with a **linear kernel**.
- Trained an SVM with an **RBF kernel**.
- Evaluated the models on the test set to get accuracy scores.

### 3. Visualizing Decision Boundary
- Reduced data dimensionality to 2D using **PCA (Principal Component Analysis)**.
- Visualized decision boundaries of the SVM trained on 2D data to understand model separability.

### 4. Hyperparameter Tuning
- Used **GridSearchCV** to tune hyperparameters `C` (regularization) and `gamma` (kernel coefficient for RBF).
- Selected the best hyperparameters based on cross-validation accuracy.

### 5. Cross-Validation
- Performed 5-fold cross-validation to evaluate and compare the performance of:
  - Linear SVM
  - RBF SVM
- Provided mean accuracy scores to assess model robustness.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
