# Breast Cancer Classification using Support Vector Machines (SVM)

## Project Overview

This project implements a machine learning pipeline to classify breast cancer tumors as **Malignant** or **Benign** using Support Vector Machines (SVM). The dataset used is the Breast Cancer Wisconsin dataset from Kaggle.

---

## Step-by-Step Explanation

### 1. Load and Prepare the Dataset

**Theory:**  
Before training a model, we need to load the data, clean it, convert categorical labels into numbers, scale the features for uniformity, and split the dataset into training and testing subsets. Scaling is essential for algorithms like SVM that are sensitive to feature magnitudes.

**What We Did:**  
- Loaded the dataset CSV into a DataFrame.  
- Removed the `id` column as it does not contribute to prediction.  
- Encoded diagnosis labels: 'M' → 1 (Malignant), 'B' → 0 (Benign).  
- Scaled all features using `StandardScaler`.  
- Split the data into training (80%) and testing (20%) sets.

---

### 2. Train SVM Models with Linear and RBF Kernels

**Theory:**  
SVM tries to find the best decision boundary that separates the classes.  
- A **linear kernel** separates data with a straight line (or hyperplane).  
- An **RBF kernel** allows for non-linear boundaries by mapping data into higher-dimensional space.

**What We Did:**  
- Trained an SVM with a linear kernel on the training data.  
- Trained an SVM with an RBF kernel on the training data.  
- Evaluated both models on the test set to check accuracy.

---

### 3. Visualize Decision Boundary with 2D Data

**Theory:**  
Since the dataset has many features (30+), visualizing decision boundaries is impossible in higher dimensions. PCA reduces dimensionality to 2D, allowing us to plot and visually inspect how well the model separates the classes.

**What We Did:**  
- Applied Principal Component Analysis (PCA) to reduce data to two principal components.  
- Trained an SVM on the 2D PCA-transformed data.  
- Plotted the decision boundary along with data points to visualize class separation.

---

### 4. Hyperparameter Tuning: C and gamma

**Theory:**  
- **C** controls the trade-off between margin size and classification error.  
- **gamma** (for RBF kernel) defines how far the influence of a single training point reaches.  
Finding the right values for these parameters improves the model’s performance and prevents overfitting.

**What We Did:**  
- Used `GridSearchCV` to test different values of `C` and `gamma`.  
- Performed cross-validation to find the best hyperparameters.  
- Selected the best model based on validation accuracy.

---

### 5. Cross-Validation to Evaluate Performance

**Theory:**  
Cross-validation splits the dataset into multiple folds. Models are trained on some folds and tested on others repeatedly. This method provides a more reliable estimate of model performance than a single train-test split.

**What We Did:**  
- Used 5-fold cross-validation on both Linear and RBF SVM models.  
- Calculated accuracy scores for each fold.  
- Computed average accuracy to measure overall model robustness.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
