# Credit Card Fraud Fraud Detection

## Overview

This repository implements a comprehensive anomaly detection pipeline to identify fraudulent credit card transactions in a highly imbalanced dataset. The workflow encompasses exploratory data analysis, unsupervised outlier detection, supervised classification with XGBoost, and deep learning-based autoencoders, combined with evaluation metrics and visualizations.

## Key Features

* Exploratory Data Analysis and Visualization
* Class Imbalance Assessment and Handling
* Unsupervised Outlier Detection: Isolation Forest, Local Outlier Factor
* Supervised Classification: XGBoost with SMOTE Oversampling
* Deep Learning Autoencoder for Anomaly Detection
* Hyperparameter Optimization using GridSearchCV
* Evaluation Reports: Accuracy, Precision, Recall, F1-Score
* Confusion Matrix Visualizations

## Data Description

* **Source File**: `creditcard.csv`
* **Dimensions**: 284,807 transactions × 31 features
* **Target Variable**: `Class` (0 = Normal, 1 = Fraud)
* **Features**: 28 anonymized principal components (`V1`–`V28`), `Time`, `Amount`

## Data Preprocessing

1. **Missing Values**

   * Verified absence of null values in all columns.

2. **Class Distribution Analysis**

   * Visualized the severe imbalance: 0.17% fraud vs. 99.83% normal transactions.

3. **Feature Exploration**

   * Histograms and log-scaled plots of `Amount` for fraud and normal classes.
   * Scatter plots of `Time` versus `Amount`.
   * Correlation heatmap to identify features most correlated with fraud.

4. **Data Sampling**

   * Randomly sampled 10% of the data for model development to reduce computational load.

## Modeling Approaches and Hyperparameters

### Unsupervised Outlier Detection

| Model                | Key Parameters                         | Methodology                                         |
| -------------------- | -------------------------------------- | --------------------------------------------------- |
| Isolation Forest     | `n_estimators=100`, `max_samples=100%` | Detects anomalies by random tree isolation.         |
| Local Outlier Factor | `n_neighbors=20`                       | Scores instances based on local density deviations. |

### Supervised Classification

| Model   | Hyperparameters                                                                                     | Notes                                                |
| ------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| XGBoost | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1` <br> `subsample=0.8`, `colsample_bytree=0.8` | Trained on SMOTE-oversampled data for class balance. |

### Deep Learning Autoencoder

| Component   | Configuration                                             | Description                                                       |
| ----------- | --------------------------------------------------------- | ----------------------------------------------------------------- |
| Autoencoder | `encoding_dim=14`, `layers=[input → 14 → 7 → 14 → input]` | Trained on normal transactions; threshold at 95th percentile MSE. |

## Evaluation Metrics

For each method, performance was evaluated using:

* **Accuracy**
* **Precision** (Fraud class)
* **Recall** (Fraud class)
* **F1-Score** (Fraud class)
* **Confusion Matrix**

| Model                | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
| -------------------- | -------: | ----------------: | -------------: | ---------------: |
| Isolation Forest     |   99.74% |              0.23 |           0.27 |             0.25 |
| Local Outlier Factor |   99.66% |              0.15 |           0.02 |             0.04 |
| XGBoost Classifier   |   99.92% |              0.85 |           0.71 |             0.77 |
| Autoencoder          |   94.84% |              0.06 |           0.03 |             0.04 |

## Execution

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/credit-card-anomaly.git
   cd credit-card-anomaly
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis notebook**

   ```bash
   jupyter notebook anomaly_detection.ipynb
   ```

## Results and Conclusion

* **XGBoost** achieved the highest performance with **99.92% accuracy** and **0.77 F1-score** for fraud detection.
* **Isolation Forest** and **LOF** demonstrate unsupervised detection capability but lower fraud recall.
* **Autoencoder** offers insights via reconstruction error but requires further tuning for class imbalance.

The pipeline showcases effective strategies for anomaly detection in imbalanced datasets, combining unsupervised, supervised, and deep learning approaches.
