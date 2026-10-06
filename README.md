# Credit Card Fraud Detection

This notebook compares four ways to find fraud in a highly imbalanced credit card transaction dataset: Isolation Forest, Local Outlier Factor, XGBoost with SMOTE, and a neural-network autoencoder.

The work starts with class balance, transaction amounts, correlations, and a 10% working sample. It then compares unsupervised anomaly detection with supervised and reconstruction-based methods.

## Saved results

The notebook contains output for a dataset with 284,807 transactions and 492 fraud cases.

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 |
| --- | ---: | ---: | ---: | ---: |
| Isolation Forest | 99.74% | 0.26 | 0.27 | 0.26 |
| Local Outlier Factor | 99.66% | 0.02 | 0.02 | 0.02 |
| XGBoost | 99.92% | 0.79 | 0.71 | 0.75 |
| Autoencoder | 94.84% | 0.00 | 0.03 | 0.00 |

Accuracy can look impressive when fraud represents about 0.17% of the data. Precision, recall, and F1 for the fraud class give a more useful comparison. In the saved run, XGBoost found the best balance.

## Run the notebook

The dataset is not committed because of its size. Place a file named `creditcard.csv` in the repository root before you start.

```bash
git clone https://github.com/dilatedtime/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv .venv
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow
jupyter notebook CreditCardFraudDetection.ipynb
```

Run the cells in order. TensorFlow training can take several minutes, depending on your hardware.

This is an educational comparison, not a fraud decision system. A real deployment would need time-aware validation, threshold tuning around investigation costs, monitoring for drift, and a clear review process for flagged transactions.
