---

# **Bankruptcy Prediction Model Using Machine Learning**

## **1. Project Overview**

Bankruptcy occurs when a company or legal entity becomes insolvent and cannot pay its debts. Early prediction of bankruptcy helps financial institutions, investors, and regulators reduce risk and make informed decisions.

This project builds a **machine learning model** to predict whether a company will become bankrupt based on historical financial data.

---

## **2. Objectives**

* Predict the probability of bankruptcy for companies using financial indicators.
* Handle **class imbalance** in the dataset (very few bankrupt cases).
* Identify the **most important features** affecting bankruptcy.
* Compare different machine learning models (Random Forest, Logistic Regression).

---

## **3. Dataset Description**

* Source: Taiwan Economic Journal (via Kaggle).
* Dataset includes **financial ratios** and metrics for multiple companies.
* Total records: 1,364
* Features: 95 numerical columns (already encoded and scaled).
* Target variable: `Bankrupt`

  * `0` → Company is healthy
  * `1` → Company went bankrupt

**Class Distribution (approximate):**

| Class | Count | Percentage |
| ----- | ----- | ---------- |
| 0     | 1320  | 96.8%      |
| 1     | 44    | 3.2%       |

> The dataset is highly imbalanced, which requires special handling.

---

## **4. Libraries Used**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import joblib
```

---

## **5. Data Preprocessing**

1. Checked for missing values:

```python
print(df.isnull().sum())
```

2. Scaled features (already done).
3. Handled **class imbalance** using **SMOTE**:

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

---

## **6. Feature Selection**

Used **Recursive Feature Elimination (RFE)** with Random Forest to select the most important 30 features:

```python
model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=30)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print(selected_features)
```

---

## **7. Model Training**

### **Random Forest Classifier**

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
```

### **Logistic Regression (for comparison)**

```python
log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_model.fit(X_train_res, y_train_res)
y_pred_log = log_model.predict(X_test)
```

---

## **8. Model Evaluation**

### **Random Forest Results**

| Metric              | Value |
| ------------------- | ----- |
| Accuracy            | 0.955 |
| Recall (class 1)    | 0.61  |
| Precision (class 1) | 0.38  |
| F1-score (class 1)  | 0.47  |
| ROC-AUC             | 0.95  |

* Confusion Matrix:

```
[[1276   44]
 [  17   27]]
```

> Random Forest performs well, especially in detecting bankrupt companies.

### **Logistic Regression Results**

| Metric              | Value |
| ------------------- | ----- |
| Accuracy            | 0.918 |
| Recall (class 1)    | 0.16  |
| Precision (class 1) | 0.09  |
| F1-score (class 1)  | 0.11  |
| ROC-AUC             | 0.56  |

> Logistic Regression struggles due to **class imbalance** and non-linear feature relationships.

---

## **9. Visualizations**

### **Confusion Matrix**

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()
```

### **Feature Importance (Random Forest)**

```python
importances = pd.Series(rf_model.feature_importances_, index=X_train_res.columns)
importances.sort_values(ascending=False).head(15).plot(kind='barh')
plt.title("Top 15 Important Features")
plt.show()
```

### **ROC and Precision–Recall Curves**

```python
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.show()
```

---

## **10. Key Observations**

* The dataset is highly imbalanced (very few bankrupt cases).
* **Random Forest with SMOTE** significantly improves recall for bankruptcies.
* Logistic Regression is not suitable without advanced feature engineering.
* The most important features (financial ratios) heavily influence bankruptcy risk.
* ROC-AUC of 0.95 indicates strong model discrimination.

---

## **11. Conclusion**

* The **Random Forest Classifier with SMOTE** is the best performing model for this project.
* The model can detect the majority of bankruptcies while maintaining high overall accuracy.
* This solution can assist banks, investors, and regulators in early identification of financial risks.

---

## **12. Future Work**

1. Test **XGBoost or LightGBM** to further improve recall.
2. Explore **ensemble methods** combining multiple models.
3. Tune thresholds to balance precision and recall based on business needs.
4. Incorporate more features like market, macroeconomic indicators for better prediction.

---

## **13. References**

* Taiwan Economic Journal (TEJ) Dataset on Kaggle
* Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Imbalanced-learn (SMOTE): [https://imbalanced-learn.org](https://imbalanced-learn.org)

---
