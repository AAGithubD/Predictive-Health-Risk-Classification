# 🏥 Predictive Health Risk Classification

An end-to-end Machine Learning pipeline built for healthcare decision support to evaluate, classify, and predict individual **Health Risk Status** (`0: Healthy`, `1: Unhealthy / At-Risk`) based on biometric vitals, clinical lab panels, and lifestyle indicators.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Data Cleaning and EDA](#data-cleaning-and-eda)
5. [Preprocessing](#preprocessing)
6. [Models and Evaluation](#models-and-evaluation)
7. [Final Model](#final-model)
8. [Feature Importance](#feature-importance)
9. [Overfitting and Data Leakage Checks](#overfitting-and-data-leakage-checks)
10. [Project Structure](#project-structure)
11. [Limitations and Future Work](#limitations-and-future-work)
12. [Tech Stack](#tech-stack)

---

## Project Overview

NovaGen Research Labs needed a reliable way to flag individuals as healthy or unhealthy from their health records. This project covers the full machine learning lifecycle: data understanding and cleaning, exploratory analysis, preprocessing, model comparison, hyperparameter tuning, validation, and model export.

**Why Recall matters here:** in a health setting, a false negative (an unhealthy person classified as healthy) is more costly than a false positive. Accuracy, Precision, Recall, and F1 were all tracked, but **Recall** was given particular importance when judging models.

---

## Dataset

| Property | Value |
|---|---|
| Records | 9,549 |
| Columns | 23 (22 features + 1 target) |
| Target | `Target` (0 = Healthy, 1 = Unhealthy) |
| Class balance | 52.14% Unhealthy (4,979) / 47.86% Healthy (4,570) |
| Missing values | None |
| Duplicate rows | None |

**Feature groups**

- **Numeric (10):** `Age`, `BMI`, `Blood_Pressure`, `Cholesterol`, `Glucose_Level`, `Heart_Rate`, `Sleep_Hours`, `Exercise_Hours`, `Water_Intake`, `Stress_Level`
- **Categorical, integer-coded 0 to 2 (7):** `Smoking`, `Alcohol`, `Diet`, `MentalHealth`, `PhysicalActivity`, `MedicalHistory`, `Allergies`
- **Pre-encoded binary (5):** `Diet_Type__Vegan`, `Diet_Type__Vegetarian`, `Blood_Group_AB`, `Blood_Group_B`, `Blood_Group_O`

The classes are close to balanced, so no resampling was needed.

---

## Workflow

```
Data understanding -> Cleaning -> EDA -> Preprocessing -> Train/test split
      -> Model comparison -> Cross-validation -> GridSearchCV tuning
      -> Final evaluation -> Feature importance -> Leakage/overfitting checks
      -> Save model (Joblib)
```

---

## Data Cleaning and EDA

**Cleaning**

- Verified no missing values and no duplicate rows.
- Converted boolean columns to integers.
- Found 22 records with `Sleep_Hours = 0`, treated as invalid, replaced with `NaN`, and imputed with the median.
- Confirmed no sleep values below 0 or above 24 hours.
- Reviewed the `Age` distribution (0 to 100; 96 records at age 0) and kept those values as valid.
- Inspected outliers with boxplots for all numeric features. `Blood_Pressure` showed the most extreme values. These were retained as plausible health readings rather than removed.

**EDA highlights**

- Target distribution and age distribution by health status.
- Boxplots of `BMI`, `Blood_Pressure`, and `Glucose_Level` against the target.
- Correlation analysis. The strongest correlations with `Target`:

| Feature | Correlation with Target |
|---|---|
| BMI | +0.41 |
| Blood_Pressure | -0.38 |
| Cholesterol | -0.32 |
| Age | -0.19 |

- `Glucose_Level` showed almost no visible separation between classes on its own.

---

## Preprocessing

Implemented with scikit-learn `Pipeline` and `ColumnTransformer`, so the same transformations are applied consistently at training and prediction time.

| Feature type | Steps |
|---|---|
| Numeric (10) | `SimpleImputer(strategy="median")` then `StandardScaler()` |
| Categorical (7) | `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")` |

**Split:** 80% train / 20% test (`test_size=0.20`, `random_state=42`, `stratify=y`). The test set has 1,910 rows (914 Healthy, 996 Unhealthy).

---

## Models and Evaluation

Four classifiers were trained inside the same preprocessing pipeline and evaluated on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.9314** | **0.9187** | **0.9528** | **0.9354** | **0.9820** |
| Gradient Boosting | 0.9152 | 0.9017 | 0.9398 | 0.9204 | 0.9718 |
| Decision Tree | 0.8911 | 0.8893 | 0.9036 | 0.8964 | 0.8905 |
| Logistic Regression | 0.8141 | 0.8183 | 0.8273 | 0.8228 | 0.8874 |

Random Forest led on every metric, including Recall, so it was selected for further evaluation.

### 5-Fold Stratified Cross-Validation (Random Forest, F1)

| Metric | Value |
|---|---|
| Fold scores | 0.941, 0.931, 0.943, 0.935, 0.939 |
| Mean F1 | 0.9378 |
| Std | 0.0044 |

The low standard deviation indicates stable performance across folds.

---

## Final Model

Hyperparameter tuning used `GridSearchCV` (5-fold, scoring = F1) over:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}
```

**Best parameters:** `n_estimators=300`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1` (best CV F1 = 0.9366).

### Final test-set performance (tuned Random Forest)

| Metric | Score |
|---|---|
| Accuracy | 0.9293 |
| Precision | 0.9167 |
| Recall | 0.9508 |
| F1 Score | 0.9335 |
| ROC-AUC | 0.9818 |

**Classification report**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Healthy | 0.94 | 0.91 | 0.92 | 914 |
| Unhealthy | 0.92 | 0.95 | 0.93 | 996 |

**Confusion matrix:** 828 true negatives, 86 false positives, about 49 false negatives, and about 947 true positives.

The tuned model performs essentially the same as the default Random Forest (differences in the third decimal place), which suggests the default configuration was already close to optimal for this dataset.

---

## Feature Importance

Top features by Random Forest importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | BMI | 0.209 |
| 2 | Blood_Pressure | 0.150 |
| 3 | Cholesterol | 0.106 |
| 4 | Stress_Level | 0.081 |
| 5 | Glucose_Level | 0.070 |
| 6 | Age | 0.070 |
| 7 | Sleep_Hours | 0.068 |
| 8 | Heart_Rate | 0.048 |
| 9 | Water_Intake | 0.048 |
| 10 | Exercise_Hours | 0.027 |

Clinical measurements (BMI, blood pressure, cholesterol) dominate the predictions. The one-hot encoded lifestyle categories each contribute very little (about 0.006 each).

---

## Overfitting and Data Leakage Checks

- **Overfitting:** training and testing performance were compared, and cross-validation scores were checked for consistency (CV F1 of 0.938 versus test F1 of 0.933).
- **Data leakage:** the workflow was reviewed to confirm that all imputation, scaling, and encoding live inside the pipeline and are fit only on training data, and that the test set was never used for tuning.

---

## Project Structure

```
.
├── README.md
├── novagen_project.ipynb        # Full analysis and modeling notebook
├── novagen_dataset.csv          # Dataset (add your own path)
└── novagen_health_model.pkl     # Saved tuned Random Forest pipeline
```
---

**Load the saved model and predict**

```python
import joblib
import pandas as pd

model = joblib.load("novagen_health_model.pkl")

# new_data must contain the same feature columns used in training (everything except "Target")
new_data = pd.read_csv("new_records.csv")

predictions = model.predict(new_data)            # 0 = Healthy, 1 = Unhealthy
probabilities = model.predict_proba(new_data)[:, 1]
```

The saved object is a full pipeline, so raw input goes straight in. No manual scaling or encoding is needed. Use the same scikit-learn version as training to avoid compatibility issues when loading.

---

## Limitations and Future Work

- The ColumnTransformer uses only the 10 numeric and 7 categorical features. The 5 pre-encoded binary columns (diet type and blood group) are not passed through, so they are currently excluded from the model.
- `Age` includes infants and children, so health thresholds for features like blood pressure may differ across age groups. Age-aware features or segment-specific models could help.
- Predictions are a screening aid, not a medical diagnosis, and should be reviewed by qualified professionals.
- Possible next steps: threshold tuning to push Recall higher, calibration of predicted probabilities, SHAP-based explanations, testing XGBoost/LightGBM, and wrapping the model in an API or Streamlit app.

---

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, Jupyter Notebook
