# Experiment Comparison

## 1. Best Run

### Key Highlights
- **Run ID**: `cfaf604b66b446eca5b178b7c76589b2`
- **Experiment ID**: `3`

### Metrics
- **Accuracy**: `0.918` (~91.8%)
  - Indicates the proportion of correct predictions.

- **Class-Specific Metrics**:
  - **Class 0 (No Disease)**:
    - **Precision**: How many predictions of "No Disease" were correct.
    - **Recall**: How many actual "No Disease" cases were identified.
    - **F1**: A balance of precision and recall.
  - **Class 1 (Heart Disease)**:
    - **Precision**: `0.935` (~93.5%) — 93.5% of instances predicted as "Heart Disease" were correct.
    - **Recall**: `0.906` (~90.6%) — 90.6% of actual "Heart Disease" cases were identified.
    - **F1 Score**: `0.921` (~92.1%) — Balances precision and recall effectively.

### Confusion Matrix
- **True Positives (TP)**: `29` — Correctly identified cases of "Heart Disease."
- **True Negatives (TN)**: `27` — Correctly identified cases of "No Disease."
- **False Positives (FP)**: `2` — Cases incorrectly identified as "Heart Disease."
- **False Negatives (FN)**: `3` — Cases incorrectly identified as "No Disease."

### Parameters
- **n_neighbors**: `7`
- **weights**: `uniform`
- **metric**: `euclidean`

### Summary
This run achieved the **highest accuracy (91.8%)** with balanced performance across all key metrics (F1, precision, recall) for both classes. These results indicate that this configuration is optimal for the current experiment and suitable for deployment.

---

## 2. Detailed Metrics for All Runs

### Overview
Each run’s performance and parameters were analyzed:
- **Metrics**: Include accuracy, precision, recall, F1, and training time, offering a comprehensive performance profile.
- **Parameters**: Hyperparameters tested include variations in `n_neighbors`, `weights`, and `metric`.

### Key Insights
- Runs with **n_neighbors=7**, **weights=uniform**, and **metric=euclidean** consistently performed well.
- There is a noticeable trade-off in metrics when using other configurations, especially in precision and recall for Class 1 ("Heart Disease").

---

## Next Steps

### 1. Model Selection
Use the model from the best run (`cfaf604b66b446eca5b178b7c76589b2`) for deployment.

### 2. Future Improvements
- Conduct further experiments with additional features or alternative metrics to evaluate robustness.
- Explore ensemble methods or advanced algorithms to improve overall performance.
