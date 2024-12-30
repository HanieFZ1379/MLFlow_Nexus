# Experiment Comparison

This document presents the results and insights from experiments conducted for heart disease prediction. The analysis includes metrics, parameter configurations, and recommendations for the best-performing model.

---

## 1. Best Run

### Key Highlights
- **Run ID**: `cfaf604b66b446eca5b178b7c76589b2`
- **Experiment ID**: `3`

### Metrics
- **Accuracy**: `0.918` (~91.8%)
  - Indicates the proportion of correct predictions.

- **Class-Specific Metrics**:
  - **Class 0 (No Disease)**:
    - **Precision**: Measures how many predictions of "No Disease" were correct.
    - **Recall**: Measures how many actual "No Disease" cases were identified.
    - **F1 Score**: Balances precision and recall.
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
- There is a noticeable trade-off in metrics when using other configurations, especially in precision and recall for Class 1.

---

## 3. Model Performance Comparison

| Run ID                           | Accuracy | Precision | Recall  | F1 Score  | Training Time |
|----------------------------------|----------|-----------|---------|-----------|---------------|
| cfaf604b66b446eca5b178b7c76589b2 | 91.8%    | 93.5%     | 90.6%   | 92.1%     | 12 mins       |
| 7af672bd9b2a4e7394b1e7fc7e2e5a90 | 89.7%    | 90.2%     | 88.3%   | 89.2%     | 10 mins       |
| b3d46cdd8e3d40fbad94c4d7308fd6e1 | 87.5%    | 88.1%     | 86.0%   | 87.0%     | 9 mins        |
| f4c1835538e547df96b43e793e1cf080 | 77.05%   | 78.13%    | 78.13%  | 78.13%    | 0.01 mins     |
| 3dae0fafb3a043f78c5811d0df1a6bec | 75.41%   | 77.42%    | 75.00%  | 76.19%    | 0.02 mins     |

---

## 4. Model Performance Comparison

The chart below summarizes the performance metrics for all runs:
- **Accuracy vs. Run Name**: Shows how accuracy varies across different runs.
  - The highest accuracy achieved was **91.8%** 
  - The lowest accuracy recorded was **75.41%** 

![Accuracy vs Run Name](compare_runs_images/accuracy_vs_run_name.jpg)


In the plot below, you can see the comparison of different metrics and their corresponding accuracy. As shown in the plot, the **Euclidean** distance metric performs the best, with the highest accuracy.

![Metric vs Accuracy Comparison](images/metric_vs_accuracy.jpg)

### Plot Analysis
- The **Euclidean** metric, as shown in the plot, leads to the highest accuracy, making it the best choice for our model.
- We can see that the **Euclidean** metric outperforms other distance metrics in terms of accuracy.

---


## 5. Next Steps

### Model Selection
- Deploy the model from the best run (`cfaf604b66b446eca5b178b7c76589b2`).
