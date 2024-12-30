# Heart Disease Prediction Model

This project contains two versions of the heart disease prediction model. Below is a comparison of the versions, highlighting the improvements and key metrics.

## Model Versions

### Version 1:
- **Accuracy**: 0.7541
- **Stage**: Archived
- **Description**: Initial model with basic features.
- **Key Metrics**:
  - **Class 1 F1**: 0.7619
  - **Accuracy**: 0.7541
  - **False Negatives**: 8
  - **Class 0 Precision**: 0.7333
  - **Class 1 Precision**: 0.7742
  - **Class 1 Recall**: 0.75
  - **Class 0 F1**: 0.7458
- **Parameters**:
  - Metric: Chebyshev
  - Weights: Uniform
  - n_neighbors: 3
  - Algorithm: Ball Tree

---

### Version 2:
- **Accuracy**: 0.9180
- **Stage**: Production
- **Description**: Added new features and performed hyperparameter tuning.
- **Key Metrics**:
  - **Class 1 F1**: 0.9206
  - **Accuracy**: 0.9180
  - **False Negatives**: 3
  - **Class 0 Precision**: 0.9
  - **Class 1 Precision**: 0.9355
  - **Class 1 Recall**: 0.9063
  - **Class 0 F1**: 0.9153
- **Parameters**:
  - Metric: Euclidean
  - Weights: Uniform
  - n_neighbors: 5
  - Algorithm: Auto

---

## Observations:
- **Version 2** shows a significant improvement in **accuracy** (16% increase) compared to **Version 1**, reaching **91.8%**.
- **Version 2** also shows improved **precision** and **recall** for both classes, especially **Class 1**:
  - **Class 1 F1** increased from **0.7619** in Version 1 to **0.9206** in Version 2.
  - **Class 1 Precision** increased from **0.7742** in Version 1 to **0.9355** in Version 2.
- **False Negatives** reduced from **8** in Version 1 to **3** in Version 2, showing better detection of positive cases.
- **Version 2** outperforms **Version 1** in all key metrics, which led to its promotion to **Production**.
- **Version 1** is now **Archived**.

## Model Performance Comparison

| Version | Accuracy | Class 1 F1 | Class 1 Precision | Class 1 Recall | Training Time | Stage      |
|---------|----------|------------|-------------------|----------------|---------------|------------|
| 1       | 0.7541   | 0.7619     | 0.7742            | 0.75           | 0.0186 mins   | Archived   |
| 2       | 0.9180   | 0.9206     | 0.9355            | 0.9063         | 0.0044 mins   | Production |

---

## Conclusion:
- **Version 2** is a significant improvement over **Version 1**, offering a more accurate model with better precision and recall for Class 1.
- **Version 1** has been archived, and **Version 2** has been promoted to production.
