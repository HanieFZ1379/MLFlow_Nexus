# Machine Learning Workflow with MLflow
This repository contains the implementation of a comprehensive machine learning workflow, integrating MLflow for experiment tracking and model registry. The project is divided into below parts:

MLflow Integration:
- Tracking experiments.
- Managing models using the MLflow Model Registry.
- Deploying and testing registered models.
# MLflow Implementation
## Part 1: MLflow Tracking
1. Set up MLflow server:
    a: configure a cloud platform and install prerequisites for run MLflow tracking server
      1. Installed Python3.8+ on your machine.
      2. Dependencies installed:
    ```sudo apt update
    sudo apt upgrade -y
    sudo apt install python3-pip
    pip install mlflow
    ```

    b:Run MLflow tracking server on cloud platform and Ensure the server can store experiment data and artifacts

    ```
    mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
    ```
    
    - --backend-store-uri: Path to the database to store metadata. Using SQLite for simplicity.
    - --default-artifact-root: Location to store artifacts such as models, logs, etc.
    - --host: Set to 0.0.0.0 to allow access from other devices.
    - --port: Port on which the server will run.
    Now the MLflow UI is running, Navigate to http://37.152.191.193:8080/ .


# Steps For Visualizing Experiment Results

## Step 1: Open MLflow UI
- Open the MLflow UI at [http://37.152.191.193:8080](http://37.152.191.193:8080).

## Step 2: Navigate to the Experiment and Analyze Metrics
- **Locate Experiments**: From the UI's home screen, find the list of experiments and click the experiment name to view its runs.
- **Visualize Metrics**: Switch to the Chart View under the Metrics tab to compare trends across runs. Use this view to analyze performance metrics visually.
- **Download or Export Results Manually**:
  - **Download CSV**: Click "Download CSV" on the experiment page to save all run data (parameters, metrics, tags, etc.) for further analysis in tools like Excel or Python.
  - **Export Plots**: Use the export button near the plots to download chart images for presentations or reports.

## Step 3: View Experiment Runs
- The main experiment view displays a table with:
  - **Run ID**: A unique identifier for each run.
  - **Parameters**: Input parameters logged during training.
  - **Metrics**: Output metrics logged during training (e.g., accuracy).
  - **Artifacts**: Links to logged models or additional files.
- Use filters (above the table) to narrow down runs based on specific parameters or metrics:
  - Filter runs with accuracy > 0.90.
  - Sort by any column, such as accuracy, to identify the best-performing model.

## Step 4: Compare and Visualize Experiment Results
- **Compare Multiple Runs**:
  - **Select Runs**: Check the boxes next to the runs you want to compare and click "Compare" to open the comparison view.
  - **Comparison View**:
    - **Metrics and Parameters**: View side-by-side bar graphs or tables showing metrics (e.g., accuracy) and parameters, helping analyze their impact.
    - **Visualization Options**: Switch between table and graphical views or use line charts to analyze trends over time (if metrics are logged iteratively).
- **Visualize Metrics**:
  - **Open Metric Plots**: Click on a specific metric (e.g., accuracy) in the table to view its plot.
  - **Customize Plots**: Adjust axes, zoom into key regions, overlay multiple metrics (e.g., accuracy vs. loss), and choose "Line" or "Scatter" plot types for deeper insights.

## Step 5: Create a Comparison Report of Different Runs by Writing a Python Script
- **Create a new Python script**: `compare_runs.py`.
- **Install Dependencies**:
  ```bash
  pip install mlflow pandas
- **Run the Script**:
  ```bash
  python compare_runs.py

The script will print the best run and its associated metrics, including details like the run ID, parameters, and other metrics. It will also save a CSV file named experiment_results.csv in the same directory, containing all experiment runs for further analysis. Additionally, the script uses the MLflow Client to fetch and display detailed information about each run, such as metrics and parameters, enabling a deeper understanding of the experiment's results.

•  For visual analysis and quick access, use Download CSV from Step 2.
•  For deeper, programmatic analysis, use the Python script in Step 5.

## Step 6: Final Interpretation of Results
- Analyze the effect of different parameters on metrics
  
- Identify anomalies or patterns, such as:
  - Significant drop in accuracy with certain parameter combinations.
  - Plateauing of metric improvement.

# Experiment Comparison

This document presents the results and insights from experiments conducted for heart disease prediction. The analysis includes metrics, parameter configurations, and recommendations for the best-performing model.

---

## 1. Best Run

### Key Highlights
- **Run ID**: `031895e3b4a7479f8e4ff1ebf4e6478d`
- **Experiment ID**: `1`

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

follow compare_runs.csv 

---

## 4. Model Performance Comparison

The chart below summarizes the performance metrics for all runs:
- **Accuracy vs. Run Name**: Shows how accuracy varies across different runs.
  - The highest accuracy achieved was **91.8%** 
  - The lowest accuracy recorded was **77.04%** 


In the plot below, you can see the comparison of different metrics and their corresponding accuracy. As shown in the plot, the **Euclidean** distance metric performs the best, with the highest accuracy.

![Metric vs Accuracy Comparison](compare_runs_images/mtreics_vs_accuracy.png)


### Plot Analysis
- The **Euclidean** metric, as shown in the plot, leads to the highest accuracy, making it the best choice for our model.
- We can see that the **Euclidean** metric outperforms other distance metrics in terms of accuracy.

---


## 5. Next Steps

### Model Selection
- Deploy the model from the best run (`031895e3b4a7479f8e4ff1ebf4e6478d`).

---
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