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