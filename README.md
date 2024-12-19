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