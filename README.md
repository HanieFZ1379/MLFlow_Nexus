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