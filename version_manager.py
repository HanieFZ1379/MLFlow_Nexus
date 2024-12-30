import mlflow
from mlflow.tracking import MlflowClient

# The tracking URI for the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Initialize the MLflow client
client = MlflowClient()

# Specify the model name to analyze
model_name = "heart_disease_classifier"

# Define the experiment name and get the experiment ID
experiment_name = "Heart Disease Prediction1"  
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")

    # Collect all printed data into a string variable to write to the log file
    log_data = []

    # Add header information
    log_data.append(f"Tracking URI: {mlflow.get_tracking_uri()}\n\n")
    log_data.append(f"Experiment ID: {experiment_id}\n")
    log_data.append(f"Experiment Name: {experiment_name}\n\n")

    # Fetch registered models
    models = client.search_registered_models()
    for model in models:
        model_info = f"Model Name: {model.name}\n"
        log_data.append(model_info)
        print(model_info)

        # Fetch model versions for the specified model
        if model.name == model_name:
            model_versions = client.search_model_versions(f"name='{model.name}'")
            for version in model_versions:
                version_info = f"  Version: {version.version}, Stage: {version.current_stage}, Status: {version.status}\n"
                log_data.append(version_info)
                print(version_info)

                # Fetch metrics and parameters using the run ID
                run_data = client.get_run(version.run_id)
                metrics = run_data.data.metrics
                params = run_data.data.params

                # Log metrics
                log_data.append(f"    Metrics:\n")
                print("    Metrics:")
                for metric, value in metrics.items():
                    metric_info = f"      {metric}: {value}\n"
                    log_data.append(metric_info)
                    print(metric_info)

                # Log parameters
                log_data.append(f"    Parameters:\n")
                print("    Parameters:")
                for param, value in params.items():
                    param_info = f"      {param}: {value}\n"
                    log_data.append(param_info)
                    print(param_info)

    # Write all collected data to the log file at once
    log_file_path = "version_manager.txt"
    with open(log_file_path, "w") as log_file:
        log_file.writelines(log_data)

    # Start an MLflow run and log the file as an artifact
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_artifact(log_file_path)
        print(f"Version comparison results logged as an artifact in experiment '{experiment_name}'.")

    print("\nVersion comparison log has been saved to 'version_manager.txt'")

else:
    print(f"Experiment '{experiment_name}' not found.")
