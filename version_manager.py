import mlflow
from mlflow.tracking import MlflowClient

# Set the tracking URI for the MLflow server
mlflow.set_tracking_uri("http://37.152.191.193:8080")  # Replace with your MLflow server's URI

# Open the file to write the results
with open("version_comparison_results.txt", "w") as file:
    # Write the tracking URI to the file
    file.write(f"Tracking URI: {mlflow.get_tracking_uri()}\n\n")
    
    # Initialize the MLflow client
    client = MlflowClient()

    # Use search_registered_models to get the list of registered models
    models = client.search_registered_models()

    # Iterate over models to get the metrics (accuracy)
    for model in models:
        model_name = model.name
        file.write(f"Model Name: {model_name}\n")
        
        # Get the versions of the model
        model_versions = client.get_registered_model(model_name).latest_versions
        
        # Extract metrics for each version
        for version in model_versions:
            file.write(f"  Version: {version.version}, Stage: {version.current_stage}, Status: {version.status}\n")
            
            # Get the run associated with the version
            run_id = version.run_id
            
            # Fetch the metrics (e.g., accuracy) from the run
            run = client.get_run(run_id)
            
            # Print the accuracy and other metrics
            accuracy = run.data.metrics.get("accuracy", "Accuracy not found")
            file.write(f"    Accuracy of Model {model_name} Version {version.version}: {accuracy}\n")
        
        file.write("\n")

print("Results saved to version_comparison_results.txt")
