import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Set the tracking URI for MLflow server
# mlflow.set_tracking_uri('http://37.152.191.193:8080')
# If you don't want to set up an MLflow tracking server, use the default local file-based tracking
mlflow.set_tracking_uri('http://127.0.0.1:5000')
# Experiment name
experiment_name = 'Heart Disease Prediction1'

# Get experiment by name
experiment = mlflow.get_experiment_by_name(experiment_name)
print(experiment)
if experiment is None:
    print(f"Experiment '{experiment_name}' not found. Please check the experiment name.")
else:
    # Load experiment data
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Analyze metrics and parameters
    if not runs.empty:
        # Find the best run based on accuracy
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        print("Best Run:")
        print(best_run)

        # Save analysis to a report file (CSV format)
        report_file = "compare_runs.csv"
        runs.to_csv(report_file, index=False)

        # Get the model artifact for the best run
        best_run_id = best_run['run_id']
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{best_run_id}/model"  # Assuming the model is logged under the name 'model'
        model = mlflow.pyfunc.load_model(model_uri)

        # Log the parameters (e.g., n_neighbors, weights) and algorithm
        algorithm = best_run['params.algorithm'] if 'params.algorithm' in best_run else 'Unknown'
        n_neighbors = best_run['params.n_neighbors'] if 'params.n_neighbors' in best_run else 'Not Logged'
        weights = best_run['params.weights'] if 'params.weights' in best_run else 'Not Logged'

        print(f"Algorithm: {algorithm}")
        print(f"n_neighbors: {n_neighbors}")
        print(f"weights: {weights}")

        # Start an MLflow run to log artifacts
        with mlflow.start_run(experiment_id=experiment_id):
            # Log the report file as an artifact 
            mlflow.log_artifact(report_file)

            # Print detailed metrics for all runs and log them as an artifact
            print("\nDetailed Metrics for All Runs:")
            with open("compare_runs.txt", "w") as log_file:
                # Log best run details
                log_file.write(f"Best Run (Accuracy):\n")
                log_file.write(f"Run ID: {best_run_id}\n")
                
                # Correctly access metrics, params, and tags
                best_run_obj = client.get_run(best_run_id)
                log_file.write(f"Metrics: {best_run_obj.data.metrics}\n")
                log_file.write(f"Parameters: {best_run_obj.data.params}\n")
                log_file.write(f"Tags: {best_run_obj.data.tags}\n")
                log_file.write("=" * 60 + "\n")

                # Log all other runs as well
                for run in client.search_runs(experiment_ids=[experiment_id]):
                    run_id = run.info.run_id
                    metrics = run.data.metrics
                    params = run.data.params
                    tags = run.data.tags
                    
                    # Write detailed metrics for the other runs
                    log_file.write(f"Run ID: {run_id}\n")
                    log_file.write(f"Metrics: {metrics}\n")
                    log_file.write(f"Parameters: {params}\n")
                    log_file.write(f"Tags: {tags}\n")
                    log_file.write("=" * 60 + "\n")

                    # Flush the content to ensure it's written to the file immediately
                    log_file.flush()

                    # Also print to console for reference
                    print("Run ID:", run_id)
                    print("Metrics:", metrics)
                    print("Parameters:", params)
                    print("Tags:", tags)
                    print("=" * 60)

            # Log the detailed metrics text file as an artifact
            mlflow.log_artifact("compare_runs.txt")

        print("\nReport and detailed metrics saved and logged as artifacts in MLflow.")

    else:
        print("No runs found for this experiment.")

# Load experiment results (from MLflow or CSV)
data = pd.read_csv ('compare_runs.csv')
results = pd.DataFrame (data)
print(results)

# Line plot for accuracy comparison across all runs
results.plot(x='run_id', y='metrics.accuracy', kind='line', figsize=(10, 6), color='b', marker='o')

plt.title('Accuracy Comparison Across All Runs', fontsize=14)
plt.xlabel('Run ID', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig('compare_runs_images/Accuracy.png')  # Save as PNG file

# Histogram for distribution of accuracy across all runs
plt.figure(figsize=(10, 6))
plt.hist(results['metrics.accuracy'], bins=10, color='g', edgecolor='black', alpha=0.7)

plt.title('Distribution of Accuracy Across All Runs', fontsize=14)
plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig('compare_runs_images/Accuracy_distribiuation.png')  # Save as PNG file