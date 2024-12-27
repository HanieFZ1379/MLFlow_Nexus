import mlflow
import pandas as pd

mlflow.set_tracking_uri('http://37.152.191.193:8080/')
# Experiment name
experiment_name = 'Heart Disease Prediction1'

# Get experiment by name
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Experiment '{experiment_name}' not found. Please check the experiment name.")
else:
    # Load experiment data
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Analyze metrics and parameters
    if not runs.empty:
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        print("Best Run:")
        print(best_run)

        # Save analysis to a report
        runs.to_csv("experiment_results.csv", index=False)

        # Use MLflow Client to fetch additional details
        client = mlflow.tracking.MlflowClient()
        all_runs = client.search_runs(experiment_ids=[experiment_id])

        # Print detailed metrics for all runs
        print("\nDetailed Metrics for All Runs:")
        for run in all_runs:
            print("Run ID:", run.info.run_id)
            print("Metrics:", run.data.metrics)
            print("Parameters:", run.data.params)
            print("=" * 60)
    else:
        print("No runs found for this experiment.")