# For model tracking
import mlflow
import mlflow.sklearn
import platform
import psutil
from datetime import datetime
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name, model_name):
    """
    Registers the best model from a given experiment in MLflow's Model Registry.
    
    Parameters:
        experiment_name: Name of the MLflow experiment
        model_name: Name to register the model under
    Returns:
        registered_model_version: The registered model version object
    """
    client = MlflowClient()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(uri='http://37.152.191.193:8080/')
    
    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment '{experiment_name}' not found")
    
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]  # Order by accuracy descending
    )
    
    if not runs:
        raise Exception("No runs found in the experiment")
    
    # Get the run with the highest accuracy
    best_run = runs[0]
    
    # Create a registered model if it doesn't exist
    try:
        client.create_registered_model(model_name)
        print(f"Created new registered model '{model_name}'")
    except Exception as e:
        print(f"Model '{model_name}' already exists")
    
    # Register the model version
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{best_run.info.run_id}/heart_disease_model",
        run_id=best_run.info.run_id
    )
    
    # Wait for the model to be ready
    client.wait_for_model_version_to_be_ready(model_version.name, model_version.version)
    
    # Transition the model version to 'Production' stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"Model registered successfully (Version: {model_version.version})")
    print(f"Model metrics from best run:")
    print(f"Accuracy: {best_run.data.metrics.get('accuracy', 'N/A')}")
    print(f"CV Mean: {best_run.data.metrics.get('cv_mean', 'N/A')}")
    
    return model_version

def log_metrics_to_mlflow(model, y_test, y_pred, cv_scores, best_params):
    """
    Logs detailed metrics and parameters to MLflow.
    
    Parameters:
        model: Trained model
        y_test: True labels
        y_pred: Predicted labels
        cv_scores: Cross-validation scores
        best_params: Best parameters from GridSearchCV
    """
    # Set your MLflow tracking URI
    mlflow.set_tracking_uri(uri='http://37.152.191.193:8080/')
    mlflow.set_experiment('Heart Disease Prediction')
    
    with mlflow.start_run():
        # Log system information
        mlflow.log_param("OS", platform.system())
        mlflow.log_param("OS Version", platform.version())
        mlflow.log_param("Processor", platform.processor())
        mlflow.log_param("RAM (GB)", round(psutil.virtual_memory().total / (1024 ** 3), 2))
        mlflow.log_param("Runtime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Log model parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Get detailed classification metrics
        cr_dict = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Log basic metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        
        # Log metrics for each class
        for class_label in ['0', '1']:  # For binary classification
            prefix = f"class_{class_label}"
            mlflow.log_metric(f"{prefix}_precision", cr_dict[class_label]['precision'])
            mlflow.log_metric(f"{prefix}_recall", cr_dict[class_label]['recall'])
            mlflow.log_metric(f"{prefix}_f1", cr_dict[class_label]['f1-score'])
            mlflow.log_metric(f"{prefix}_support", cr_dict[class_label]['support'])
        
        # Log macro and weighted averages
        for avg_type in ['macro avg', 'weighted avg']:
            prefix = avg_type.replace(' ', '_')
            mlflow.log_metric(f"{prefix}_precision", cr_dict[avg_type]['precision'])
            mlflow.log_metric(f"{prefix}_recall", cr_dict[avg_type]['recall'])
            mlflow.log_metric(f"{prefix}_f1", cr_dict[avg_type]['f1-score'])
            mlflow.log_metric(f"{prefix}_support", cr_dict[avg_type]['support'])
        
        # Log confusion matrix values
        mlflow.log_metric("true_negatives", cm[0][0])
        mlflow.log_metric("false_positives", cm[0][1])
        mlflow.log_metric("false_negatives", cm[1][0])
        mlflow.log_metric("true_positives", cm[1][1])
        
        # Log cross-validation metrics
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{i+1}", score)
        
        # Log the model
        mlflow.sklearn.log_model(model, "heart_disease_model")

# First, log the metrics
log_metrics_to_mlflow(
    model=best_model_knn,
    y_test=y_test,
    y_pred=y_pred_knn,
    cv_scores=cv_scores,
    best_params=grid_clf_knn.best_params_
)

# Then, register the best model
registered_model = register_best_model(
    experiment_name="Heart Disease Prediction",
    model_name="heart_disease_classifier"
)