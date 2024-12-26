# For model tracking 
import mlflow
import mlflow.sklearn
import platform
import psutil
from datetime import datetime


# model tracking function

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


log_metrics_to_mlflow(
    model=best_model_knn,
    y_test=y_test,
    y_pred=y_pred_knn,
    cv_scores=cv_scores,
    best_params=grid_clf_knn.best_params_
)