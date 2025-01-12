import mlflow
import mlflow.sklearn
import platform
import psutil
import time
from datetime import datetime
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def prepare_data():
    """Prepare the data for training"""
    # Read Dataset
    df = pd.read_csv("./assets/dataset/heart.csv")
    
    # Prepare features and target
    X = df.drop(["target"], axis=1)
    y = df["target"]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test

def log_system_info():
    """Log system information as tags in MLflow"""
    return {
        "os_name": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cpu_count": psutil.cpu_count(),
        "runtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_user_parameters():
    """Get model parameters from user input"""
    print("\nEnter KNN model parameters:")
    
    # Get n_neighbors
    while True:
        try:
            n_neighbors = int(input("Enter number of neighbors (e.g., 3, 5, 7): "))
            if n_neighbors > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get weights
    while True:
        weights = input("Enter weight function ('uniform' or 'distance'): ").lower()
        if weights in ['uniform', 'distance']:
            break
        print("Please enter either 'uniform' or 'distance'.")
    
    # Get metric
    while True:
        metric = input("Enter distance metric ('euclidean', 'manhattan', or 'chebyshev'): ").lower()
        if metric in ['euclidean', 'manhattan', 'chebyshev']:
            break
        print("Please enter a valid metric.")
    
    # Get algorithm
    while True:
        algorithm = input("Enter algorithm ('auto', 'ball_tree', 'kd_tree', or 'brute'): ").lower()
        if algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            break
        print("Please enter a valid algorithm.")
    
    return {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'metric': metric,
        'algorithm': algorithm
    }

def train_and_evaluate_model(X_train, X_test, y_train, y_test, params):
    """Train and evaluate model with given parameters"""
    model = KNeighborsClassifier(**params)
    
    # Record training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cr_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, y_pred, {
        "accuracy": accuracy,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "training_time": training_time,
        "confusion_matrix": cm,
        "classification_report": cr_dict
    }

def run_single_experiment():
    """Run a single experiment with user-defined parameters"""
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data()
    
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    mlflow.set_experiment('Heart Disease Prediction1')
    
    # Get parameters from user
    params = get_user_parameters()
    
    # System information to log
    system_info = log_system_info()
    
    with mlflow.start_run():
        # Log system information
        for key, value in system_info.items():
            mlflow.set_tag(key, value)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train and evaluate model
        model, y_pred, metrics = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, params
        )
        
        # Log basic metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("cv_mean", metrics["cv_mean"])
        mlflow.log_metric("cv_std", metrics["cv_std"])
        mlflow.log_metric("training_time", metrics["training_time"])
        
        # Log detailed classification metrics
        cr_dict = metrics["classification_report"]
        cm = metrics["confusion_matrix"]
        
        # Log metrics for each class
        for class_label in ['0', '1']:
            prefix = f"class_{class_label}"
            mlflow.log_metric(f"{prefix}_precision", cr_dict[class_label]['precision'])
            mlflow.log_metric(f"{prefix}_recall", cr_dict[class_label]['recall'])
            mlflow.log_metric(f"{prefix}_f1", cr_dict[class_label]['f1-score'])
            mlflow.log_metric(f"{prefix}_support", cr_dict[class_label]['support'])
        
        # Log confusion matrix values
        mlflow.log_metric("true_negatives", cm[0][0])
        mlflow.log_metric("false_positives", cm[0][1])
        mlflow.log_metric("false_negatives", cm[1][0])
        mlflow.log_metric("true_positives", cm[1][1])
        
        # Log the model with signature
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5]
        )
        
        # Print results
        print("\nExperiment Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"CV Mean: {metrics['cv_mean']:.4f}")
        print(f"Training Time: {metrics['training_time']:.2f} seconds")

def continue_running():
    """Ask if user wants to run another experiment"""
    while True:
        response = input("\nDo you want to run another experiment? (yes/no): ").lower()
        if response in ['yes', 'no', 'y', 'n']:
            return response.startswith('y')
        print("Please enter 'yes' or 'no'.")

def find_best_model(experiment_name):
    """Find the best performing model from all runs in an experiment"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Search all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    
    if not runs:
        raise ValueError("No runs found in the experiment")
    
    best_run = runs[0]
    print("\nBest Model Details:")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}")
    print(f"CV Mean: {best_run.data.metrics['cv_mean']:.4f}")
    print("Parameters:")
    for param_name, param_value in best_run.data.params.items():
        print(f"  {param_name}: {param_value}")
    
    return best_run

def register_best_model(experiment_name, model_name, description=None):
    """Register the best model from an experiment in the model registry"""
    try:
        client = MlflowClient()
        
        # Find the best performing run
        best_run = find_best_model(experiment_name)
        
        # Create or get registered model
        try:
            registered_model = client.create_registered_model(
                name=model_name,
                description=description or f"Best performing model from {experiment_name}"
            )
            print(f"\nCreated new registered model: '{model_name}'")
        except Exception as e:
            print(f"\nModel '{model_name}' already exists, creating new version")
        
        # Create new model version
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{best_run.info.run_id}/model",
            run_id=best_run.info.run_id,
            description=f"Model accuracy: {best_run.data.metrics['accuracy']:.4f}"
        )
        
        # Wait for model version to be ready
        print("Waiting for model version to be ready...")
        while model_version.status not in ["READY", "FAILED"]:
            model_version = client.get_model_version(model_name, model_version.version)
        
        if model_version.status == "READY":
            # Transition model version to production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            print(f"\nModel Registration Success:")
            print(f"Name: {model_name}")
            print(f"Version: {model_version.version}")
            print(f"Stage: Production")
            print(f"Run ID: {best_run.info.run_id}")
            
            # Log registration as tags in the run
            client.set_tag(best_run.info.run_id, "registered_model_name", model_name)
            client.set_tag(best_run.info.run_id, "registered_model_version", model_version.version)
            
            return model_version
        else:
            print(f"Error: Model version status is {model_version.status}")
            return None
            
    except Exception as e:
        print(f"Error registering model: {e}")
        return None

def manage_model_registry():
    """Interactive function to manage model registration"""
    print("\nModel Registry Management")
    print("-" * 50)
    
    # Get experiment name
    experiment_name = input("Enter the experiment name to find the best model from: ") or "Heart Disease Prediction1"
    
    # Get model name for registry
    model_name = input("Enter name for the registered model: ") or "heart_disease_classifier"
    
    # Get optional description
    description = input("Enter model description (optional): ")
    
    # Register the model
    print("\nRegistering best model...")
    registered_model = register_best_model(
        experiment_name=experiment_name,
        model_name=model_name,
        description=description
    )
    
    if registered_model:
        print("\nModel registered successfully!")
        
        # Ask if user wants to see all registered models
        if input("\nDo you want to see all registered models? (yes/no): ").lower().startswith('y'):
            client = MlflowClient()
            registered_models = client.search_registered_models()
            
            print("\nRegistered Models:")
            print("-" * 50)
            for rm in registered_models:
                print(f"\nModel: {rm.name}")
                print(f"Description: {rm.description}")
                print(f"Latest Version: {rm.latest_versions[0].version}")
                print(f"Stage: {rm.latest_versions[0].current_stage}")
                print(f"Creation Time: {datetime.fromtimestamp(rm.creation_timestamp/1000.0)}")

# Main execution
if __name__ == "__main__":
    print("Welcome to Heart Disease Prediction Model Experimentation")
    
    # Run experiments
    while True:
        run_single_experiment()
        if not continue_running():
            break
    
    print("\nAll experiments completed.")
    
    # Ask if user wants to register a model
    if input("\nDo you want to register the best performing model? (yes/no): ").lower().startswith('y'):
        manage_model_registry()
    
    print("\nProcess completed. You can view all details in MLflow UI.")
    print("MLflow UI URL: http://127.0.0.1:5000/")