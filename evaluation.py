import mlflow
import pandas as pd
import time
from sklearn.metrics import accuracy_score

# Set the tracking URI for the MLflow server
mlflow.set_tracking_uri("http://37.152.191.193:8080/") 
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Step 1: Load the Deployed Model
logged_model = "models:/heart_disease_classifier/2"  # Replace '2' with the correct version or stage
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Step 2: Load the CSV Data
df = pd.read_csv('normlized_dataset.csv')

# Ensure all columns are of type float64 to match the model schema
print(f"Columns in the DataFrame: {df.columns}")

# Drop the target column for predictions, explicitly check for 'target'
if "target" in df.columns:
    df = df.drop(columns=["target"])

# Check for missing columns (ensure model input columns are correct)
expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Ensure correct data types
df = df.astype(float)

# Step 3: Test Predictions
try:
    start_time = time.time()  # Record start time
    predictions = loaded_model.predict(df)
    end_time = time.time()  # Record end time

    # Calculate Latency
    latency = (end_time - start_time) / len(df)  # Average latency per sample

    # Display Predictions
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"Sample {i + 1}: {'Heart Disease' if prediction == 1 else 'No Disease'}")

    # Assuming you have true labels for accuracy calculation
    true_labels = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Average Latency per Prediction: {latency:.4f} seconds")

    # Step 4: Log Observations to MLflow and Save Evaluation to File
    # Define the experiment name and get the experiment ID
    experiment_name = "Heart Disease Prediction1"  # Replace with your experiment name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Experiment ID: {experiment_id}")

        # Log the results to MLflow and save the evaluation file
        with mlflow.start_run(experiment_id=experiment_id):
            # Log parameters and metrics
            mlflow.log_param("model_used", logged_model)
            mlflow.log_param("tracking_uri", mlflow.get_tracking_uri())
            mlflow.log_metric("accuracy", accuracy * 100)
            mlflow.log_metric("average_latency", latency)

            # Save predictions and the evaluation summary to a file
            output_filename = "evaluation.txt"
            with open(output_filename, "w") as log_file:
                log_file.write(f"Tracking URI: {mlflow.get_tracking_uri()}\n")
                log_file.write(f"Model Used: {logged_model}\n\n")

                log_file.write("Test Dataset:\n")
                log_file.write(df.to_string() + "\n\n")

                log_file.write("Predictions:\n")
                for i, prediction in enumerate(predictions):
                    log_file.write(f"Sample {i + 1}: {'Heart Disease' if prediction == 1 else 'No Disease'}\n")

                log_file.write(f"\nAccuracy: {accuracy * 100:.2f}%\n")
                log_file.write(f"Average Latency per Prediction: {latency:.4f} seconds\n")

            # Log the .txt file as an artifact
            mlflow.log_artifact(output_filename)
            print(f"Prediction results logged as an artifact in experiment '{experiment_name}'.")

    else:
        print(f"Experiment '{experiment_name}' not found.")

except Exception as e:
    print(f"Error during prediction: {e}")
