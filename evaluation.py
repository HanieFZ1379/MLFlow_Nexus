import mlflow
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Sample data: list of features for each sample
data = [
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
    [69, 0, 3, 140, 239, 0, 1, 151, 0, 1.8, 2, 2, 2],
    [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],
    [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2],
    [71, 0, 1, 160, 302, 0, 1, 162, 0, 0.4, 2, 2, 2],
    [35, 1, 0, 120, 198, 0, 1, 130, 1, 1.6, 1, 0, 3],
    [52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3],
    [67, 0, 0, 106, 223, 0, 1, 142, 0, 0.3, 2, 2, 2],
    [60, 1, 2, 140, 185, 0, 0, 155, 0, 3, 1, 0, 2],
    [42, 0, 0, 102, 265, 0, 0, 122, 0, 0.6, 1, 0, 2],
    [51, 0, 0, 130, 305, 0, 1, 142, 1, 1.2, 1, 0, 3],
    [39, 1, 0, 118, 219, 0, 1, 140, 0, 1.2, 1, 0, 3],
    [68, 1, 0, 144, 193, 1, 1, 141, 0, 3.4, 1, 2, 3]
]

# Define column names
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Normalize the data (exclude non-numeric columns if necessary)
scaler = MinMaxScaler()

# Columns to normalize (numerical features only)
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Apply scaling
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # Save the normalized dataset to CSV
# df.to_csv('normlized_dataset.csv', index=False)

# print("Normalized dataset saved as 'normlized_dataset.csv'")

# Set the tracking URI for the MLflow server
# mlflow.set_tracking_uri("http://37.152.191.193:8080") 
mlflow.set_tracking_uri('http://127.0.0.1:5000')
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Step 1: Load the Deployed Model
logged_model = "models:/heart_disease_classifier/1"
loaded_model = mlflow.pyfunc.load_model(logged_model)

# # Step 2: Load the CSV Data
# df = pd.read_csv('normlized_dataset.csv')

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
            output_filename = "evaluated_result/evaluation.txt"
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
