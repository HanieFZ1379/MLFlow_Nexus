import mlflow
import pandas as pd

# The tracking URI for the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Step 1: Load the Deployed Model
logged_model = "models:/heart_disease_classifier/2"  
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Step 2: Send Test Data
data = {
    "age": [0.4, 0.45],  # Normalize age (e.g., 37 -> 0.4, 41 -> 0.45)
    "sex": [0.1, 0.0],  # 0 for Female, 1 for Male
    "cp": [2.0, 1.0],  # 2 for Non-anginal pain, 1 for Atypical angina
    "trestbps": [1.3, 1.3],  # Normalize blood pressure (e.g., 130 -> 1.3)
    "chol": [2.5, 2.04],  # Normalize cholesterol (e.g., 250 -> 2.5, 204 -> 2.04)
    "fbs": [0.0, 0.0],  # 0 for False (Fasting Blood Sugar)
    "restecg": [1.0, 0.0],  # 1 for Abnormality, 0 for Normal
    "thalach": [1.87, 1.72],  # Normalize max heart rate (e.g., 187 -> 1.87, 172 -> 1.72)
    "exang": [0.0, 0.0],  # 0 for No
    "oldpeak": [0.35, 0.14],  # Normalize old peak values (e.g., 3.5 -> 0.35, 1.4 -> 0.14)
    "slope": [0.0, 2.0],  # 0 for Upsloping, 2 for Downsloping
    "ca": [0.0, 0.0],  # 0 for No major vessels
    "thal": [2.0, 2.0]  # 2 for Reversible defect
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure all columns are of type float64 to match the model schema
df = df.astype(float)

# Step 3: Make Predictions
try:
    predictions = loaded_model.predict(df)
    prediction_results = []
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        result = 'Heart Disease' if prediction == 1 else 'No Disease'
        prediction_results.append(result)
        print(f"Sample {i + 1}: {result}")
except Exception as e:
    print(f"Error during prediction: {e}")

# Save Predictions to a .txt file
output_filename = "client.txt"
with open(output_filename, 'w') as f:
    f.write("Prediction Results:\n")
    for i, result in enumerate(prediction_results):
        f.write(f"Sample {i + 1}: {result}\n")

print(f"Predictions saved to {output_filename}")

# Step 4: Query Experiment Details
experiment_name = "Heart Disease Prediction1"  
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
    print(f"\nCurrent experiment ID: {experiment_id}")

    # Start an MLflow run and log the predictions file as an artifact
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_artifact(output_filename)  # Log the .txt file as an artifact
        print(f"Prediction results logged as an artifact in experiment '{experiment_name}'")
        
    # Query and display runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    print("\nRuns in the experiment:")
    print(runs[['run_id', 'status', 'experiment_id']])
else:
    print(f"Experiment '{experiment_name}' not found.")
