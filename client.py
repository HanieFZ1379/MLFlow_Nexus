import mlflow
import pandas as pd

# Set the tracking URI for the MLflow server
mlflow.set_tracking_uri("http://37.152.191.193:8080")  # Replace with your MLflow server's URI

# Step 1: Load the Deployed Model
logged_model = "models:/heart_disease_classifier/2"  # Replace '2' with the correct version or stage
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Step 2: Send Test Data
# Example test data (replace with the actual input data)
data = {
    "age": [63, 69, 41, 51],
    "sex": [1, 0, 0, 0],  # Male=1, Female=0
    "cp": [3, 3, 1, 0],  # 3=Asymptomatic, 2=Non-anginal pain, 1=Atypical angina
    "trestbps": [145, 140, 130, 130],
    "chol": [233, 239, 204, 305],
    "fbs": [1, 0, 0, 0],  # Fasting blood sugar True=1, False=0
    "restecg": [0, 1, 0, 1],  # Normal=0, Abnormality=1
    "thalach": [150, 151, 172, 142],
    "exang": [0, 0, 0, 1],  # No=0, Yes=1
    "oldpeak": [2.3, 1.8, 1.4, 1.2],
    "slope": [0, 2, 0, 1],  # Upsloping=0, Downsloping=2
    "ca": [0, 2, 0, 0],  # Number of major vessels
    "thal": [1, 2, 2, 3]  # Fixed defect=1, Reversible defect=2
}


# Create DataFrame
df = pd.DataFrame(data)

# Ensure all columns are of type float64 to match the model schema
df = df.astype(float)

# Step 3: Make Predictions
# Pass the DataFrame to the model for prediction
try:
    predictions = loaded_model.predict(df)
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"Sample {i + 1}: {'Heart Disease' if prediction == 1 else 'No Disease'}")
except Exception as e:
    print(f"Error during prediction: {e}")

# Step 4: Query Experiment Details
experiment_name = "Heart Disease Prediction1"  # Replace with the correct experiment name
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
    print(f"\nCurrent experiment ID: {experiment_id}")

    # Query and display runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    print("\nRuns in the experiment:")
    print(runs[['run_id', 'status', 'experiment_id']])
else:
    print(f"Experiment '{experiment_name}' not found.")
