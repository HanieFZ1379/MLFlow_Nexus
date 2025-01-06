import pytest
import time
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_predict(client):
    # Test dataset with ground-truth labels
    test_dataset = [
    # Features (as a dictionary with field names) + Ground-truth Label
    ({"age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0, "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0}, 0),
    ({"age": 69.0, "sex": 0.0, "cp": 3.0, "trestbps": 140.0, "chol": 239.0, "fbs": 0.0, "restecg": 1.0, "thalach": 151.0, "exang": 0.0, "oldpeak": 1.8, "slope": 2.0, "ca": 2.0, "thal": 2.0}, 1),
    ({"age": 37.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0, "chol": 250.0, "fbs": 0.0, "restecg": 1.0, "thalach": 187.0, "exang": 0.0, "oldpeak": 3.5, "slope": 0.0, "ca": 0.0, "thal": 2.0}, 1),
    ({"age": 41.0, "sex": 0.0, "cp": 1.0, "trestbps": 130.0, "chol": 204.0, "fbs": 0.0, "restecg": 0.0, "thalach": 172.0, "exang": 0.0, "oldpeak": 1.4, "slope": 2.0, "ca": 0.0, "thal": 2.0}, 0),
    ({"age": 71.0, "sex": 0.0, "cp": 1.0, "trestbps": 160.0, "chol": 302.0, "fbs": 0.0, "restecg": 1.0, "thalach": 162.0, "exang": 0.0, "oldpeak": 0.4, "slope": 2.0, "ca": 2.0, "thal": 2.0}, 1),
    ({"age": 35.0, "sex": 1.0, "cp": 0.0, "trestbps": 120.0, "chol": 198.0, "fbs": 0.0, "restecg": 1.0, "thalach": 130.0, "exang": 1.0, "oldpeak": 1.6, "slope": 1.0, "ca": 0.0, "thal": 3.0}, 0),
    ({"age": 52.0, "sex": 1.0, "cp": 0.0, "trestbps": 125.0, "chol": 212.0, "fbs": 0.0, "restecg": 1.0, "thalach": 168.0, "exang": 0.0, "oldpeak": 1.0, "slope": 2.0, "ca": 2.0, "thal": 3.0}, 0),
    ({"age": 67.0, "sex": 0.0, "cp": 0.0, "trestbps": 106.0, "chol": 223.0, "fbs": 0.0, "restecg": 1.0, "thalach": 142.0, "exang": 0.0, "oldpeak": 0.3, "slope": 2.0, "ca": 2.0, "thal": 2.0}, 1),
    ({"age": 60.0, "sex": 1.0, "cp": 2.0, "trestbps": 140.0, "chol": 185.0, "fbs": 0.0, "restecg": 0.0, "thalach": 155.0, "exang": 0.0, "oldpeak": 3.0, "slope": 1.0, "ca": 0.0, "thal": 2.0}, 0),
    ({"age": 42.0, "sex": 0.0, "cp": 0.0, "trestbps": 102.0, "chol": 265.0, "fbs": 0.0, "restecg": 0.0, "thalach": 122.0, "exang": 0.0, "oldpeak": 0.6, "slope": 1.0, "ca": 0.0, "thal": 2.0}, 1),
    ({"age": 51.0, "sex": 0.0, "cp": 0.0, "trestbps": 130.0, "chol": 305.0, "fbs": 0.0, "restecg": 1.0, "thalach": 142.0, "exang": 1.0, "oldpeak": 1.2, "slope": 1.0, "ca": 0.0, "thal": 3.0}, 1),
    ({"age": 39.0, "sex": 1.0, "cp": 0.0, "trestbps": 118.0, "chol": 219.0, "fbs": 0.0, "restecg": 1.0, "thalach": 140.0, "exang": 0.0, "oldpeak": 1.2, "slope": 1.0, "ca": 0.0, "thal": 3.0}, 0),
    ({"age": 68.0, "sex": 1.0, "cp": 0.0, "trestbps": 144.0, "chol": 193.0, "fbs": 1.0, "restecg": 1.0, "thalach": 141.0, "exang": 0.0, "oldpeak": 3.4, "slope": 1.0, "ca": 2.0, "thal": 3.0}, 1),
]


  
    predictions = []  # Store predictions for display

    for idx, (features, ground_truth) in enumerate(test_dataset):
        start_time = time.time()
        response = client.post('/predict', json={"features": features})

        # Ensure the response is valid
        assert response.status_code == 200, f"Failed at index {idx}: {response.data}"

        # Parse the JSON response
        response_json = response.get_json()
        assert "prediction" in response_json, f"Prediction missing in response at index {idx}"

        prediction = response_json["prediction"]
        predictions.append(prediction)

        # Log prediction time
        elapsed_time = time.time() - start_time
        print(f"Test {idx + 1}: Prediction: {prediction}, Ground Truth: {ground_truth}, Time: {elapsed_time:.4f} seconds")


    # Log all predictions
    print("All Predictions:", predictions)
