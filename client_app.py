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
        # Features + Ground-truth Label
        ([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1], 0),
        ([69, 0, 3, 140, 239, 0, 1, 151, 0, 1.8, 2, 2, 2], 1),
        ([37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2], 1),
        ([41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2], 0),
        ([71, 0, 1, 160, 302, 0, 1, 162, 0, 0.4, 2, 2, 2], 1),
        ([35, 1, 0, 120, 198, 0, 1, 130, 1, 1.6, 1, 0, 3], 0),
        ([52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3], 0),
        ([67, 0, 0, 106, 223, 0, 1, 142, 0, 0.3, 2, 2, 2], 1),
        ([60, 1, 2, 140, 185, 0, 0, 155, 0, 3, 1, 0, 2], 0),
        ([42, 0, 0, 102, 265, 0, 0, 122, 0, 0.6, 1, 0, 2], 1),
        ([51, 0, 0, 130, 305, 0, 1, 142, 1, 1.2, 1, 0, 3], 1),
        ([39, 1, 0, 118, 219, 0, 1, 140, 0, 1.2, 1, 0, 3], 0),
        ([68, 1, 0, 144, 193, 1, 1, 141, 0, 3.4, 1, 2, 3], 1),
    ]

    failed_cases = []
    predictions = []  # Store predictions for display
    latencies = []
    correct_predictions = 0

    for idx, (features, ground_truth) in enumerate(test_dataset):
        start_time = time.time()
        response = client.post('/predict', json={"features": features})
        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)

        if response.status_code != 200:
            failed_cases.append({
                "index": idx,
                "features": features,
                "error": "HTTP Error",
                "status_code": response.status_code
            })
            continue

        try:
            data = response.get_json()
            prediction = data.get("prediction")
            predictions.append({
                "index": idx,
                "features": features,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "status": "Correct" if prediction == ground_truth else "Incorrect"
            })
            if prediction == ground_truth:
                correct_predictions += 1
            else:
                failed_cases.append({
                    "index": idx,
                    "features": features,
                    "error": "Incorrect prediction",
                    "prediction": prediction,
                    "ground_truth": ground_truth
                })
        except Exception as e:
            failed_cases.append({"index": idx, "features": features, "error": str(e)})

    # Accuracy calculation
    accuracy = correct_predictions / len(test_dataset)

    # Log statistics
    average_latency = sum(latencies) / len(latencies)
    print(f"\n=== Test Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Latency: {average_latency:.4f} seconds")
    print(f"Failed Cases: {len(failed_cases)}\n")

    # Display predictions
    print(f"\n=== Predictions ===")
    for pred in predictions:
        print(f"Case {pred['index']}: Features: {pred['features']}, Prediction: {pred['prediction']}, "
              f"Ground Truth: {pred['ground_truth']}, Status: {pred['status']}")

    # Assertions for CI/CD pipeline
    assert accuracy >= 0.9, f"Accuracy is below 90%. Achieved: {accuracy * 100:.2f}%"
    assert len(failed_cases) == 0, f"Test failed for {len(failed_cases)} cases. See logs for details."
    assert average_latency < 1.0, "Average latency exceeds 1 second. Needs optimization."
