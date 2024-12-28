import pytest
from app import app


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_predict(client):
    Test_dataset = [
    #  sample 1:
    # 63, Male, Asymptomatic, 145, 233, True, Normal, 150, No, 2.3, Upsloping, 0, Fixed defect
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
    # sample 2
    # 69, Female, Asymptomatic, 140, 239, False, Abnormality, 151, No, 1.8, Downsloping, 2, Reversible defect
    [69, 0, 3, 140, 239, 0, 1, 151, 0, 1.8, 2, 2, 2],
    # sample 3
    # 37, Male, Non-anginal pain, 130, 250, False, Abnormality, 187, No, 3.5, Upsloping, 0, Reversible defect
    [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],
    # sample 4
    # 41, Female, Atypical angina, 130, 204, False, Normal, 172, No, 1.4, Downsloping, 0, Reversible defect
    [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2],
    # sample 5
    # 71, Female, Atypical angina, 160, 302, False, Abnormality, 162, No, 0.4, Downsloping, 2, Reversible defect
    [71, 0, 1, 160, 302, 0, 1, 162, 0, 0.4, 2, 2, 2],
    # sample 6
    # 35, Male, Typical angina, 120, 198, False, Abnormality, 130, Yes, 1.6, Flat, 0, Not described
    [35, 1, 0, 120, 198, 0, 1, 130, 1, 1.6, 1, 0, 3],
    # sample 7
    # 52, Male, Typical angina, 125, 212, False, Abnormality, 168, No, 1, Downsloping, 2, Not described
    [52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3],
    # sample 8
    # 67, Female, Typical angina, 106, 223, False, Abnormality, 142, No, 0.3, Downsloping, 2, Reversible defect
    [67, 0, 0, 106, 223, 0, 1, 142, 0, 0.3, 2, 2, 2],
    # sample 9
    # 60, Male, Non-anginal pain, 140, 185, False, Normal, 155, No, 3, Flat, 0, Reversible defect
    [60, 1, 2, 140, 185, 0, 0, 155, 0, 3, 1, 0, 2],
    # sample 10
    # 42, Female, Typical angina, 102, 265, False, Normal, 122, No, 0.6, Flat, 0, Reversible defect
    [42, 0, 0, 102, 265, 0, 0, 122, 0, 0.6, 1, 0, 2],
    # sample 11
    # 51, Female, Typical angina, 130, 305, False, Abnormality, 142, Yes, 1.2, Flat, 0, Not described
    [51, 0, 0, 130, 305, 0, 1, 142, 1, 1.2, 1, 0, 3],
    # sample 12
    # 39, Male, Typical angina, 118, 219, False, Abnormality, 140, No, 1.2, Flat, 0, Not described
    [39, 1, 0, 118, 219, 0, 1, 140, 0, 1.2, 1, 0, 3],
    # sample 13
    # 68, Male, Typical angina, 144, 193, True, Abnormality, 141, No, 3.4, Flat, 2, Not described
    [68, 1, 0, 144, 193, 1, 1, 141, 0, 3.4, 1, 2, 3]]
    for feature in Test_dataset:
        response = client.post('/predict', json={"features": feature})
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)