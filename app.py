from flask import Flask, request, jsonify
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
logged_model = "models:/heart_disease_classifier/1"  
loaded_model = mlflow.pyfunc.load_model(logged_model)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data['features']
    prediction = int(loaded_model.predict([input_data])[0])
    return jsonify({'prediction': prediction})


if __name__ == 'main':
    app.run(host='0.0.0.0', port=5001)