import json

import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.joblib')
    encoder = joblib.load('encoder.joblib')
    symptoms = request.get_json()['symptoms']

    # if the length of array is smaller than 17, add -1 for unknown value
    # otherwise I get the error: ValueError: X has 1 features, but OrdinalEncoder is expecting 17 features as input.
    # as there are 17 columns of symptoms

    while len(symptoms) != 17:
        symptoms.append(-1)

    symptoms_encoded = encoder.transform([symptoms])
    disease = clf.predict(symptoms_encoded)[0]
    return jsonify({'disease': disease})


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
        with open('symptoms.json', 'r') as f:
            symptoms_data = json.load(f)
        return jsonify(symptoms_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
