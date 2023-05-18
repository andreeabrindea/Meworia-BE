import json

import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('/Users/andreea/PycharmProjects/Meowria/model.joblib')
    encoder = joblib.load('/Users/andreea/PycharmProjects/Meowria/encoder.joblib')
    symptoms = request.get_json()['symptoms']

    while len(symptoms) < 17:
        symptoms.append('unknown')

    symptoms_encoded = encoder.transform([symptoms])
    disease = clf.predict(symptoms_encoded)[0]
    return jsonify({'disease': disease})



@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    with open('symptoms.json', 'r') as f:
        symptoms_data = json.load(f)
    return jsonify(symptoms_data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
