from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    # Load the dataset
    data = pd.read_csv('/Users/andreea/Desktop/Licenta/dataset.csv')
    data = data.drop(columns=["Disease"])

    # Filter out rows with NaN values and replace them with 'unknown'
    data = data.fillna("unknown")

    symptoms = data.values

    # Convert the array to a list of symptoms
    list_of_symptoms = list(symptoms.flatten())

    k = []
    for i in list_of_symptoms:
        j = i.replace(' ', '')
        j = j.replace('_', " ")
        k.append(j)

    # Print the list of unique symptoms
    a = list(set(k))
    a.remove('unknown')

    return jsonify({'symptoms': a})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
