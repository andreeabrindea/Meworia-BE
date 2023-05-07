from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
data = pd.read_csv('/Users/andreea/Desktop/Licenta/dataset.csv')

# Remove unwanted spaces (as some cells have a space in front)
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# fill the empty values with "unknown"
data = data.fillna("unknown")

# Split the dataset into features (symptoms) and target (disease)
X = data.drop(columns=["Disease"])
y = data["Disease"]

# Encode categorical variables using ordinal encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = encoder.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a random forest classifier, as it gave me the best accuracy score (0.94)
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test)
y_pr = clf.predict(X_encoded)

accuracy = clf.score(X_test, y_test)
precision = precision_score(y_test, y_pred, zero_division=1, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=1, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=1, average='weighted')
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")


# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
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
