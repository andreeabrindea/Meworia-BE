import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

# Load the trained model and encoder
clf = joblib.load('model.joblib')
encoder = joblib.load('encoder.joblib')

# Define the symptoms for prediction
symptoms = ['skin_rash', 'nodal_skin_eruptions']

# Create a DataFrame with the symptoms
symptoms_data = pd.DataFrame([symptoms], columns=X.columns)

# Encode the symptoms using the trained encoder
symptoms_encoded = encoder.transform(symptoms_data)

# Make a prognosis prediction
prognosis_prediction = clf.predict(symptoms_encoded)[0]
print("Prognosis:", prognosis_prediction)