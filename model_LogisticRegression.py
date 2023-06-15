import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./archive/dataset.csv')

# Remove unwanted spaces (as some cells have a space in front)
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Fill the empty values with "unknown"
data = data.fillna("unknown")

# Split the dataset into features (symptoms) and target (disease)
X = data.drop(columns=["Disease"])
y = data["Disease"]

# Encode categorical variables using ordinal encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = encoder.fit_transform(X.values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train the logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
precision = precision_score(y_test, y_pred, zero_division=1, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=1, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=1, average='weighted')
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

# Save the trained model and encoder to files
joblib.dump(model, 'model_logistic_regression.joblib')
joblib.dump(encoder, 'encoder_logistic_regression.joblib')
