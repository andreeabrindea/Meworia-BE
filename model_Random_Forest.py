import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('./archive/dataset.csv')

# Remove unwanted spaces (as some cells have a space in front)
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# fill the empty values with "unknown"
data = data.fillna("unknown")

# Split the dataset into features (symptoms) and target (disease)
X = data.drop(columns=["Disease"])
y = data["Disease"]

# Encode categorical variables using ordinal encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = encoder.fit_transform(X.values)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a random forest classifier, as it gave me the best accuracy score (0.94)
model1 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
model1.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model1.predict(X_test)
y_pr = model1.predict(X_encoded)

accuracy = model1.score(X_test, y_test)
precision = precision_score(y_test, y_pred, zero_division=1, average='weighted')
recall = recall_score(y_test, y_pred, zero_division=1, average='weighted')
f1 = f1_score(y_test, y_pred, zero_division=1, average='weighted')
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

# Save the trained model and encoder to files
joblib.dump(model1, 'model_random_forest.joblib')
joblib.dump(encoder, 'encoder_random_forest.joblib')
