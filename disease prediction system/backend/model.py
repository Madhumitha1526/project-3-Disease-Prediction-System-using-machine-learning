import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\madhu\OneDrive\Documents\NEXUS prj 3\disease prediction system\backend\data\health_data.csv')

# Preprocess the data
X = data.drop('disease', axis=1)
y = data['disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')

# Evaluate the model
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")

# Load the model for prediction
def predict_disease(features):
    model = joblib.load('model.pkl')
    prediction = model.predict([features])
    return 'Disease Positive' if prediction[0] == 1 else 'Disease Negative'
