import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load datasets
disease_data = pd.read_csv('data/disease_dataset.csv')
description_data = pd.read_csv('data/symptom_Description.csv')
precaution_data = pd.read_csv('data/symptom_precaution.csv')

print("Data loaded successfully!")


# Preview the first 5 rows of each dataset
print("Disease Dataset:")
print(disease_data.head())

print("\nSymptom Description Dataset:")
print(description_data.head())

print("\nSymptom Precaution Dataset:")
print(precaution_data.head())


from sklearn.preprocessing import LabelEncoder

# Fill missing symptoms with 'None'
disease_data = disease_data.fillna('None')

# Combine all symptom columns into one list
all_symptoms = set(disease_data.iloc[:, 1:].values.flatten())
all_symptoms.discard('None')

# Create a dictionary to assign a unique number to each symptom
symptom_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Encode symptoms using the dictionary
def encode_symptoms(row):
    return [symptom_dict.get(symptom, -1) for symptom in row[1:]]

# Apply the encoding
X = disease_data.apply(encode_symptoms, axis=1)
Y = disease_data['Disease']

print("Symptoms encoded successfully!")


# Convert the encoded symptoms into a DataFrame
X = pd.DataFrame(X.tolist())

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Data Split Successfully!")



# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

print("Model Trained Successfully!")



# Predict on the test set
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display a classification report
print("Classification Report:")
print(classification_report(Y_test, Y_pred))


#save the model

import pickle

# Save the trained model
with open('models/disease_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model Saved Successfully!")
