import streamlit as st
import pandas as pd
import pickle

# Load additional data
description_data = pd.read_csv('data/symptom_Description.csv')
precaution_data = pd.read_csv('data/symptom_precaution.csv')

# Function to get disease description
def get_disease_description(disease_name):
    description_row = description_data[description_data['Disease'] == disease_name]
    if not description_row.empty:
        return description_row['Description'].values[0]
    return "No description available."

# Function to get disease precautions
def get_precautions(disease_name):
    precaution_row = precaution_data[precaution_data['Disease'] == disease_name]
    if not precaution_row.empty:
        return list(precaution_row.iloc[0, 1:].dropna().values)
    return ["No precautions available."]

# Load the model
with open('models/disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the symptom-to-index mapping
symptom_data = pd.read_csv('data/disease_dataset.csv')
all_symptoms = list(symptom_data.iloc[:, 1:].values.flatten())
all_symptoms = sorted(set(symptom for symptom in all_symptoms if str(symptom) != 'nan'))

# Title
st.title("Disease Prediction Using Symptoms")

# Select Symptoms
st.write("Select symptoms you are experiencing:")
selected_symptoms = st.multiselect("Symptoms", all_symptoms)

# Predict Button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        # Encode selected symptoms
        symptom_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
        input_data = [symptom_dict.get(symptom, -1) for symptom in selected_symptoms]
        input_data += [-1] * (17 - len(input_data))

        # Predict Disease
        prediction = model.predict([input_data])[0]
        confidence_scores = model.predict_proba([input_data])[0]

        # Get the top 3 predictions with confidence scores
        top_3_indices = confidence_scores.argsort()[-3:][::-1]
        top_3_diseases = [(model.classes_[i], confidence_scores[i] * 100) for i in top_3_indices]

        st.success("Top Predictions:")
        for disease, confidence in top_3_diseases:
            st.write(f"**{disease}** with **{confidence:.2f}%** confidence")

        # Display details for the most likely disease
        primary_prediction = top_3_diseases[0][0]
        description = get_disease_description(primary_prediction)
        precautions = get_precautions(primary_prediction)

        st.info(f"**About the Disease:**\n{description}")
        st.warning("**Precautions to Take:**")
        for i, precaution in enumerate(precautions, 1):
            st.write(f"{i}. {precaution}")

