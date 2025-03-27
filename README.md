# Disease Prediction Using Symptoms

This project is an AI-powered **Disease Prediction System** that predicts possible diseases based on user-provided symptoms using machine learning. It also provides disease descriptions and suggests precautions.

## 🚀 **Features**
- Predicts disease based on selected symptoms.
- Provides a confidence score for predictions.
- Displays a description of the predicted disease.
- Suggests relevant precautions.

---

## 🛠 **Project Structure**
```bash
Disease_Prediction_Project/
├── app.py                  # Streamlit app for UI
├── disease_prediction.py   # Machine learning model and prediction functions
├── data/                   # Contains the dataset CSV files
│   ├── disease_dataset.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
├── models/                 # Trained ML model
│   ├── disease_prediction_model.pkl
├── requirements.txt        # Required libraries
```

---

## 📥 **Installation**
1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/disease-prediction.git
    cd disease-prediction
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## 📊 **Datasets Used**
- **disease_dataset.csv**: Contains symptoms and their corresponding diseases.
- **symptom_Description.csv**: Provides descriptions for diseases.
- **symptom_precaution.csv**: Suggests precautions for diseases.

---

## 🤖 **Model Training**
- A **Random Forest Classifier** is used for disease prediction.
- The model is trained on the symptom dataset using **sklearn**.
- The accuracy and confidence scores are displayed during predictions.

---

## 🧑‍💻 **How to Use**
1. Launch the app using the command: `streamlit run app.py`
2. Select your symptoms from the dropdown.
3. Click on **Predict Disease**.
4. View the predicted disease, confidence score, description, and precautions.

---

## 🛡 **Future Enhancements**
- Integrate severity-based symptom analysis for better predictions.
- Implement additional disease datasets.
- Provide suggestions for nearby hospitals or doctors.

---

## 📝 **Contributing**
Feel free to contribute by submitting pull requests or reporting bugs.

---



**Enjoy predicting diseases responsibly!** 😊

