import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Preprocessing function
def preprocess_input(input_df):
    try:
        # Example: Assume that the columns are similar to your training data
        columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        
        # Ensure input has the same columns as the training data
        input_df = input_df[columns]
        
        # Scale the data using the StandardScaler
        scaler = StandardScaler()
        input_df_scaled = scaler.fit_transform(input_df)  # Fit and transform for new data
        
        return input_df_scaled
    except Exception as e:
        return f"Error in preprocessing input: {str(e)}"

# Prediction function
def predict(input_data):
    try:
        # Ensure input_data is 2D (reshape if it's a single row)
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make prediction using the trained model
        prediction = model.predict(input_data)
        
        # Map the prediction output (if needed, for binary classification)
        if prediction == 1:
            return "Survived"
        else:
            return "Not Survived"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Streamlit UI
st.title('Titanic Survival Prediction')

# Input fields for the user
Pclass = st.selectbox('Pclass', [1, 2, 3])
Age = st.number_input('Age', min_value=0, max_value=100, value=30)
SibSp = st.number_input('SibSp (Number of Siblings/Spouse)', min_value=0, max_value=10, value=0)
Parch = st.number_input('Parch (Number of Parents/Children)', min_value=0, max_value=10, value=0)
Fare = st.number_input('Fare', min_value=0, max_value=500, value=30)

sex = st.selectbox('Sex', ['male', 'female'])
if sex == 'female':
    Sex_female = 1
    Sex_male = 0
else:
    Sex_female = 0
    Sex_male = 1

embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
Embarked_C = 1 if embarked == 'C' else 0
Embarked_Q = 1 if embarked == 'Q' else 0
Embarked_S = 1 if embarked == 'S' else 0

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Sex_female': [Sex_female],
    'Sex_male': [Sex_male],
    'Embarked_C': [Embarked_C],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
})

# Predict button
if st.button('Predict'):
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Check if preprocessing was successful
    if isinstance(processed_input, str):
        st.error(processed_input)  # Show error message
    else:
        # Make the prediction
        prediction_result = predict(processed_input)
        
        # Show the result to the user
        st.write(f"The predicted outcome is: {prediction_result}")
