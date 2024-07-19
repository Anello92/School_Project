# Machine Learning Model on AWS Cloud for School Management
# Web App

# Imports
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Create the app
app = Flask(__name__)

# Function to load the model or scaler
def load_file(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Load the model and scaler
model = load_file('final_model.pkl')
scaler = load_file('scaler_final.pkl')

# Route for the entry web page
@app.route('/')
def index():
    return render_template('index_v1.html')

# Route for the prediction function
@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Extract values sent via form
        english_exam = float(request.form.get('english_exam', 0))
        psychometric_exam = int(request.form.get('psychometric_exam', 0))
        iq_score = int(request.form.get('iq_score', 0))

        # Input validation
        if not (0 <= english_exam <= 10 and 0 <= iq_score <= 200 and 0 <= psychometric_exam <= 100):
            raise ValueError("Invalid input values")

        # Create an array with input data adjusting the shape
        input_data = np.array([english_exam, iq_score, psychometric_exam]).reshape(1, 3)

        # Dataframe with data and column names
        input_data_df = pd.DataFrame(input_data, columns=['english_exam_score', 'iq_score', 'psychometric_exam_score'])

        # Standardize input data
        standardized_input_data = scaler.transform(input_data_df)

        # Make prediction with the model using the standardized data (the same way the model was trained)
        pred = model.predict(standardized_input_data)
        
        result = 'The Student Can Be Enrolled in the Course' if pred[0] == 1 else 'The Student Cannot Be Enrolled in the Course'

    except Exception as e:
        result = f"Prediction error: {e}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run()
