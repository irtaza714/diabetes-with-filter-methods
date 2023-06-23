from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = str (request.form.get('gender')),
            age = float(request.form.get('age')),
            hypertension = int (request.form.get('hypertension')),
            heart_disease = int (request.form.get('heart_disease')),
            smoking_history = str (request.form.get('smoking_history')),
            bmi = float(request.form.get('bmi')),
            HbA1c_level = float(request.form.get('HbA1c_level')),
            blood_glucose_level = int ((request.form.get('blood_glucose_level')))

    )
        pred_df = data.get_data_as_data_frame()
        print("Before Prediction", pred_df)

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        if results == 0:
            results = 'The person does not have diabetes'
        else:
            results = "The person has diabetes"

        print("After Prediction", results)
        return render_template('home.html', results=results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)