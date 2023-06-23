import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            data_scaled_df = pd.DataFrame(data_scaled)
            for i in range(len(data_scaled_df.columns)):  
                data_scaled_df = data_scaled_df.rename(columns={data_scaled_df.columns[i]: f'c{i+1}'})
            
            qc_columns = ['c9']

            data_scaled_df_filter = data_scaled_df.drop(qc_columns, axis =1)
            
            data_scaled_df_filter_np = np.array(data_scaled_df_filter)
            preds = model.predict(data_scaled_df_filter_np)

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, age: float, hypertension: int, heart_disease: int,
       smoking_history: str, bmi: float, HbA1c_level: float,
       blood_glucose_level: int):
        
        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.smoking_history = smoking_history
        self.bmi = bmi
        self.HbA1c_level = HbA1c_level
        self.blood_glucose_level = blood_glucose_level
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "hypertension" : [self.hypertension], 
                "heart_disease" : [self.heart_disease],
                "smoking_history" : [self.smoking_history],
                "bmi" : [self.bmi], 
                "HbA1c_level" : [self.HbA1c_level],
                "blood_glucose_level" : [self.blood_glucose_level]
               }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)