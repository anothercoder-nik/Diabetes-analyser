from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, use ["http://localhost:3000"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


model = joblib.load('diabetes_model.joblib')

@app.post("/predict")
def predict(features: dict):
    try:
    
        input_features = np.array([[
            features['Pregnancies'],
            features['Glucose'],
            features['BloodPressure'],
            features['SkinThickness'],
            features['Insulin'],
            features['BMI'],
            features['DiabetesPedigreeFunction'],
            features['Age']
        ]])

       
        prediction = model.predict(input_features)
        return {"diabetes": bool(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
