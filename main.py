from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Trained model load karo (yeh file same folder mein honi chahiye)
model = joblib.load("crop_model.pkl")

# Input format define karo - Tejas isi format mein data bhejega
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def home():
    return {"message": "KrishiMitra Crop Recommendation API is running"}

@app.post("/predict-crop")
def predict_crop(data: CropInput):
    input_data = np.array([[data.N, data.P, data.K, data.temperature,
                             data.humidity, data.ph, data.rainfall]])
    prediction = model.predict(input_data)
    return {"recommended_crop": prediction[0]}
