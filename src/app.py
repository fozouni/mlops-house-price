# src/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import List, Optional

# ایجاد نمونه‌ی برنامه
app = FastAPI(
    title="House Price Prediction API",
    description="An API for predicting house prices based on features like square footage, bedrooms, etc.",
    version="1.0.0"
)

# تعیین مدل داده‌ی ورودی
class HouseFeatures(BaseModel):
    """
    مدل داده‌ی ورودی برای پیش‌بینی قیمت مسکن
    """
    sqft: float          # متراژ (فوت مربع)
    bedrooms: int        # تعداد اتاق‌ها
    year_built: int      # سال ساخت
    distance_to_center: float  # فاصله از مرکز شهر (کیلومتر)

    class Config:
        json_schema_extra = {
            "example": {
                "sqft": 1500,
                "bedrooms": 3,
                "year_built": 2005,
                "distance_to_center": 5.5
            }
        }

# تعیین مدل داده‌ی ورودی برای پیش‌بینی چندگانه
class HousesFeatures(BaseModel):
    houses: List[HouseFeatures]

# بارگذاری مدل
def load_model():
    """
    مدل آموزش‌دیده را از فایل بارگذاری می‌کند.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_v1.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# بارگذاری مدل هنگام راه‌اندازی سرور
try:
    model = load_model()
    print("[SUCCESS] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# تعریف مسیرهای ای‌پی‌آی

@app.get("/")
def read_root():
    """
    مسیر اصلی برای بررسی سلامتی ای‌پی‌آی
    """
    return {
        "message": "House Price Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    """
    بررسی وضعیت سلامت ای‌پی‌آی و مدل
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": "models/model_v1.pkl"
    }

@app.post("/predict")
def predict_price(house: HouseFeatures):
    """
    پیش‌بینی قیمت برای یک خانه‌ی واحد
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # تبدیل داده‌های ورودی به دیتافریم
        input_data = pd.DataFrame([{
            'sqft': house.sqft,
            'bedrooms': house.bedrooms,
            'year_built': house.year_built,
            'distance_to_center': house.distance_to_center
        }])

        # انجام پیش‌بینی
        prediction = model.predict(input_data)

        return {
            "predicted_price": round(prediction[0], 2),
            "features": house.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
def predict_prices_batch(houses: HousesFeatures):
    """
    پیش‌بینی قیمت برای چند خانه به صورت دسته‌ای
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        # تبدیل داده‌های ورودی به دیتافریم
        input_data = pd.DataFrame([{
            'sqft': h.sqft,
            'bedrooms': h.bedrooms,
            'year_built': h.year_built,
            'distance_to_center': h.distance_to_center
        } for h in houses.houses])

        # انجام پیش‌بینی
        predictions = model.predict(input_data)

        # ساخت لیست نتایج
        results = []
        for i, house in enumerate(houses.houses):
            results.append({
                "house": i + 1,
                "predicted_price": round(predictions[i], 2),
                "features": house.dict()
            })

        return {
            "total": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/info")
def model_info():
    """
    نمایش اطلاعات مدل
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return {
        "model_type": "RandomForestRegressor",
        "model_loaded": True,
        "features": ["sqft", "bedrooms", "year_built", "distance_to_center"],
        "version": "1.0.0"
    }