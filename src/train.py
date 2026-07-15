# Open the file named src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# ۱. تابع تولید داده‌های نمونه
def create_sample_data():
    """
    این تابع یک دیتاست نمونه برای پیش‌بینی قیمت مسکن تولید می‌کند.
    ویژگی‌ها شامل متراژ، تعداد اتاق‌ها، سال ساخت و فاصله از مرکز شهر است.
    """
    np.random.seed(42)
    n_samples = 1000

    # تولید ویژگی‌ها
    sqft = np.random.randint(500, 3000, n_samples)          # متراژ (فوت مربع)
    bedrooms = np.random.randint(1, 6, n_samples)           # تعداد اتاق‌ها
    year_built = np.random.randint(1960, 2023, n_samples)   # سال ساخت
    distance_to_center = np.random.uniform(0.5, 20, n_samples)  # فاصله از مرکز (کیلومتر)

    # محاسبه‌ی قیمت به عنوان ترکیبی از ویژگی‌ها
    price = (sqft * 150) + (bedrooms * 5000) + ((2023 - year_built) * 1000) - (distance_to_center * 10000)
    price = price + np.random.normal(0, 5000, n_samples)    # افزودن نویز تصادفی

    # ساخت دیتافریم
    data = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'year_built': year_built,
        'distance_to_center': distance_to_center,
        'price': price
    })

    return data

# ۲. تابع ذخیره‌سازی مدل
def save_model(model, filepath='models/model_v1.pkl'):
    """
    مدل آموزش‌دیده را در مسیر مشخص شده ذخیره می‌کند.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ مدل در مسیر '{filepath}' ذخیره شد.")

# ۳. تابع اصلی
def main():
    print("=" * 50)
    print("🚀 شروع فرایند آموزش مدل پیش‌بینی قیمت مسکن")
    print("=" * 50)

    # مرحله ۱: بارگذاری داده
    print("\n📊 مرحله ۱: تولید داده‌های نمونه...")
    df = create_sample_data()
    print(f"   تعداد رکوردها: {len(df)}")
    print(f"   ویژگی‌ها: {list(df.columns)}")

    # مرحله ۲: جداسازی ویژگی‌ها و هدف
    print("\n🔧 مرحله ۲: جداسازی ویژگی‌ها و متغیر هدف...")
    X = df.drop('price', axis=1)
    y = df['price']
    print(f"   تعداد ویژگی‌ها: {X.shape[1]}")

    # مرحله ۳: تقسیم به مجموعه‌های آموزش و تست
    print("\n✂️  مرحله ۳: تقسیم داده به مجموعه‌های آموزش و تست...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   تعداد نمونه‌های آموزش: {len(X_train)}")
    print(f"   تعداد نمونه‌های تست: {len(X_test)}")

    # مرحله ۴: آموزش مدل
    print("\n🤖 مرحله ۴: آموزش مدل جنگل تصادفی...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   آموزش مدل با موفقیت انجام شد.")

    # مرحله ۵: ارزیابی مدل
    print("\n📈 مرحله ۵: ارزیابی عملکرد مدل...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   خطای میانگین مربعات (MSE): {mse:,.0f}")
    print(f"   ضریب تعیین (R²): {r2:.3f}")

    # مرحله ۶: ذخیره‌سازی مدل
    print("\n💾 مرحله ۶: ذخیره‌سازی مدل...")
    save_model(model, 'models/model_v1.pkl')

    print("\n" + "=" * 50)
    print("✅ فرایند آموزش با موفقیت به پایان رسید.")
    print("=" * 50)

# اجرای تابع اصلی
if __name__ == "__main__":
    main()