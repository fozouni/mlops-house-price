# Open the file named src/train-v2.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Function to generate sample data
def create_sample_data():
    """
    This function generates a sample dataset for house price prediction.
    Features include square footage, number of bedrooms, year built, and distance to city center.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    sqft = np.random.randint(500, 3000, n_samples)          # Square footage
    bedrooms = np.random.randint(1, 6, n_samples)           # Number of bedrooms
    year_built = np.random.randint(1960, 2023, n_samples)   # Year built
    distance_to_center = np.random.uniform(0.5, 20, n_samples)  # Distance to center (km)
    
    # Calculate price as a combination of features
    price = (sqft * 150) + (bedrooms * 5000) + ((2023 - year_built) * 1000) - (distance_to_center * 10000)
    price = price + np.random.normal(0, 5000, n_samples)    # Add random noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'year_built': year_built,
        'distance_to_center': distance_to_center,
        'price': price
    })
    
    return data

# Function to save the trained model
def save_model(model, filepath='models/model_v1_en.pkl'):
    """
    Saves the trained model to the specified path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[SUCCESS] Model saved to '{filepath}'")

# Main function
def main():
    print("=" * 50)
    print("STARTING HOUSE PRICE PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n[STEP 1] Generating sample data...")
    df = create_sample_data()
    print(f"   Total records: {len(df)}")
    print(f"   Features: {list(df.columns)}")
    
    # Step 2: Separate features and target
    print("\n[STEP 2] Separating features and target variable...")
    X = df.drop('price', axis=1)
    y = df['price']
    print(f"   Number of features: {X.shape[1]}")
    
    # Step 3: Split into training and test sets
    print("\n[STEP 3] Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Step 4: Train the model
    print("\n[STEP 4] Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   Model training completed successfully.")
    
    # Step 5: Evaluate the model
    print("\n[STEP 5] Evaluating model performance...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   Mean Squared Error (MSE): {mse:,.0f}")
    print(f"   R-squared (R²): {r2:.3f}")
    
    # Step 6: Save the model
    print("\n[STEP 6] Saving the model...")
    save_model(model, 'models/model_v1_en.pkl')
    
    print("\n" + "=" * 50)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 50)

# Execute main function
if __name__ == "__main__":
    main()