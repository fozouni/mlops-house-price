# src/train_with_mlflow.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# Function to generate sample data
def create_sample_data():
    """
    This function generates a sample dataset for house price prediction.
    """
    np.random.seed(42)
    n_samples = 1000

    sqft = np.random.randint(500, 3000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    year_built = np.random.randint(1960, 2023, n_samples)
    distance_to_center = np.random.uniform(0.5, 20, n_samples)

    price = (sqft * 150) + (bedrooms * 5000) + ((2023 - year_built) * 1000) - (distance_to_center * 10000)
    price = price + np.random.normal(0, 5000, n_samples)

    data = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'year_built': year_built,
        'distance_to_center': distance_to_center,
        'price': price
    })

    return data

# Main function with MLflow tracking
def main():
    print("=" * 50)
    print("STARTING HOUSE PRICE PREDICTION WITH MLFLOW TRACKING")
    print("=" * 50)

    # Set MLflow experiment name
    mlflow.set_experiment("House Price Prediction - Random Forest")

    # Start an MLflow run
    with mlflow.start_run():
        # Log experiment parameters
        # n_estimators = 100
        n_estimators = 200
        # max_depth = 10
        max_depth = 15
        test_size = 0.2
        random_state = 42

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        print("\n[INFO] Experiment parameters logged to MLflow")

        # Step 1: Generate and prepare data
        print("\n[STEP 1] Generating sample data...")
        df = create_sample_data()
        X = df.drop('price', axis=1)
        y = df['price']

        # Step 2: Split data
        print("\n[STEP 2] Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        # Step 3: Train model
        print("\n[STEP 3] Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        print("   Model training completed successfully.")

        # Step 4: Evaluate model
        print("\n[STEP 4] Evaluating model performance...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(f"   Mean Squared Error (MSE): {mse:,.0f}")
        print(f"   R-squared (R²): {r2:.3f}")
        print("   Metrics logged to MLflow")

        # Step 5: Save model with MLflow
        print("\n[STEP 5] Saving model with MLflow...")
        mlflow.sklearn.log_model(model, name="random_forest_model")
        print("   Model logged to MLflow")

        # Display run information
        run_id = mlflow.active_run().info.run_id
        print(f"\n[SUCCESS] MLflow Run ID: {run_id}")
        print("[INFO] To view results, run: mlflow ui")

    print("\n" + "=" * 50)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 50)

if __name__ == "__main__":
    main()