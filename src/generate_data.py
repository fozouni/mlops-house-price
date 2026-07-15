# src/generate_data.py

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """
    Generate sample dataset for house price prediction.
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

def main():
    print("=" * 50)
    print("GENERATING SAMPLE DATA FOR HOUSE PRICE PREDICTION")
    print("=" * 50)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Generate and save data
    print("\n[STEP 1] Generating sample data...")
    df = create_sample_data()

    # Save to CSV
    filepath = 'data/housing_data.csv'
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Data saved to '{filepath}'")
    print(f"   Total records: {len(df)}")
    print(f"   File size: {os.path.getsize(filepath):,} bytes")

if __name__ == "__main__":
    main()