import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    """Load the Boston Housing dataset"""
    return pd.read_csv('data/boston_housing.csv')

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values:\n", missing_values[missing_values > 0])
    
    # If there are missing values, handle them
    if missing_values.any():
        # For numerical columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df

def handle_outliers(df, columns, method='zscore', threshold=3):
    """Handle outliers in the dataset"""
    df_clean = df.copy()
    
    for column in columns:
        if method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def prepare_data():
    """Main function to prepare the data"""
    # Load data
    df = load_data()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = handle_outliers(df, numeric_columns)
    
    # Split features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("Data preparation completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}") 