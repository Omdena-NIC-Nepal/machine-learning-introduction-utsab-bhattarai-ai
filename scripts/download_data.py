import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def download_boston_dataset():
    """
    Download the Boston House Prices dataset from Kaggle
    """
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download the dataset
    print("Downloading Boston House Prices dataset...")
    api.dataset_download_files(
        'vikrishnan/boston-house-prices',
        path='data',
        unzip=True
    )
    
    print("Dataset downloaded successfully!")
    
    # Read and display the first few rows of the data
    df = pd.read_csv('data/boston_housing.csv')
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nDataset shape:", df.shape)
    print("\nDataset columns:", df.columns.tolist())

if __name__ == "__main__":
    download_boston_dataset() 