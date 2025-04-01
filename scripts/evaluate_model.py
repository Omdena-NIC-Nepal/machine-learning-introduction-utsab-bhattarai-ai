import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from data_preprocessing import prepare_data

def load_model():
    """Load the trained model"""
    return joblib.load('models/linear_regression_model.joblib')

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    return y_pred

def plot_residuals(y_test, y_pred):
    """Plot residuals to check assumptions"""
    residuals = y_test - y_pred
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Predicted Values
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    
    # Residuals Distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Count')
    ax2.set_title('Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig('models/residuals_plot.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance = np.abs(model.coef_)
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()

def main():
    # Load the model and prepare test data
    model = load_model()
    _, X_test, _, y_test = prepare_data()
    
    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Create plots
    plot_residuals(y_test, y_pred)
    
    # Get feature names from the dataset
    import pandas as pd
    df = pd.read_csv('data/boston_housing.csv')
    feature_names = df.drop('MEDV', axis=1).columns
    plot_feature_importance(model, feature_names)

if __name__ == "__main__":
    main() 