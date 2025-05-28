"""
Model I/O utility functions for the Apple Scab detection system.
Contains functions to save/load model parameters in CSV format.
"""
import os
import pandas as pd
import numpy as np
from src.svm import LinearSVM

def save_model_to_csv(svm_model, filepath):
    """
    Save trained SVM model parameters to CSV file.
    
    Args:
        svm_model (LinearSVM): Trained SVM model
        filepath (str): Path to save the model CSV
        
    Returns:
        str: Path to the saved model
    """
    # Check if model is trained
    if svm_model.w is None or svm_model.feature_means is None or svm_model.feature_stds is None:
        raise ValueError("Model is not trained, cannot save to CSV")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Create a dictionary to hold model parameters
    model_data = {
        'parameter_type': [],
        'feature_index': [],
        'value': []
    }
    
    # Save weights
    for i, weight in enumerate(svm_model.w):
        model_data['parameter_type'].append('weight')
        model_data['feature_index'].append(i)
        model_data['value'].append(weight)
    
    # Save bias
    model_data['parameter_type'].append('bias')
    model_data['feature_index'].append(0)
    model_data['value'].append(svm_model.b)
    
    # Save feature means
    for i, mean in enumerate(svm_model.feature_means):
        model_data['parameter_type'].append('feature_mean')
        model_data['feature_index'].append(i)
        model_data['value'].append(mean)
    
    # Save feature standard deviations
    for i, std in enumerate(svm_model.feature_stds):
        model_data['parameter_type'].append('feature_std')
        model_data['feature_index'].append(i)
        model_data['value'].append(std)
    
    # Convert to DataFrame and save to CSV
    model_df = pd.DataFrame(model_data)
    model_df.to_csv(filepath, index=False)
    
    return filepath

def load_model_from_csv(filepath):
    """
    Load SVM model from CSV file.
    
    Args:
        filepath (str): Path to the model CSV file
        
    Returns:
        LinearSVM: Loaded SVM model
    """
    # Load the model parameters
    model_df = pd.read_csv(filepath)
    
    # Create a new SVM model
    model = LinearSVM()
    
    # Extract weights
    weight_rows = model_df[model_df['parameter_type'] == 'weight']
    max_feature_index = weight_rows['feature_index'].max()
    model.w = np.zeros(max_feature_index + 1)
    
    for _, row in weight_rows.iterrows():
        feature_idx = int(row['feature_index'])
        model.w[feature_idx] = row['value']
    
    # Extract bias
    bias_row = model_df[model_df['parameter_type'] == 'bias'].iloc[0]
    model.b = bias_row['value']
    
    # Extract feature means
    mean_rows = model_df[model_df['parameter_type'] == 'feature_mean']
    model.feature_means = np.zeros(max_feature_index + 1)
    
    for _, row in mean_rows.iterrows():
        feature_idx = int(row['feature_index'])
        model.feature_means[feature_idx] = row['value']
    
    # Extract feature standard deviations
    std_rows = model_df[model_df['parameter_type'] == 'feature_std']
    model.feature_stds = np.zeros(max_feature_index + 1)
    
    for _, row in std_rows.iterrows():
        feature_idx = int(row['feature_index'])
        model.feature_stds[feature_idx] = row['value']
    
    return model
