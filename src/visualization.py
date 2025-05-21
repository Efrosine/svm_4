"""
Visualization module for Apple Scab detection system.
Contains helper functions for visualizing data and results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import (FEATURES_DIR, SVM_DIR, EVAL_DIR, COMPARISON_DIR, 
                   get_timestamp, get_timestamped_filename)

def plot_feature_distribution(features_df):
    """
    Plot distribution of features by class.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features
        
    Returns:
        str: Path to saved plot
    """
    # Get unique classes
    unique_classes = features_df['label'].unique()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different classes
    colors = ['red', 'green', 'blue', 'orange']
    
    # Plot mean saturation distribution
    for i, class_name in enumerate(unique_classes):
        class_data = features_df[features_df['label'] == class_name]
        axes[0].hist(class_data['mean_saturation'], alpha=0.5, 
                    bins=20, label=class_name, color=colors[i % len(colors)])
    
    axes[0].set_title('Mean Saturation Distribution')
    axes[0].set_xlabel('Mean Saturation')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Plot black spot ratio distribution
    for i, class_name in enumerate(unique_classes):
        class_data = features_df[features_df['label'] == class_name]
        axes[1].hist(class_data['black_spot_ratio'], alpha=0.5, 
                    bins=20, label=class_name, color=colors[i % len(colors)])
    
    axes[1].set_title('Black Spot Ratio Distribution')
    axes[1].set_xlabel('Black Spot Ratio')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(FEATURES_DIR, exist_ok=True)
    filename = get_timestamped_filename("feature_distribution", "png")
    output_path = os.path.join(FEATURES_DIR, filename)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path

def visualize_feature_importance(svm_model, feature_names):
    """
    Visualize feature importance based on SVM weights.
    
    Args:
        svm_model: Trained SVM model
        feature_names (list): Names of features
        
    Returns:
        str: Path to saved plot
    """
    # Check if model is trained
    if svm_model.w is None:
        print("Model not trained yet. Cannot visualize feature importance.")
        return None
    
    # Get absolute weight values
    weights = np.abs(svm_model.w)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, weights)
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Absolute Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(SVM_DIR, exist_ok=True)
    filename = get_timestamped_filename("feature_importance", "png")
    output_path = os.path.join(SVM_DIR, filename)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path

def visualize_prediction_examples(X_test, y_test, y_pred, filenames=None, n_samples=5):
    """
    Visualize examples of correct and incorrect predictions.
    
    Args:
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        filenames (list): List of filenames corresponding to test samples
        n_samples (int): Number of examples to show for each category
        
    Returns:
        str: Path to saved plot
    """
    # Find correct and incorrect predictions
    correct_mask = y_test == y_pred
    incorrect_mask = ~correct_mask
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
    
    # Plot correct predictions
    correct_indices = np.where(correct_mask)[0]
    if len(correct_indices) > 0:
        sample_indices = correct_indices[:n_samples] if len(correct_indices) >= n_samples else correct_indices
        
        for i, idx in enumerate(sample_indices):
            axes[0, i].scatter(X_test[idx, 0], X_test[idx, 1], 
                            c='green' if y_test[idx] == 1 else 'red', s=100)
            axes[0, i].set_title(f"True: {'healthy' if y_test[idx] == 1 else 'apple scab'}")
            if filenames is not None and idx < len(filenames):
                axes[0, i].set_xlabel(filenames[idx])
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
    
    # Plot incorrect predictions
    incorrect_indices = np.where(incorrect_mask)[0]
    if len(incorrect_indices) > 0:
        sample_indices = incorrect_indices[:n_samples] if len(incorrect_indices) >= n_samples else incorrect_indices
        
        for i, idx in enumerate(sample_indices):
            axes[1, i].scatter(X_test[idx, 0], X_test[idx, 1], 
                           c='red' if y_pred[idx] == -1 else 'green', s=100)
            axes[1, i].set_title(f"True: {'healthy' if y_test[idx] == 1 else 'apple scab'}\nPred: {'healthy' if y_pred[idx] == 1 else 'apple scab'}")
            if filenames is not None and idx < len(filenames):
                axes[1, i].set_xlabel(filenames[idx])
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
    
    # Add row labels
    fig.text(0.02, 0.75, "Correct", ha="center", va="center", fontsize=12, rotation=90)
    fig.text(0.02, 0.25, "Incorrect", ha="center", va="center", fontsize=12, rotation=90)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(EVAL_DIR, exist_ok=True)
    filename = get_timestamped_filename("prediction_examples", "png")
    output_path = os.path.join(EVAL_DIR, filename)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path
