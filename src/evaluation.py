"""
Evaluation module for the SVM classifier.
Computes metrics and creates visualizations for classifier performance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (EVAL_DIR, COMPARISON_DIR, 
                   get_timestamp, get_timestamped_filename)

def train_test_split(X, y, train_ratio=0.8, random_state=42):
    """
    Split data into training and test sets.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        train_ratio (float): Ratio of training data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Get number of samples
    n_samples = len(X)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def evaluate_classifier(y_true, y_pred):
    """
    Evaluate classifier performance.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # True positives, false positives, etc.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        class_names (list): Class names
        
    Returns:
        str: Path to saved confusion matrix plot
    """
    # Compute confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    
    # Fill confusion matrix
    # [0,0] = TN, [0,1] = FP
    # [1,0] = FN, [1,1] = TP
    cm[0, 0] = np.sum((y_true == -1) & (y_pred == -1))  # TN
    cm[0, 1] = np.sum((y_true == -1) & (y_pred == 1))   # FP
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == -1))   # FN
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))    # TP
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text to cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(EVAL_DIR, exist_ok=True)
    filename = get_timestamped_filename("confusion_matrix", "png")
    output_path = os.path.join(EVAL_DIR, filename)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path

def save_evaluation_metrics(metrics):
    """
    Save evaluation metrics to CSV.
    
    Args:
        metrics (dict): Dictionary with evaluation metrics
        
    Returns:
        str: Path to saved CSV file
    """
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame([metrics])
    
    # Create output directory if it doesn't exist
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Generate timestamped filename
    filename = get_timestamped_filename("evaluation", "csv")
    output_path = os.path.join(EVAL_DIR, filename)
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False)
    
    return output_path

def plot_comparison(X, y, svm_model):
    """
    Create side-by-side comparison plot:
    Left: feature scatter plot before training
    Right: feature scatter plot with SVM decision boundary
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        svm_model: Trained SVM model
        
    Returns:
        str: Path to saved comparison plot
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Feature scatter plot before training
    axes[0].scatter(X[y == -1][:, 0], X[y == -1][:, 1], 
                   c='red', label='apple scab', alpha=0.7)
    axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
                   c='green', label='healthy', alpha=0.7)
    axes[0].set_title('Feature Space before SVM')
    axes[0].set_xlabel('Mean Saturation')
    axes[0].set_ylabel('Black Spot Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Feature scatter plot with SVM decision boundary
    # Normalize the features first
    X_norm = svm_model.normalize_features(X)
    
    # Create a mesh grid
    h = .02  # step size in the mesh
    x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
    y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Compute decision function on the mesh grid
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], svm_model.w) + svm_model.b
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    axes[1].contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'],
                   linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    # Fill the regions
    axes[1].contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')],
                    colors=['#FFAAAA', '#AAFFAA'], alpha=0.3)
    
    # Plot the data
    axes[1].scatter(X_norm[y == -1][:, 0], X_norm[y == -1][:, 1], 
                   c='red', label='apple scab', alpha=0.7)
    axes[1].scatter(X_norm[y == 1][:, 0], X_norm[y == 1][:, 1], 
                   c='green', label='healthy', alpha=0.7)
    
    # Highlight support vectors if available
    if svm_model.support_vectors is not None and len(svm_model.support_vectors) > 0:
        axes[1].scatter(svm_model.support_vectors[:, 0], svm_model.support_vectors[:, 1], 
                      s=100, linewidth=1, facecolors='none', 
                      edgecolors='black', label='Support Vectors')
    
    axes[1].set_title('SVM Decision Boundary')
    axes[1].set_xlabel('Normalized Mean Saturation')
    axes[1].set_ylabel('Normalized Black Spot Ratio')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    filename = get_timestamped_filename("pre_vs_post_svm", "png")
    output_path = os.path.join(COMPARISON_DIR, filename)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path
