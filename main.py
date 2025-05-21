"""
Main entry point for Apple Scab detection system.
Runs the full pipeline from preprocessing to evaluation.
"""
import os
import numpy as np
import pandas as pd
import time

# Import configuration
from config import (
    DATASET_PATH, CLASS_APPLE_SCAB, CLASS_HEALTHY, 
    SVM_LEARNING_RATE, SVM_REGULARIZATION, SVM_EPOCHS, 
    TRAIN_TEST_SPLIT, get_timestamp
)

# Import modules
from src.preprocessing import preprocess_image
from src.feature_extraction import (
    extract_features, process_dataset, save_features, create_feature_scatter_plot
)
from src.svm import LinearSVM
from src.evaluation import (
    train_test_split, evaluate_classifier, 
    plot_confusion_matrix, save_evaluation_metrics, plot_comparison
)
from src.visualization import (
    plot_feature_distribution, visualize_feature_importance, 
    visualize_prediction_examples
)

def main():
    """
    Run the full Apple Scab detection pipeline.
    """
    print("\n=== Apple Scab Detection System ===\n")
    
    start_time = time.time()
    
    # Step 1: Process dataset and extract features
    print("\n1. Processing dataset and extracting features...")
    features_df = process_dataset(DATASET_PATH, [CLASS_APPLE_SCAB, CLASS_HEALTHY])
    
    # Save features to CSV
    features_path = save_features(features_df)
    print(f"   Features saved to: {features_path}")
    
    # Create feature scatter plot
    scatter_plot_path = create_feature_scatter_plot(features_df)
    print(f"   Feature scatter plot saved to: {scatter_plot_path}")
    
    # Additionally create feature distribution visualization
    dist_plot_path = plot_feature_distribution(features_df)
    print(f"   Feature distribution plot saved to: {dist_plot_path}")
    
    # Step 2: Prepare data for SVM training
    print("\n2. Preparing data for SVM training...")
    
    # Convert labels to numeric (-1 for apple scab, 1 for healthy)
    y = np.where(features_df['label'] == CLASS_HEALTHY, 1, -1)
    
    # Select features
    X = features_df[['mean_saturation', 'black_spot_ratio']].values
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, TRAIN_TEST_SPLIT)
    print(f"   Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Step 3: Train SVM classifier
    print(f"\n3. Training SVM classifier (learning_rate={SVM_LEARNING_RATE}, C={SVM_REGULARIZATION}, epochs={SVM_EPOCHS})...")
    svm = LinearSVM(
        learning_rate=SVM_LEARNING_RATE,
        regularization=SVM_REGULARIZATION,
        epochs=SVM_EPOCHS
    )
    
    svm.train(X_train, y_train)
    
    # Plot loss curve
    loss_curve_path = svm.plot_loss_curve()
    print(f"   Loss curve saved to: {loss_curve_path}")
    
    # Plot decision boundary
    decision_boundary_path = svm.plot_decision_boundary(X_train, y_train)
    print(f"   Decision boundary plot saved to: {decision_boundary_path}")
    
    # Visualize feature importance
    importance_path = visualize_feature_importance(svm, ['Mean Saturation', 'Black Spot Ratio'])
    print(f"   Feature importance plot saved to: {importance_path}")
    
    # Step 4: Evaluate classifier
    print("\n4. Evaluating classifier...")
    
    # Make predictions on test set
    y_pred = svm.predict(X_test)
    
    # Compute evaluation metrics
    metrics = evaluate_classifier(y_test, y_pred)
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-score:  {metrics['f1']:.4f}")
    
    # Save evaluation metrics
    metrics_path = save_evaluation_metrics(metrics)
    print(f"   Evaluation metrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    confusion_matrix_path = plot_confusion_matrix(
        y_test, y_pred, 
        ['apple scab', 'healthy']
    )
    print(f"   Confusion matrix saved to: {confusion_matrix_path}")
    
    # Plot prediction examples
    examples_path = visualize_prediction_examples(
        X_test, y_test, y_pred,
        filenames=None,  # Would need to track filenames during train/test split
        n_samples=5
    )
    print(f"   Prediction examples saved to: {examples_path}")
    
    # Step 5: Create comparison visualization
    print("\n5. Creating comparison visualization...")
    comparison_path = plot_comparison(X, y, svm)
    print(f"   Comparison plot saved to: {comparison_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nPipeline completed in {elapsed_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()
