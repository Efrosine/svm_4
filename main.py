"""
Main entry point for Apple Scab detection system.
Provides a command-line interface for running different stages of the pipeline.
"""
import os
import numpy as np
import pandas as pd
import time
import argparse
import pickle

# Import configuration
from config import (
    DATASET_PATH, CLASS_APPLE_SCAB, CLASS_HEALTHY, 
    SVM_LEARNING_RATE, SVM_REGULARIZATION, SVM_EPOCHS, 
    TRAIN_TEST_SPLIT, get_timestamp, OUTPUT_DIR
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

def setup_argparse():
    """
    Setup command line argument parser.
    """
    parser = argparse.ArgumentParser(
        description='Apple Scab Detection System CLI Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'extract', 'train', 'test'],
        default='full',
        help='Pipeline mode: full (run entire pipeline), extract (only extract features), '
             'train (train SVM from features), test (evaluate model on test data)'
    )
    
    # Dataset paths
    parser.add_argument(
        '--dataset',
        type=str,
        default=DATASET_PATH,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--test-dataset',
        type=str,
        default=None,
        help='Path to test dataset directory (if different from training)'
    )
    
    # Feature extraction settings
    parser.add_argument(
        '--feature-path',
        type=str,
        default=None,
        help='Path to pre-extracted features CSV file (for train mode)'
    )
    
    parser.add_argument(
        '--save-features',
        type=str,
        default=None,
        help='Path to save extracted features (default: auto-generated timestamped filename)'
    )
    
    parser.add_argument(
        '--feature-method',
        type=str,
        choices=['default', 'advanced'],
        default='default',
        help='Feature extraction method: default (mean saturation and black spot ratio) or '
             'advanced (includes additional texture features)'
    )
    
    # Preprocessing options
    parser.add_argument(
        '--preprocess-method',
        type=str,
        choices=['default', 'enhanced'],
        default='default',
        help='Image preprocessing method: default (standard cropping) or '
             'enhanced (with additional noise reduction)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step (use raw images)'
    )
    
    # SVM hyperparameters
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=SVM_LEARNING_RATE,
        help='SVM learning rate'
    )
    
    parser.add_argument(
        '--regularization',
        type=float,
        default=SVM_REGULARIZATION,
        help='SVM regularization parameter (C)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=SVM_EPOCHS,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--optimization',
        type=str,
        choices=['gradient', 'subgradient'],
        default='gradient',
        help='SVM optimization method: gradient or subgradient descent'
    )
    
    parser.add_argument(
        '--use-mini-batch',
        action='store_true',
        help='Use mini-batch optimization'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Mini-batch size (if using mini-batch optimization)'
    )
    
    # Split and balance options
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=TRAIN_TEST_SPLIT,
        help='Train/test split ratio (percentage of data for training)'
    )
    
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance dataset classes'
    )
    
    parser.add_argument(
        '--balance-method',
        type=str,
        choices=['oversampling', 'undersampling'],
        default='oversampling',
        help='Method for balancing classes: oversampling (duplicate minority) or '
             'undersampling (reduce majority)'
    )
    
    # Model loading/saving options
    parser.add_argument(
        '--save-model',
        type=str,
        default=None,
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to load pre-trained model'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--visualize-dir',
        type=str,
        default=None,
        help='Directory to save visualization files'
    )
    
    # Specific image testing
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to a specific image for testing (test mode only)'
    )
    
    # Feature selection
    parser.add_argument(
        '--selected-features',
        nargs='+',
        default=['mean_saturation', 'black_spot_ratio'],
        help='List of feature names to use for training'
    )
    
    return parser

def extract_features_pipeline(args):
    """
    Run the feature extraction pipeline.
    
    Args:
        args: Command line arguments
    
    Returns:
        pandas.DataFrame: Extracted features
    """
    print("\n=== Feature Extraction Pipeline ===\n")
    start_time = time.time()
    
    print("\nProcessing dataset and extracting features...")
    
    # For now, we'll just print the selected methods without applying them
    # since the current implementation doesn't support these parameters
    if not args.skip_preprocessing:
        if args.preprocess_method == 'enhanced':
            print("Using enhanced preprocessing with noise reduction")
            print("Note: Enhanced preprocessing is not fully implemented yet. Using default.")
        else:
            print("Using default preprocessing")
    else:
        print("Skipping preprocessing (using raw images)")
        print("Note: Skipping preprocessing is not fully implemented yet. Using default.")
    
    # Handle feature extraction method selection
    if args.feature_method == 'advanced':
        print("Using advanced feature extraction with texture features")
        print("Note: Advanced feature extraction is not fully implemented yet. Using default.")
    else:
        print("Using default feature extraction (mean saturation and black spot ratio)")
    
    # Process dataset using the current implementation
    features_df = process_dataset(
        args.dataset, 
        [CLASS_APPLE_SCAB, CLASS_HEALTHY]
    )
    
    # Save features to CSV
    features_path = save_features(features_df)
    if args.save_features:
        # Rename the saved file to the user-specified path
        os.rename(features_path, args.save_features)
        features_path = args.save_features
    print(f"Features saved to: {features_path}")
    
    if args.visualize:
        # Create feature scatter plot
        scatter_plot_path = create_feature_scatter_plot(features_df)
        print(f"Feature scatter plot saved to: {scatter_plot_path}")
        
        # Create feature distribution visualization
        dist_plot_path = plot_feature_distribution(features_df)
        print(f"Feature distribution plot saved to: {dist_plot_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nFeature extraction completed in {elapsed_time:.2f} seconds.")
    
    return features_df, features_path

def train_svm_pipeline(args, features_df=None):
    """
    Run the SVM training pipeline.
    
    Args:
        args: Command line arguments
        features_df: Pre-loaded features DataFrame (optional)
    
    Returns:
        tuple: (svm, X_train, X_test, y_train, y_test)
    """
    print("\n=== SVM Training Pipeline ===\n")
    start_time = time.time()
    
    # Load features if not provided
    if features_df is None:
        if not args.feature_path:
            raise ValueError("Feature path must be provided in train mode when no features are loaded")
        
        print(f"\nLoading features from {args.feature_path}...")
        features_df = pd.read_csv(args.feature_path)
    
    print("\nPreparing data for SVM training...")
    
    # Convert labels to numeric (-1 for apple scab, 1 for healthy)
    y = np.where(features_df['label'] == CLASS_HEALTHY, 1, -1)
    
    # Select features based on user input
    try:
        X = features_df[args.selected_features].values
        print(f"Using selected features: {', '.join(args.selected_features)}")
    except KeyError as e:
        print(f"Warning: Selected feature not found in dataset. Using default features.")
        X = features_df[['mean_saturation', 'black_spot_ratio']].values
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, args.train_ratio)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Balance dataset if requested
    if args.balance:
        print(f"Balancing dataset using {args.balance_method} method...")
        min_class = -1 if np.sum(y_train == -1) < np.sum(y_train == 1) else 1
        min_indices = np.where(y_train == min_class)[0]
        maj_indices = np.where(y_train != min_class)[0]
        
        if args.balance_method == 'oversampling':
            # Duplicate minority samples to match majority class count
            resampled_indices = np.random.choice(
                min_indices, 
                size=len(maj_indices) - len(min_indices), 
                replace=True
            )
            
            balanced_indices = np.concatenate([np.arange(len(y_train)), resampled_indices])
            X_train = X_train[balanced_indices]
            y_train = y_train[balanced_indices]
        else:  # undersampling
            # Randomly select from majority class to match minority class count
            undersampled_indices = np.random.choice(
                maj_indices,
                size=len(min_indices),
                replace=False
            )
            
            balanced_indices = np.concatenate([min_indices, undersampled_indices])
            X_train = X_train[balanced_indices]
            y_train = y_train[balanced_indices]
            
        print(f"Balanced training data: {len(X_train)} samples")
    
    print(f"\nTraining SVM classifier (learning_rate={args.learning_rate}, "
          f"C={args.regularization}, epochs={args.epochs}, "
          f"optimization={args.optimization})")
    
    # Configure SVM with parameters
    svm_params = {
        'learning_rate': args.learning_rate,
        'regularization': args.regularization,
        'epochs': args.epochs
    }
    
    # Note: optimization method is not currently implemented in the LinearSVM class
    if args.optimization != 'gradient':
        print(f"Note: {args.optimization} optimization is not fully implemented yet. Using gradient descent.")
    
    if args.use_mini_batch:
        print(f"Note: Mini-batch optimization is not fully implemented yet. Using full-batch gradient descent.")
    
    svm = LinearSVM(**svm_params)
    
    svm.train(X_train, y_train)
    
    if args.visualize:
        # Plot loss curve
        loss_curve_path = svm.plot_loss_curve()
        print(f"Loss curve saved to: {loss_curve_path}")
        
        # Plot decision boundary
        decision_boundary_path = svm.plot_decision_boundary(X_train, y_train)
        print(f"Decision boundary plot saved to: {decision_boundary_path}")
        
        # Visualize feature importance
        importance_path = visualize_feature_importance(
            svm, 
            args.selected_features
        )
        print(f"Feature importance plot saved to: {importance_path}")
    
    # Save the model if requested
    if args.save_model:
        model_path = args.save_model
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    else:
        timestamp = get_timestamp()
        model_path = os.path.join(OUTPUT_DIR, f"svm_model_{timestamp}.pkl")
    
    # Save model based on file extension
    if model_path.lower().endswith('.csv'):
        from src.model_io import save_model_to_csv
        save_model_to_csv(svm, model_path)
        print(f"Model saved in CSV format to: {model_path}")
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(svm, f)
        print(f"Model saved in pickle format to: {model_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nSVM training completed in {elapsed_time:.2f} seconds.")
    
    return svm, X_train, X_test, y_train, y_test

def evaluate_pipeline(args, svm=None, X_test=None, y_test=None):
    """
    Run the evaluation pipeline.
    
    Args:
        args: Command line arguments
        svm: Trained SVM model (optional)
        X_test: Test feature data (optional)
        y_test: Test labels (optional)
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n=== Evaluation Pipeline ===\n")
    start_time = time.time()
    
    # Load model if not provided
    if svm is None:
        if not args.load_model:
            raise ValueError("Model path must be provided in test mode when no model is loaded")
        
        print(f"\nLoading model from {args.load_model}...")
        if args.load_model.lower().endswith('.csv'):
            from src.model_io import load_model_from_csv
            svm = load_model_from_csv(args.load_model)
            print("Model loaded from CSV format.")
        else:
            with open(args.load_model, 'rb') as f:
                svm = pickle.load(f)
            print("Model loaded from pickle format.")
    
    # Load test data if not provided
    if X_test is None or y_test is None:
        if args.image:
            # Process a single image
            print(f"\nProcessing single image: {args.image}")
            filename = os.path.basename(args.image)
            
            # Determine label from filename or path (if possible)
            if CLASS_APPLE_SCAB in args.image.lower():
                label = CLASS_APPLE_SCAB
            elif CLASS_HEALTHY in args.image.lower():
                label = CLASS_HEALTHY
            else:
                label = "unknown"  # Cannot determine ground truth
            
            # Print info about preprocessing options
            if not args.skip_preprocessing:
                if args.preprocess_method == 'enhanced':
                    print("Using enhanced preprocessing (not fully implemented yet)")
                else:
                    print("Using default preprocessing")
            else:
                print("Skipping preprocessing (not fully implemented yet)")
                
            # Preprocess and extract features
            hsv_img, processed_img = preprocess_image(args.image, save_preview=True)
            
            # Print info about feature extraction options
            if args.feature_method == 'advanced':
                print("Using advanced feature extraction (not fully implemented yet)")
            else:
                print("Using default feature extraction")
            
            # Extract features
            mean_saturation, black_spot_ratio = extract_features(hsv_img)
            features = {
                'mean_saturation': mean_saturation,
                'black_spot_ratio': black_spot_ratio
            }
            
            # Create a test dataset with one sample
            try:
                X_test = np.array([[features[f] for f in args.selected_features]])
            except KeyError:
                print(f"Warning: Selected feature not found. Using default features.")
                X_test = np.array([[features['mean_saturation'], features['black_spot_ratio']]])
            
            y_test = np.array([-1 if label == CLASS_APPLE_SCAB else 1])
            
            # Print extracted features
            features_str = ", ".join([f"{f}={features.get(f, 'N/A'):.4f}" for f in args.selected_features])
            print(f"Extracted features: {features_str}")
            
        else:
            # Process a test dataset
            test_path = args.test_dataset or args.dataset
            print(f"\nProcessing test dataset: {test_path}")
            
            # Handle preprocessing and feature extraction options - print info only
            if not args.skip_preprocessing:
                if args.preprocess_method == 'enhanced':
                    print("Using enhanced preprocessing (not fully implemented yet)")
                else:
                    print("Using default preprocessing")
            else:
                print("Skipping preprocessing (not fully implemented yet)")
                
            if args.feature_method == 'advanced':
                print("Using advanced feature extraction (not fully implemented yet)")
            else:
                print("Using default feature extraction")
            
            # Extract features from test dataset
            features_df = process_dataset(
                test_path, 
                [CLASS_APPLE_SCAB, CLASS_HEALTHY]
            )
            
            # Convert labels to numeric (-1 for apple scab, 1 for healthy)
            y = np.where(features_df['label'] == CLASS_HEALTHY, 1, -1)
            
            # Select features based on user input
            try:
                X = features_df[args.selected_features].values
            except KeyError:
                print(f"Warning: Selected feature not found. Using default features.")
                X = features_df[['mean_saturation', 'black_spot_ratio']].values
            
            # Use all data as test data
            X_test = X
            y_test = y
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = svm.predict(X_test)
    
    # For single image testing, print a clear classification result
    if args.image:
        prediction_label = CLASS_HEALTHY if y_pred[0] == 1 else CLASS_APPLE_SCAB
        print("\n" + "="*50)
        print(f"CLASSIFICATION RESULT: {prediction_label.upper()}")
        print("="*50 + "\n")
    
    # Compute evaluation metrics
    print("\nEvaluating classifier...")
    metrics = evaluate_classifier(y_test, y_pred)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    
    if args.visualize:
        # Save evaluation metrics
        metrics_path = save_evaluation_metrics(metrics)
        print(f"Evaluation metrics saved to: {metrics_path}")
        
        # Plot confusion matrix
        confusion_matrix_path = plot_confusion_matrix(
            y_test, y_pred, 
            ['apple scab', 'healthy']
        )
        print(f"Confusion matrix saved to: {confusion_matrix_path}")
        
        if len(X_test) > 1:  # Don't try to plot examples for a single image
            # Plot prediction examples
            examples_path = visualize_prediction_examples(
                X_test, y_test, y_pred,
                filenames=None,
                n_samples=min(5, len(X_test))
            )
            print(f"Prediction examples saved to: {examples_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds.")
    
    return metrics

def full_pipeline(args):
    """
    Run the full pipeline from feature extraction to evaluation.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Apple Scab Detection System: Full Pipeline ===\n")
    start_time = time.time()
    
    # Step 1: Extract features
    features_df, _ = extract_features_pipeline(args)
    
    # Step 2: Train SVM
    svm, X_train, X_test, y_train, y_test = train_svm_pipeline(args, features_df)
    
    # Step 3: Evaluate
    metrics = evaluate_pipeline(args, svm, X_test, y_test)
    
    # Step 4: Create comparison visualization
    if args.visualize:
        print("\nCreating comparison visualization...")
        # Convert labels to numeric (-1 for apple scab, 1 for healthy)
        y = np.where(features_df['label'] == CLASS_HEALTHY, 1, -1)
        
        # Select features based on user input
        try:
            X = features_df[args.selected_features].values
        except KeyError:
            print(f"Warning: Selected feature not found. Using default features.")
            X = features_df[['mean_saturation', 'black_spot_ratio']].values
        
        comparison_path = plot_comparison(X, y, svm)
        print(f"Comparison plot saved to: {comparison_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nFull pipeline completed in {elapsed_time:.2f} seconds.")

def main():
    """
    Main entry point for the CLI pipeline.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run the requested pipeline mode
    if args.mode == 'full':
        full_pipeline(args)
    elif args.mode == 'extract':
        extract_features_pipeline(args)
    elif args.mode == 'train':
        train_svm_pipeline(args)
    elif args.mode == 'test':
        evaluate_pipeline(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
if __name__ == "__main__":
    main()
