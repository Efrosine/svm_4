# Apple Scab Detection System Documentation

This documentation provides a detailed explanation of each function used in the Apple Scab Detection System.

## Table of Contents

1. [Configuration (config.py)](#configuration)
2. [Preprocessing (preprocessing.py)](#preprocessing)
3. [Feature Extraction (feature_extraction.py)](#feature-extraction)
4. [SVM Classification (svm.py)](#svm-classification)
5. [Evaluation (evaluation.py)](#evaluation)
6. [Visualization (visualization.py)](#visualization)
7. [Main Pipeline (main.py)](#main-pipeline)

## Configuration

### get_timestamp()

**Description:** Generates a current timestamp in YYYY-MM-DD_HH-MM format.  
**Arguments:** None  
**Returns:** String with formatted current timestamp.  
**Usage:** Used for creating unique filenames for outputs.

### get_timestamped_filename(prefix, extension)

**Description:** Creates a timestamped filename with the given prefix and extension.  
**Arguments:**

- `prefix` (str): Prefix for the filename
- `extension` (str): File extension without dot

**Returns:** String with the formatted filename pattern `prefix_YYYY-MM-DD_HH-MM.extension`.  
**Usage:** Used for organizing output files with consistent naming.

## Preprocessing

### load_image(image_path)

**Description:** Loads an image from the specified path using matplotlib.  
**Arguments:**

- `image_path` (str): Path to the image file

**Returns:** numpy.ndarray - RGB image as a NumPy array with shape (H, W, 3).  
**Usage:** First step in image processing pipeline to read the image file.

### crop_background(img)

**Description:** Removes background from image and crops to bounding box.  
**Arguments:**

- `img` (numpy.ndarray): RGB image as NumPy array

**Returns:** numpy.ndarray - Cropped RGB image without background.  
**Usage:** Focuses analysis on the leaf by removing white background.

### rgb_to_hsv(img)

**Description:** Converts RGB image to HSV using manual implementation with NumPy.  
**Arguments:**

- `img` (numpy.ndarray): RGB image as NumPy array with values in [0, 255]

**Returns:** numpy.ndarray - HSV image with H in [0, 360], S in [0, 1], V in [0, 1].  
**Usage:** Transforms color representation to HSV which is better for saturation analysis.

### preprocess_image(image_path)

**Description:** Applies the full preprocessing pipeline: load, crop, and convert to HSV.  
**Arguments:**

- `image_path` (str): Path to the image file

**Returns:** tuple - (hsv_img, rgb_img) containing processed HSV and RGB images.  
**Usage:** Complete preprocessing workflow for each image in the dataset.

## Feature Extraction

### extract_features(hsv_img)

**Description:** Extracts mean saturation and black spot ratio features from a HSV image.  
**Arguments:**

- `hsv_img` (numpy.ndarray): HSV image

**Returns:** tuple - (mean_saturation, black_spot_ratio)  
**Usage:** Computes key metrics for distinguishing healthy leaves from infected ones.

### process_dataset(dataset_path, classes)

**Description:** Processes all images in the dataset and extracts features.  
**Arguments:**

- `dataset_path` (str): Path to the dataset
- `classes` (list): List of class names to process

**Returns:** pandas.DataFrame - DataFrame with features for all images.  
**Usage:** Batch processes all images and organizes their features.

### save_features(features_df)

**Description:** Saves extracted features to CSV file.  
**Arguments:**

- `features_df` (pandas.DataFrame): DataFrame with features

**Returns:** str - Path to the saved CSV file.  
**Usage:** Persists feature data for future reference.

### create_feature_scatter_plot(features_df)

**Description:** Creates scatter plot of features by class.  
**Arguments:**

- `features_df` (pandas.DataFrame): DataFrame with features

**Returns:** str - Path to saved plot.  
**Usage:** Visualizes feature space to assess class separability.

## SVM Classification

### LinearSVM.**init**(learning_rate=0.01, regularization=1.0, epochs=1000)

**Description:** Initializes SVM classifier parameters.  
**Arguments:**

- `learning_rate` (float): Learning rate for gradient descent
- `regularization` (float): Regularization parameter (C)
- `epochs` (int): Number of training epochs

**Returns:** None  
**Usage:** Creates and configures a new SVM classifier instance.

### LinearSVM.normalize_features(X)

**Description:** Normalizes features using means and standard deviations from training.  
**Arguments:**

- `X` (numpy.ndarray): Feature matrix

**Returns:** numpy.ndarray - Normalized features.  
**Usage:** Standardizes input features for better SVM performance.

### LinearSVM.train(X, y)

**Description:** Trains the SVM classifier using gradient descent.  
**Arguments:**

- `X` (numpy.ndarray): Training features, shape (n_samples, n_features)
- `y` (numpy.ndarray): Training labels, should be -1 or 1

**Returns:** self - Trained classifier.  
**Usage:** Fits the SVM model to the training data.

### LinearSVM.predict(X)

**Description:** Makes predictions with the trained SVM model.  
**Arguments:**

- `X` (numpy.ndarray): Features to predict, shape (n_samples, n_features)

**Returns:** numpy.ndarray - Predicted class labels (-1 or 1).  
**Usage:** Classifies new data samples.

### LinearSVM.decision_function(X)

**Description:** Calculates decision values (distances from decision boundary) for samples.  
**Arguments:**

- `X` (numpy.ndarray): Input features

**Returns:** numpy.ndarray - Decision values.  
**Usage:** Gets confidence scores for classification.

### LinearSVM.plot_loss_curve()

**Description:** Plots the loss curve from training history.  
**Arguments:** None  
**Returns:** str - Path to saved loss curve plot.  
**Usage:** Visualizes model convergence during training.

### LinearSVM.plot_decision_boundary(X, y)

**Description:** Visualizes the decision boundary of the trained SVM.  
**Arguments:**

- `X` (numpy.ndarray): Input features
- `y` (numpy.ndarray): True labels

**Returns:** str - Path to saved decision boundary plot.  
**Usage:** Shows how the model separates classes in feature space.

## Evaluation

### train_test_split(X, y, train_ratio=0.8, random_state=42)

**Description:** Splits data into training and test sets.  
**Arguments:**

- `X` (numpy.ndarray): Features
- `y` (numpy.ndarray): Labels
- `train_ratio` (float): Ratio of training data
- `random_state` (int): Random seed for reproducibility

**Returns:** tuple - (X_train, X_test, y_train, y_test)  
**Usage:** Creates separate datasets for model training and evaluation.

### evaluate_classifier(y_true, y_pred)

**Description:** Evaluates classifier performance.  
**Arguments:**

- `y_true` (numpy.ndarray): True labels
- `y_pred` (numpy.ndarray): Predicted labels

**Returns:** dict - Dictionary with evaluation metrics.  
**Usage:** Computes accuracy, precision, recall, and F1 score.

### plot_confusion_matrix(y_true, y_pred, class_names)

**Description:** Plots confusion matrix.  
**Arguments:**

- `y_true` (numpy.ndarray): True labels
- `y_pred` (numpy.ndarray): Predicted labels
- `class_names` (list): Class names

**Returns:** str - Path to saved confusion matrix plot.  
**Usage:** Visualizes true positives, false positives, etc.

### save_evaluation_metrics(metrics, class_labels)

**Description:** Saves evaluation metrics to CSV file.  
**Arguments:**

- `metrics` (dict): Dictionary with evaluation metrics
- `class_labels` (list): List of class labels

**Returns:** str - Path to saved metrics file.  
**Usage:** Persists performance metrics for record keeping.

### plot_comparison(metrics_before, metrics_after)

**Description:** Plots comparison of metrics before and after improvements.  
**Arguments:**

- `metrics_before` (dict): Metrics from first model
- `metrics_after` (dict): Metrics from improved model

**Returns:** str - Path to saved comparison plot.  
**Usage:** Visualizes improvements between model iterations.

## Visualization

### plot_feature_distribution(features_df)

**Description:** Plots distribution of features by class.  
**Arguments:**

- `features_df` (pandas.DataFrame): DataFrame with features

**Returns:** str - Path to saved plot.  
**Usage:** Shows how feature values are distributed across classes.

### visualize_feature_importance(svm_model, feature_names)

**Description:** Visualizes feature importance based on SVM weights.  
**Arguments:**

- `svm_model`: Trained SVM model
- `feature_names` (list): Names of features

**Returns:** str - Path to saved plot.  
**Usage:** Shows which features contribute most to classification decisions.

### visualize_prediction_examples(X_test, y_test, y_pred, filenames=None, n_samples=5)

**Description:** Shows examples of predictions (correct and incorrect).  
**Arguments:**

- `X_test` (numpy.ndarray): Test features
- `y_test` (numpy.ndarray): True labels
- `y_pred` (numpy.ndarray): Predicted labels
- `filenames` (list, optional): List of filenames
- `n_samples` (int): Number of examples to show

**Returns:** str - Path to saved visualization.  
**Usage:** Helps understand model performance through visual examples.

## Main Pipeline

### main()

**Description:** Runs the full Apple Scab detection pipeline.  
**Arguments:** None  
**Returns:** None  
**Usage:** Entry point for the full system, coordinates execution of all components.

The main function executes the following key steps:

1. Processes dataset and extracts features
2. Prepares data for SVM training
3. Trains SVM classifier
4. Evaluates classifier performance
5. Generates visualizations and saves results

```
cd /home/efrosine/Desktop/project/svm_4 && source venv/bin/activate && python main.py
```

source venv/bin/activate && python main.py --mode extract --dataset dataset/apple --save-features output/features_test_extract.csv

source venv/bin/activate && python main.py --mode train --feature-path output/features_test_extract.csv --save-model output/test_model.csv

source venv/bin/activate && python main.py --mode test --load-model output/test_model.csv --dataset dataset/apple

source venv/bin/activate && python main.py --mode test --load-model output/test_model.csv --image "dataset/apple/apple scab/f3303560-12ce-4e37-b62c-bfb5aad8aa0d\_\_\_FREC_Scab 3332.JPG"
