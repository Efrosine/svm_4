"""
Feature extraction module for Apple Scab detection.
Extracts mean saturation and black spot ratio features from HSV images.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from config import (VALUE_THRESHOLD, OUTPUT_DIR, FEATURES_DIR, 
                    get_timestamp, get_timestamped_filename)

def extract_features(hsv_img):
    """
    Extract features from a HSV image using manual loops.
    
    Args:
        hsv_img (numpy.ndarray): HSV image
        
    Returns:
        tuple: (mean_saturation, black_spot_ratio)
    """
    # Get image dimensions
    height, width = hsv_img.shape[:2]
    total_pixels = height * width
    
    # Initialize counters
    saturation_sum = 0.0
    black_spot_count = 0
    
    # Extract mean saturation (S channel) and black spot ratio using manual loops
    for y in range(height):
        for x in range(width):
            # Add saturation value to sum
            saturation_sum += hsv_img[y, x, 1]
            
            # Check if this pixel is a black spot (V < VALUE_THRESHOLD)
            if hsv_img[y, x, 2] < VALUE_THRESHOLD:
                black_spot_count += 1
    
    # Calculate mean saturation and black spot ratio
    mean_saturation = saturation_sum / total_pixels
    black_spot_ratio = black_spot_count / total_pixels
    
    return mean_saturation, black_spot_ratio

def process_dataset(dataset_path, classes):
    """
    Process all images in the dataset and extract features.
    
    Args:
        dataset_path (str): Path to the dataset
        classes (list): List of class names to process
        
    Returns:
        pandas.DataFrame: DataFrame with features for all images
    """
    from src.preprocessing import preprocess_image
    
    # Initialize lists to store results
    filenames = []
    labels = []
    mean_saturations = []
    black_spot_ratios = []
    
    # Process each class
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        
        # Check if directory exists
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
            
        # Process each image in the class directory
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_dir, filename)
                
                # Preprocess image - enable preview generation for a sample of images
                generate_preview = (filename.startswith('0') or filename.startswith('1')) and np.random.random() < 0.1  # random sample
                hsv_img, _ = preprocess_image(image_path, save_preview=generate_preview)
                
                if hsv_img is not None:
                    # Extract features
                    mean_saturation, black_spot_ratio = extract_features(hsv_img)
                    
                    # Store results
                    filenames.append(filename)
                    labels.append(class_name)
                    mean_saturations.append(mean_saturation)
                    black_spot_ratios.append(black_spot_ratio)
    
    # Create DataFrame
    features_df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'mean_saturation': mean_saturations,
        'black_spot_ratio': black_spot_ratios
    })
    
    return features_df

def save_features(features_df):
    """
    Save extracted features to CSV file.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate timestamped filename
    filename = f"features_{get_timestamp()}.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save to CSV
    features_df.to_csv(output_path, index=False)
    
    return output_path

def create_feature_scatter_plot(features_df):
    """
    Create a 2D scatter plot of the extracted features.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features
        
    Returns:
        str: Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = features_df['label'].unique()
    
    # Define colors for each class
    colors = ['red', 'green', 'blue', 'orange']
    
    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        class_data = features_df[features_df['label'] == label]
        plt.scatter(
            class_data['mean_saturation'], 
            class_data['black_spot_ratio'],
            c=colors[i % len(colors)],
            label=label,
            alpha=0.7
        )
    
    # Add labels and legend
    plt.xlabel('Mean Saturation')
    plt.ylabel('Black Spot Ratio')
    plt.title('Feature Space: Mean Saturation vs Black Spot Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = get_timestamped_filename("feature_space", "png")
    output_path = os.path.join(FEATURES_DIR, filename)
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path
