"""
Configuration settings for the Apple Scab Detection System
"""
import os
from datetime import datetime

# Dataset paths
DATASET_PATH = "dataset/apple"
CLASS_APPLE_SCAB = "apple scab"
CLASS_HEALTHY = "healthy"

# Preprocessing parameters
BACKGROUND_THRESHOLD = 220  # R,G,B > 220 considered as background
CROP_METHOD = 'hsv'  # 'threshold', 'otsu', or 'hsv'
CROP_PADDING = 15  # Base padding around the detected leaf
CROP_ADAPTIVE_PADDING = True  # Automatically adjust padding based on image size
CROP_DENOISE = True  # Apply noise reduction to segmentation mask
CROP_MIN_LEAF_AREA = 0.01  # Minimum ratio of image area that must be leaf

# Feature extraction parameters
VALUE_THRESHOLD = 0.2  # V < 0.2 considered as black spots

# SVM parameters
SVM_LEARNING_RATE = 0.01
SVM_REGULARIZATION = 1.0  # C parameter
SVM_EPOCHS = 1000
TRAIN_TEST_SPLIT = 0.8  # 80% for training, 20% for testing

# Output paths
OUTPUT_DIR = "output"
VISUALS_DIR = "visuals"
PREVIEW_DIR = os.path.join(VISUALS_DIR, "previews")
FEATURES_DIR = os.path.join(VISUALS_DIR, "features")
SVM_DIR = os.path.join(VISUALS_DIR, "svm")
EVAL_DIR = os.path.join(VISUALS_DIR, "eval")
COMPARISON_DIR = os.path.join(VISUALS_DIR, "comparisons")

# Function for creating timestamped filenames
def get_timestamp():
    """Generate current timestamp in YYYY-MM-DD_HH-MM format"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def get_timestamped_filename(prefix, extension):
    """Create a timestamped filename with given prefix and extension"""
    return f"{prefix}_{get_timestamp()}.{extension}"
