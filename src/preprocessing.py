"""
Preprocessing module for Apple Scab detection system.
Handles image loading, cropping via background removal, and RGB to HSV conversion.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive, can run without display)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import BACKGROUND_THRESHOLD

def load_image(image_path):
    """
    Load an image from the specified path using matplotlib.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: RGB image as a NumPy array with shape (H, W, 3)
    """
    try:
        # Read the image using matplotlib
        img = mpimg.imread(image_path)
        
        # Check if image is loaded as float (0-1) and convert to uint8 (0-255) if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        # Ensure image has 3 channels (RGB)
        if len(img.shape) == 2:  # Grayscale image
            img = np.stack([img] * 3, axis=2)
        elif img.shape[2] == 4:  # RGBA image
            img = img[:, :, :3]  # Drop alpha channel
            
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def crop_background(img):
    """
    Remove background from image and crop to bounding box.
    Background is defined as R > 220, G > 220, B > 220.
    
    Args:
        img (numpy.ndarray): RGB image as NumPy array
        
    Returns:
        numpy.ndarray: Cropped RGB image
    """
    # Create binary mask for non-background pixels
    mask = ~np.all(img > BACKGROUND_THRESHOLD, axis=2)
    
    # Find non-zero indices (row, column) of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find min/max row and column to form bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop the original image to the bounding box
    cropped_img = img[rmin:rmax+1, cmin:cmax+1]
    
    return cropped_img

def rgb_to_hsv(img):
    """
    Convert RGB image to HSV using manual implementation with NumPy.
    
    Args:
        img (numpy.ndarray): RGB image as NumPy array with values in [0, 255]
        
    Returns:
        numpy.ndarray: HSV image with H in [0, 360], S in [0, 1], V in [0, 1]
    """
    # Normalize RGB values to [0, 1]
    rgb = img.astype(np.float32) / 255.0
    
    # Reshape to handle each pixel as a separate item
    original_shape = rgb.shape
    rgb_reshaped = rgb.reshape(-1, 3)
    
    # Extract RGB components
    r, g, b = rgb_reshaped[:, 0], rgb_reshaped[:, 1], rgb_reshaped[:, 2]
    
    # Compute Value (V)
    v = np.max(rgb_reshaped, axis=1)
    
    # Compute Saturation (S)
    delta = v - np.min(rgb_reshaped, axis=1)
    s = np.zeros_like(v)
    non_zero_v = v > 0
    s[non_zero_v] = delta[non_zero_v] / v[non_zero_v]
    
    # Compute Hue (H)
    h = np.zeros_like(v)
    
    # Where delta is not zero (to avoid division by zero)
    non_zero_delta = delta > 0
    
    # r is max
    r_max = np.logical_and(non_zero_delta, r == v)
    h[r_max] = 60.0 * ((g[r_max] - b[r_max]) / delta[r_max]) % 6
    
    # g is max
    g_max = np.logical_and(non_zero_delta, g == v)
    h[g_max] = 60.0 * ((b[g_max] - r[g_max]) / delta[g_max] + 2)
    
    # b is max
    b_max = np.logical_and(non_zero_delta, b == v)
    h[b_max] = 60.0 * ((r[b_max] - g[b_max]) / delta[b_max] + 4)
    
    # Reshape back to original image dimensions
    hsv = np.stack([h, s, v], axis=1).reshape(original_shape)
    
    return hsv

def create_preprocessing_preview(original_img, cropped_img, hsv_img, filename):
    """
    Create and save a side-by-side preview of preprocessing steps.
    
    Args:
        original_img (numpy.ndarray): Original RGB image
        cropped_img (numpy.ndarray): Cropped RGB image
        hsv_img (numpy.ndarray): HSV image
        filename (str): Base filename for saving
        
    Returns:
        str: Path to the saved preview image
    """
    from config import PREVIEW_DIR, get_timestamp
    
    # Extract saturation channel and black spots mask
    saturation = hsv_img[:, :, 1]  # S channel
    black_spots = hsv_img[:, :, 2] < 0.2  # V < 0.2
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Cropped image
    axes[0, 1].imshow(cropped_img)
    axes[0, 1].set_title("Cropped Image")
    axes[0, 1].axis('off')
    
    # HSV Saturation channel
    axes[1, 0].imshow(saturation, cmap='gray')
    axes[1, 0].set_title("Saturation Channel")
    axes[1, 0].axis('off')
    
    # Black spots mask
    axes[1, 1].imshow(black_spots, cmap='binary')
    axes[1, 1].set_title("Black Spots (V < 0.2)")
    axes[1, 1].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory and save
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    output_path = os.path.join(PREVIEW_DIR, f"img_{os.path.basename(filename).split('.')[0]}_preview.png")
    print(f"Saving preview to: {output_path}")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def preprocess_image(image_path, save_preview=False):
    """
    Perform full preprocessing pipeline on a single image.
    
    Args:
        image_path (str): Path to the image
        save_preview (bool): Whether to save preprocessing preview
        
    Returns:
        tuple: (hsv_image, cropped_image)
    """
    # Load image
    original_img = load_image(image_path)
    if original_img is None:
        return None, None
    
    # Crop background
    cropped_img = crop_background(original_img)
    
    # Convert RGB to HSV
    hsv_img = rgb_to_hsv(cropped_img)
    
    # Optionally save preview
    if save_preview:
        preview_path = create_preprocessing_preview(
            original_img, cropped_img, hsv_img, 
            os.path.basename(image_path)
        )
        print(f"Saved preprocessing preview to: {preview_path}")
    
    return hsv_img, cropped_img
