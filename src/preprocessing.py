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

def crop_background(img, padding=10, method='threshold', threshold=BACKGROUND_THRESHOLD, 
                 min_leaf_area=0.01, adaptive_padding=True, denoise=True):
    """
    Remove background from image and crop to bounding box with improved methods.
    
    Args:
        img (numpy.ndarray): RGB image as NumPy array
        padding (int): Number of pixels to pad around the leaf (default: 10)
        method (str): Method for background removal ('threshold', 'otsu', 'hsv')
        threshold (int): Threshold value for background removal (default: from config)
        min_leaf_area (float): Minimum ratio of image area that must be leaf (default: 0.01)
        adaptive_padding (bool): Automatically adjust padding based on image size (default: True)
        denoise (bool): Apply noise reduction to mask (default: True)
        
    Returns:
        numpy.ndarray: Cropped RGB image
    """
    height, width = img.shape[:2]
    
    # Calculate adaptive padding if enabled (roughly 2% of image dimension)
    if adaptive_padding:
        dimension = min(height, width)
        padding = max(padding, int(dimension * 0.02))
    
    # Choose background removal method
    if method == 'threshold':
        # Simple threshold-based masking
        mask = ~np.all(img > threshold, axis=2)
    elif method == 'otsu':
        # Convert to grayscale and use Otsu's method
        gray = np.mean(img, axis=2).astype(np.uint8)
        # Calculate histogram
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        # Normalize histogram
        hist_norm = hist / hist.sum()
        # Calculate cumulative sums
        cumsum = np.cumsum(hist_norm)
        # Calculate means
        i_indices = np.arange(256)
        mean_global = np.sum(i_indices * hist_norm)
        var_between = np.zeros(256)
        for t in range(1, 256):
            # Background
            w_bg = cumsum[t-1]
            if w_bg > 0:
                mean_bg = np.sum(i_indices[:t] * hist_norm[:t]) / w_bg
            else:
                mean_bg = 0
            # Foreground
            w_fg = 1 - w_bg
            if w_fg > 0:
                mean_fg = (mean_global - w_bg * mean_bg) / w_fg
            else:
                mean_fg = 0
            # Between-class variance
            var_between[t] = w_bg * w_fg * ((mean_bg - mean_fg) ** 2)
        # Find threshold
        otsu_threshold = np.argmax(var_between)
        # Apply threshold to create mask
        mask = gray < otsu_threshold
    elif method == 'hsv':
        # Convert to HSV to better separate leaf from background
        hsv = rgb_to_hsv(img)
        # Use combination of saturation and value channels for better leaf detection
        # Green leaves typically have high saturation and moderate value
        hue = hsv[:,:,0]
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Create mask with saturation as primary indicator
        sat_mask = saturation > 0.15  # Base saturation threshold
        
        # Additional check for green hue range (common for leaves)
        # Hue is in degrees (0-360), green is roughly 60-180
        green_mask = (hue >= 60) & (hue <= 180)
        
        # Combine masks with adaptive weighting based on image statistics
        # If the image has overall low saturation, give more weight to hue
        if np.mean(saturation) < 0.2:
            mask = sat_mask | (green_mask & (value > 0.2))
        else:
            mask = sat_mask
            
    else:
        # Default to simple thresholding
        mask = ~np.all(img > threshold, axis=2)
    
    # Apply denoising if requested
    if denoise:
        # Simple noise reduction by removing small isolated pixels
        # First dilate (expand) then erode (contract) - this is "closing"
        kernel_size = max(3, min(height, width) // 100)  # Adaptive kernel size
        
        # Simple morphology operations without requiring OpenCV
        # Dilation - expand
        dilated = mask.copy()
        for _ in range(2):  # Dilate twice
            temp = dilated.copy()
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if np.any(temp[i-1:i+2, j-1:j+2]):
                        dilated[i, j] = True
                        
        # Erosion - contract
        eroded = dilated.copy()
        for _ in range(2):  # Erode twice
            temp = eroded.copy()
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if not np.all(temp[i-1:i+2, j-1:j+2]):
                        eroded[i, j] = False
                        
        mask = eroded
    
    # Find non-zero indices (row, column) of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Check if any foreground pixels found
    if not np.any(rows) or not np.any(cols):
        print("Warning: No foreground pixels found in image. Using entire image.")
        return img
    
    # Check if detected leaf area is too small (might be noise)
    leaf_area = np.sum(mask) / mask.size
    if leaf_area < min_leaf_area:
        print(f"Warning: Detected leaf area too small ({leaf_area:.3f} < {min_leaf_area}). Using entire image.")
        return img
        
    # Find min/max row and column to form bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add padding (with boundary checking)
    rmin = max(0, rmin - padding)
    rmax = min(height - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(width - 1, cmax + padding)
    
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

def create_preprocessing_preview(original_img, cropped_img, hsv_img, filename, show_masks=True, 
                                  crop_method='hsv', crop_info=None):
    """
    Create and save a side-by-side preview of preprocessing steps with enhanced visualization.
    
    Args:
        original_img (numpy.ndarray): Original RGB image
        cropped_img (numpy.ndarray): Cropped RGB image
        hsv_img (numpy.ndarray): HSV image
        filename (str): Base filename for saving
        show_masks (bool): Whether to show detailed masks for feature extraction
        crop_method (str): Method used for cropping ('threshold', 'otsu', 'hsv')
        crop_info (dict): Optional additional cropping information
        
    Returns:
        str: Path to the saved preview image
    """
    from config import PREVIEW_DIR, get_timestamp, VALUE_THRESHOLD
    
    # Extract saturation channel and black spots mask
    saturation = hsv_img[:, :, 1]  # S channel
    black_spots = hsv_img[:, :, 2] < VALUE_THRESHOLD  # V channel threshold
    
    # Calculate feature values for display
    mean_sat = np.mean(saturation)
    black_ratio = np.sum(black_spots) / black_spots.size
    
    if show_masks:
        # Create figure with 2x3 subplots for more detailed view
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Generate segmentation mask for visualization
        if crop_method == 'threshold':
            threshold = 220 if crop_info is None else crop_info.get('threshold', 220)
            mask = ~np.all(original_img > threshold, axis=2)
            mask_title = f"Segmentation Mask (RGB thresh={threshold})"
        elif crop_method == 'otsu':
            gray = np.mean(original_img, axis=2).astype(np.uint8)
            hist = np.histogram(gray, bins=256, range=(0, 255))[0]
            hist_norm = hist / hist.sum()
            cumsum = np.cumsum(hist_norm)
            i_indices = np.arange(256)
            mean_global = np.sum(i_indices * hist_norm)
            var_between = np.zeros(256)
            for t in range(1, 256):
                w_bg = cumsum[t-1]
                if w_bg > 0:
                    mean_bg = np.sum(i_indices[:t] * hist_norm[:t]) / w_bg
                else:
                    mean_bg = 0
                w_fg = 1 - w_bg
                if w_fg > 0:
                    mean_fg = (mean_global - w_bg * mean_bg) / w_fg
                else:
                    mean_fg = 0
                var_between[t] = w_bg * w_fg * ((mean_bg - mean_fg) ** 2)
            otsu_threshold = np.argmax(var_between)
            mask = gray < otsu_threshold
            mask_title = f"Segmentation Mask (Otsu thresh={otsu_threshold})"
        elif crop_method == 'hsv':
            orig_hsv = rgb_to_hsv(original_img)
            mask = orig_hsv[:,:,1] > 0.15
            mask_title = "Segmentation Mask (HSV saturation > 0.15)"
        else:
            # Default simple thresholding
            mask = ~np.all(original_img > 220, axis=2)
            mask_title = "Segmentation Mask (default)"
        
        # Calculate segmentation stats
        segment_ratio = np.sum(mask) / mask.size
        
        # Create composite visualization showing segmentation
        axes[0, 1].imshow(original_img)
        overlay_mask = np.zeros_like(original_img)
        overlay_mask[:, :, 1] = mask * 150  # Green overlay for segmented area
        axes[0, 1].imshow(overlay_mask, alpha=0.5)
        axes[0, 1].set_title(f"Segmentation\nLeaf area: {segment_ratio:.1%}")
        axes[0, 1].axis('off')
        
        # Cropped image
        axes[0, 2].imshow(cropped_img)
        axes[0, 2].set_title("Cropped Image")
        axes[0, 2].axis('off')
                
        # HSV visualization (using Hue for colorization)
        hsv_vis = hsv_img.copy()
        hsv_vis[:,:,2] = np.clip(hsv_vis[:,:,2]*1.5, 0, 1)  # Enhance brightness for better vis
        rgb_vis = np.zeros_like(cropped_img)
        
        # Simple HSV to RGB conversion for visualization
        h, s, v = hsv_vis[:,:,0], hsv_vis[:,:,1], hsv_vis[:,:,2]
        c = v * s
        x = c * (1 - np.abs(((h / 60) % 2) - 1))
        m = v - c
        
        # RGB components based on hue
        idx = (h < 60)
        rgb_vis[idx] = np.column_stack([c[idx], x[idx], np.zeros_like(c[idx])]) * 255
        
        idx = (h >= 60) & (h < 120)
        rgb_vis[idx] = np.column_stack([x[idx], c[idx], np.zeros_like(c[idx])]) * 255
        
        idx = (h >= 120) & (h < 180)
        rgb_vis[idx] = np.column_stack([np.zeros_like(c[idx]), c[idx], x[idx]]) * 255
        
        idx = (h >= 180) & (h < 240)
        rgb_vis[idx] = np.column_stack([np.zeros_like(c[idx]), x[idx], c[idx]]) * 255
        
        idx = (h >= 240) & (h < 300)
        rgb_vis[idx] = np.column_stack([x[idx], np.zeros_like(c[idx]), c[idx]]) * 255
        
        idx = (h >= 300)
        rgb_vis[idx] = np.column_stack([c[idx], np.zeros_like(c[idx]), x[idx]]) * 255
        
        rgb_vis = (rgb_vis + m[:,:,np.newaxis] * 255).astype(np.uint8)
        
        # Saturation channel
        axes[1, 0].imshow(saturation, cmap='hot')
        axes[1, 0].set_title(f"Saturation Channel\nMean: {mean_sat:.3f}")
        axes[1, 0].axis('off')
        
        # Value channel 
        axes[1, 1].imshow(hsv_img[:,:,2], cmap='gray')
        axes[1, 1].set_title("Value Channel")
        axes[1, 1].axis('off')
        
        # Black spots mask
        axes[1, 2].imshow(black_spots, cmap='binary')
        axes[1, 2].set_title(f"Black Spots (V < {VALUE_THRESHOLD})\nRatio: {black_ratio:.3f}")
        axes[1, 2].axis('off')
    else:
        # Create figure with 2x2 subplots (original layout)
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
        axes[1, 0].imshow(saturation, cmap='hot')
        axes[1, 0].set_title(f"Saturation Channel\nMean: {mean_sat:.3f}")
        axes[1, 0].axis('off')
        
        # Black spots mask
        axes[1, 1].imshow(black_spots, cmap='binary')
        axes[1, 1].set_title(f"Black Spots (V < {VALUE_THRESHOLD})\nRatio: {black_ratio:.3f}")
        axes[1, 1].axis('off')
    
    # Add a title with filename and cropping method
    plt.suptitle(f"Image Processing Preview: {os.path.basename(filename)} (Method: {crop_method})", fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Create output directory and save
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    output_path = os.path.join(PREVIEW_DIR, f"img_{os.path.basename(filename).split('.')[0]}_preview.png")
    print(f"Saving preview to: {output_path}")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def preprocess_image(image_path, save_preview=False, crop_method='hsv', padding=15,
                  adaptive_padding=True, denoise=True, min_leaf_area=0.01):
    """
    Perform full preprocessing pipeline on a single image.
    
    Args:
        image_path (str): Path to the image
        save_preview (bool): Whether to save preprocessing preview
        crop_method (str): Method for background removal ('threshold', 'otsu', 'hsv')
        padding (int): Base number of pixels to pad around the leaf
        adaptive_padding (bool): Automatically adjust padding based on image size
        denoise (bool): Apply noise reduction to segmentation mask
        min_leaf_area (float): Minimum ratio of image area that must be leaf
        
    Returns:
        tuple: (hsv_image, cropped_image)
    """
    # Load image
    original_img = load_image(image_path)
    if original_img is None:
        return None, None
    
    # Get threshold from config
    from config import BACKGROUND_THRESHOLD
    
    # Store cropping parameters for preview
    crop_info = {
        'method': crop_method,
        'padding': padding,
        'threshold': BACKGROUND_THRESHOLD,
        'adaptive_padding': adaptive_padding,
        'denoise': denoise
    }
    
    # Crop background with improved method
    cropped_img = crop_background(
        original_img, 
        padding=padding,
        method=crop_method,
        threshold=BACKGROUND_THRESHOLD,
        adaptive_padding=adaptive_padding,
        denoise=denoise,
        min_leaf_area=min_leaf_area
    )
    
    # Convert RGB to HSV
    hsv_img = rgb_to_hsv(cropped_img)
    
    # Optionally save preview
    if save_preview:
        preview_path = create_preprocessing_preview(
            original_img, cropped_img, hsv_img, 
            os.path.basename(image_path),
            show_masks=True,
            crop_method=crop_method,
            crop_info=crop_info
        )
        print(f"Saved preprocessing preview to: {preview_path}")
    
    return hsv_img, cropped_img
