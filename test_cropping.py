#!/usr/bin/env python3
"""
Test script for the improved cropping algorithm.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import load_image, crop_background, rgb_to_hsv, create_preprocessing_preview

def test_all_crop_methods(image_path):
    """Test all cropping methods on a single image and show comparison"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    # Load the image
    print(f"Loading image from: {image_path}")
    original_img = load_image(image_path)
    
    if original_img is None:
        print("Error: Failed to load image")
        return
    
    # Apply different cropping methods and parameters
    test_configs = [
        ('threshold', 220, 15, False, False),  # Simple threshold, fixed padding
        ('otsu', 0, 15, False, True),          # Otsu method with denoising
        ('hsv', 0, 15, False, False),          # Standard HSV method
        ('hsv', 0, 15, True, True),            # HSV with adaptive padding and denoising
        ('hsv', 0, 30, True, True),            # HSV with more padding
    ]
    
    cropped_images = {}
    
    for config in test_configs:
        method, threshold, padding, adaptive, denoise = config
        config_name = f"{method}-pad{padding}{'-adapt' if adaptive else ''}{'-denoise' if denoise else ''}"
        print(f"Testing configuration: {config_name}")
        
        cropped = crop_background(
            original_img, 
            method=method, 
            threshold=threshold,
            padding=padding,
            adaptive_padding=adaptive,
            denoise=denoise
        )
        hsv = rgb_to_hsv(cropped)
        cropped_images[config_name] = (cropped, hsv, config)
    
    # Create a figure to compare results - adjust rows based on number of configs
    num_methods = len(test_configs)
    fig, axes = plt.subplots(num_methods + 1, 2, figsize=(14, 4 * (num_methods + 1)))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')  # Empty cell
    
    # Show results for each configuration
    for i, (config_name, (cropped, hsv, config)) in enumerate(cropped_images.items()):
        method, threshold, padding, adaptive, denoise = config
        
        # Calculate feature values for display
        saturation = hsv[:, :, 1]
        black_spots = hsv[:, :, 2] < 0.2
        mean_sat = np.mean(saturation)
        black_ratio = np.sum(black_spots) / black_spots.size
        
        # Display cropped image
        axes[i+1, 0].imshow(cropped)
        axes[i+1, 0].set_title(f"Method: {config_name}")
        axes[i+1, 0].axis('off')
        
        # Display black spot mask
        axes[i+1, 1].imshow(black_spots, cmap='binary')
        axes[i+1, 1].set_title(f"Black Spots - Ratio: {black_ratio:.3f}, Sat: {mean_sat:.3f}")
        axes[i+1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    output_dir = "visuals/previews"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cropping_comparison_{os.path.basename(image_path).split('.')[0]}.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    
    # Also create detailed preview for the recommended method (HSV with adaptive padding and denoising)
    # This is our recommended config
    cropped, hsv, config = cropped_images['hsv-pad15-adapt-denoise']
    method, threshold, padding, adaptive, denoise = config
    
    preview_path = create_preprocessing_preview(
        original_img, cropped, hsv,
        os.path.basename(image_path),
        show_masks=True,
        crop_method=method,
        crop_info={'method': method, 'threshold': threshold, 'padding': padding,
                  'adaptive_padding': adaptive, 'denoise': denoise}
    )
    print(f"Saved detailed preview to: {preview_path}")
    
    return output_path, preview_path

if __name__ == "__main__":
    # If image path is provided as argument, use it
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Otherwise look for an image in the test dataset
        dataset_path = "dataset/apple_test/apple scab"
        if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
            image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                image_path = os.path.join(dataset_path, image_files[0])
                print(f"Using test image: {image_path}")
            else:
                print("No test images found in the test dataset.")
                sys.exit(1)
        else:
            print("Test dataset not found. Please provide an image path as argument.")
            sys.exit(1)
    
    test_all_crop_methods(image_path)
