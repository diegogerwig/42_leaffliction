#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, GREEN, BLUE, RED, YELLOW, CYAN, NC
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize augmented images')
    parser.add_argument('original_image', type=str, help='Path to the original image')
    parser.add_argument('augmented_dir', type=str, help='Directory containing augmented images')
    parser.add_argument('--samples', type=int, default=5, help='Number of augmented samples to display')
    
    return parser.parse_args()

def find_augmented_images(original_image_path, augmented_dir):
    """Find augmented versions of the original image."""
    # Get the base name of the original image without extension
    original_basename = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Find all augmented images in the augmented directory
    augmented_images = []
    for root, _, files in os.walk(augmented_dir):
        for file in files:
            if file.startswith(original_basename) and "_aug_" in file:
                augmented_images.append(os.path.join(root, file))
    
    return augmented_images

def display_augmentations(original_image_path, augmented_images, num_samples=5):
    """Display the original image and its augmented versions."""
    # Read the original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Could not read original image: {original_image_path}")
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # If there are more augmented images than requested samples, randomly select some
    if len(augmented_images) > num_samples:
        augmented_images = random.sample(augmented_images, num_samples)
    
    # Read augmented images
    augmented_rgb_images = []
    for aug_path in augmented_images:
        aug_image = cv2.imread(aug_path)
        if aug_image is not None:
            # Convert BGR to RGB for display
            aug_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            augmented_rgb_images.append((aug_rgb, os.path.basename(aug_path)))
    
    # Create a figure for display
    total_images = 1 + len(augmented_rgb_images)
    num_cols = min(3, total_images)
    num_rows = (total_images + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 5 * num_rows))
    
    # Display original image
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display augmented images
    for i, (aug_image, aug_name) in enumerate(augmented_rgb_images):
        plt.subplot(num_rows, num_cols, i + 2)
        plt.imshow(aug_image)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return total_images

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Find augmented images
        print_colored(f"Looking for augmented versions of {args.original_image}...", BLUE)
        augmented_images = find_augmented_images(args.original_image, args.augmented_dir)
        
        if not augmented_images:
            print_colored(f"No augmented images found for {args.original_image}", YELLOW)
            return 1
        
        print_colored(f"Found {len(augmented_images)} augmented images", GREEN)
        
        # Display augmentations
        num_displayed = display_augmentations(
            args.original_image, 
            augmented_images, 
            args.samples
        )
        
        print_colored(f"Displayed {num_displayed} images (1 original + {num_displayed-1} augmented)", GREEN)
        
        return 0
    
    except Exception as e:
        print_colored(f"Error: {e}", RED)
        return 1

if __name__ == "__main__":
    sys.exit(main())