#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyfiglet import Figlet
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, run_command, run_progress_spinner, extract_source_category,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)
from utils.plot_images import plot_image_set


def parse_arguments():
    parser = argparse.ArgumentParser(description='Apply transformations to images or directories of images')
    parser.add_argument('path', type=str, help='Path to the image or directory')
    parser.add_argument('--dest', type=str, default='transformed_images', 
                        help='Destination directory for transformed images')
    
    # Add transformation options
    parser.add_argument('--grayscale', action='store_true', help='Convert to grayscale')
    parser.add_argument('--edges', action='store_true', help='Edge detection')
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian blur')
    parser.add_argument('--sharpen', action='store_true', help='Sharpen the image')
    parser.add_argument('--binary', action='store_true', help='Convert to binary (black and white)')
    parser.add_argument('--contrast', action='store_true', help='Enhance contrast')
    # Add new transformation options
    parser.add_argument('--mask', action='store_true', help='Create leaf mask')
    parser.add_argument('--roi', action='store_true', help='Identify ROI objects')
    parser.add_argument('--analyze', action='store_true', help='Analyze leaf objects')
    parser.add_argument('--landmarks', action='store_true', help='Generate pseudolandmarks')
    parser.add_argument('--all', action='store_true', help='Apply all transformations')
    
    args = parser.parse_args()
    
    # If no transformations are specified or --all is used, enable all transformations
    if not any([args.grayscale, args.edges, args.blur, args.sharpen, 
                args.binary, args.contrast, args.mask, args.roi,
                args.analyze, args.landmarks]) or args.all:
        options = {
            'grayscale': True,
            'edges': True,
            'blur': True,
            'sharpen': True,
            'binary': True,
            'contrast': True,
            'mask': True,
            'roi': True,
            'analyze': True,
            'landmarks': True
        }
    else:
        options = {
            'grayscale': args.grayscale,
            'edges': args.edges,
            'blur': args.blur,
            'sharpen': args.sharpen,
            'binary': args.binary,
            'contrast': args.contrast,
            'mask': args.mask,
            'roi': args.roi,
            'analyze': args.analyze,
            'landmarks': args.landmarks
        }
    
    return args.path, args.dest, options

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Keep the image in BGR format (OpenCV default)
    return img

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    If the image is already grayscale, return it as is.
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection.
    """
    # Convert to grayscale if needed
    gray = convert_to_grayscale(image)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Convert back to color format for consistency
    if len(image.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges

def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Sharpen an image using unsharp masking.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # Subtract blurred image from original
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    return sharpened

def convert_to_binary(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to binary (black and white).
    """
    # Convert to grayscale if needed
    gray = convert_to_grayscale(image)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to color format for consistency
    if len(image.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return binary

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using histogram equalization.
    """
    # If image is color, convert to YUV and equalize only the Y channel
    if len(image.shape) == 3:
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Equalize the Y channel
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # If image is grayscale, directly apply equalization
        return cv2.equalizeHist(image)

def generate_roi_objects(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Generate ROI (Region of Interest) objects on the leaf.
    """
    # Create output image
    roi_image = image.copy()
    
    # Create a green overlay
    overlay = np.zeros_like(roi_image)
    overlay[mask > 0] = [0, 255, 0]  # Green color for the overlay
    
    # Apply overlay with transparency
    cv2.addWeighted(overlay, 0.5, roi_image, 1, 0, roi_image)
    
    # Draw a blue border around the image
    roi_image = cv2.copyMakeBorder(roi_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 255])
    
    return roi_image

def analyze_leaf_object(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Analyze the leaf object with shape detection.
    """
    # Create output image
    analyzed_image = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw contour outline in blue
        cv2.drawContours(analyzed_image, [largest_contour], 0, (0, 0, 255), 2)
        
        # Find the minimum area enclosing ellipse
        if len(largest_contour) >= 5:  # Need at least 5 points for ellipse
            ellipse = cv2.fitEllipse(largest_contour)
            cv2.ellipse(analyzed_image, ellipse, (255, 0, 255), 2)
        
        # Find veins using edge detection on the leaf area
        leaf_area = cv2.bitwise_and(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
        veins = cv2.Canny(leaf_area, 100, 200)
        veins_dilated = cv2.dilate(veins, np.ones((3, 3), np.uint8), iterations=1)
        
        # Draw veins in blue
        analyzed_image[veins_dilated > 0] = [0, 0, 255]
    
    return analyzed_image

def generate_pseudolandmarks(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Generate pseudolandmarks on the leaf.
    """
    # Create output image
    landmark_image = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw contour outline in magenta
        cv2.drawContours(landmark_image, [largest_contour], 0, (255, 0, 255), 2)
        
        # Generate boundary landmarks (30 points around the contour)
        boundary_landmarks = []
        for i in range(0, len(largest_contour), len(largest_contour) // 30):
            if len(boundary_landmarks) < 30:  # Limit to 30 points
                point = tuple(largest_contour[i][0])
                boundary_landmarks.append(point)
                cv2.circle(landmark_image, point, 5, (255, 0, 255), -1)  # Magenta circles
        
        # Find the center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
            
            # Draw center point
            cv2.circle(landmark_image, center, 8, (255, 0, 255), -1)
            
            # Find disease spots (simplified approach - looking for darker/browner regions)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_brown = np.array([0, 50, 50])
            upper_brown = np.array([30, 255, 150])
            disease_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Apply mask to only look within the leaf area
            disease_mask = cv2.bitwise_and(disease_mask, mask)
            
            # Find disease contours
            disease_contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw disease landmarks in orange
            for contour in disease_contours:
                if cv2.contourArea(contour) > 20:  # Filter small noise
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        dcx = int(M["m10"] / M["m00"])
                        dcy = int(M["m01"] / M["m00"])
                        cv2.circle(landmark_image, (dcx, dcy), 5, (255, 100, 0), -1)  # Orange circles
    
    return landmark_image

def create_leaf_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask of the leaf area.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find the largest contour (assumed to be the leaf)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros_like(binary)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fill the largest contour
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def save_transformed_images(transformed_images, original_image_path, output_dir):
    """
    Save all transformed images to the output directory.
    
    Args:
        transformed_images: Dictionary with transformation names as keys and images as values
        original_image_path: Path to the original image
        output_dir: Directory to save transformed images
    """
    # Get the filename without extension
    basename = os.path.basename(original_image_path)
    name, ext = os.path.splitext(basename)
    
    # Extract source directory information with improved function
    source_dir = extract_source_category(original_image_path)
    
    print(f"{GREEN}Saving transformed images to: {output_dir}{NC}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip the original image when saving
    for transform_name, image in transformed_images.items():
        if transform_name == "Original":
            continue
            
        # Construct the new filename with the transformation type and source info
        new_filename = os.path.join(output_dir, f"{name}_{source_dir}_{transform_name.lower()}{ext}")
        
        # Save the image
        success = cv2.imwrite(new_filename, image)
        if success:
            print(f"Saved: {os.path.basename(new_filename)}")
        else:
            print(f"{RED}Failed to save:{NC} {os.path.basename(new_filename)}")


def main():
    # Parse arguments
    image_path, output_dir, options = parse_arguments()
    
    try:
        # Load the original image
        original_image = load_image(image_path)
        
        # Dictionary to store all transformed images
        transformed_images = {"Original": original_image}
        
        # Create a mask for advanced transformations
        mask = None
        if options['mask'] or options['roi'] or options['analyze'] or options['landmarks']:
            mask = create_leaf_mask(original_image)
            # Convert mask to BGR for visualization
            mask_visual = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            transformed_images["Mask"] = mask_visual
        
        # Apply transformations based on options
        if options['grayscale']:
            gray = convert_to_grayscale(original_image)
            # Convert back to BGR for consistent processing
            if len(gray.shape) == 2:
                gray_visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                gray_visual = gray
            transformed_images["Grayscale"] = gray_visual
        
        if options['edges']:
            edges = apply_edge_detection(original_image)
            transformed_images["Edges"] = edges
        
        if options['blur']:
            blurred = apply_gaussian_blur(original_image)
            transformed_images["Blur"] = blurred
        
        if options['sharpen']:
            sharpened = sharpen_image(original_image)
            transformed_images["Sharpen"] = sharpened
        
        if options['binary']:
            binary = convert_to_binary(original_image)
            transformed_images["Binary"] = binary
        
        if options['contrast']:
            contrast = enhance_contrast(original_image)
            transformed_images["Contrast"] = contrast
        
        if options['roi'] and mask is not None:
            roi = generate_roi_objects(original_image, mask)
            transformed_images["ROI"] = roi
        
        if options['analyze'] and mask is not None:
            analyzed = analyze_leaf_object(original_image, mask)
            transformed_images["Analyzed"] = analyzed
        
        if options['landmarks'] and mask is not None:
            landmarks = generate_pseudolandmarks(original_image, mask)
            transformed_images["Landmarks"] = landmarks
        
        # Save transformed images
        save_transformed_images(transformed_images, image_path, output_dir)
        
        # Extract source directory information for title
        from utils.utils import extract_source_category
        source_dir = extract_source_category(image_path)
        
        # Get base name without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create custom title with directory path format, without the "from repo" part
        custom_title = f"Transformation: {base_filename} (./{source_dir})"
        
        # Generate plot
        plot_filename = plot_image_set(
            transformed_images,
            image_path,
            title_prefix="transformation_",
            max_cols=3,
            custom_title=custom_title
        )
        
        print(f"✅ Successfully transformed image: {os.path.basename(image_path)}")
        print(f"✅ Plot saved to: {plot_filename}")
        
        return 0
    
    except Exception as e:
        print(f"{RED}Error:{NC} {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())