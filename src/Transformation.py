#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyfiglet import Figlet
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


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
    
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_leaf_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask of the leaf area.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
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
        leaf_area = cv2.bitwise_and(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask)
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
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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

def apply_transformations(image: np.ndarray, options: Dict[str, bool]) -> Dict[str, np.ndarray]:
    transformations = {}
    
    # Original image
    transformations['original'] = image
    
    # Grayscale
    if options.get('grayscale', False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        transformations['grayscale'] = gray
    
    # Edge detection (Canny)
    if options.get('edges', False):
        if 'grayscale' in transformations:
            gray = transformations['grayscale']
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Automatic edge detection
        sigma = 0.33
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # Convert to RGB for display consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        transformations['edges'] = edges_rgb
    
    # Gaussian blur
    if options.get('blur', False):
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        transformations['blur'] = blurred
    
    # Sharpen
    if options.get('sharpen', False):
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        transformations['sharpen'] = sharpened
    
    # Binary (black and white)
    if options.get('binary', False):
        if 'grayscale' in transformations:
            gray = transformations['grayscale']
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        transformations['binary'] = binary_rgb
    
    # Contrast enhancement
    if options.get('contrast', False):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        transformations['contrast'] = enhanced
    
    # Create leaf mask (required for advanced transformations)
    mask = None
    if options.get('mask', False) or options.get('roi', False) or options.get('analyze', False) or options.get('landmarks', False):
        mask = create_leaf_mask(image)
    
    # Mask transformation
    if options.get('mask', False) and mask is not None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        transformations['mask'] = mask_rgb
    
    # ROI objects
    if options.get('roi', False) and mask is not None:
        roi_image = generate_roi_objects(image, mask)
        transformations['roi'] = roi_image
    
    # Analyze leaf
    if options.get('analyze', False) and mask is not None:
        analyzed_image = analyze_leaf_object(image, mask)
        transformations['analyze'] = analyzed_image
    
    # Pseudolandmarks
    if options.get('landmarks', False) and mask is not None:
        landmark_image = generate_pseudolandmarks(image, mask)
        transformations['landmarks'] = landmark_image
    
    return transformations

def display_transformations(transformations: Dict[str, np.ndarray], title: str = "Image Transformations") -> None:
    """
    This function is kept for compatibility but not used in the main workflow.
    It displays all transformations in a matplotlib figure.
    """
    n = len(transformations)
    cols = 3
    rows = (n + cols - 1) // cols  # Ceiling division
    
    plt.figure(figsize=(15, 4 * rows))
    plt.suptitle(title, fontsize=16)
    
    for i, (name, img) in enumerate(transformations.items()):
        plt.subplot(rows, cols, i + 1)
        
        # Handle grayscale images
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
            
        # Format title as "Figure IV.N: Name"
        plt.title(f"Figure IV.{i+1}: {name.replace('_', ' ').capitalize()}")
        plt.axis('on')  # Show axes for better visualization
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for title
    plt.show()

def save_transformations(transformations: Dict[str, np.ndarray], output_dir: str, filename: str) -> None:
    """
    Save transformed images to the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print output directory similar to augmentation.py
    print(f"Output directory: {output_dir}")
    print(f"Saving transformed images to: {output_dir}")
    
    for name, img in transformations.items():
        # Convert back to BGR for saving (OpenCV uses BGR)
        if name != 'grayscale' and len(img.shape) == 3:
            img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img
        
        # Always use lowercase .jpg extension
        output_path = os.path.join(output_dir, f"{filename}_{name}.jpg")
        cv2.imwrite(output_path, img_to_save)
        print(f"Saved: {filename}_{name}.jpg")

def plot_transformations(transformations: Dict[str, np.ndarray], image_path: str) -> str:
    """
    Generate a plot with all transformations and save it.
    Similar to plot_images from Augmentation.py but with added axes and scales.
    """
    # Convert from BGR to RGB for display with matplotlib
    rgb_images = {}
    for name, img in transformations.items():
        if len(img.shape) == 2:  # Grayscale image
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # Already RGB
            rgb_img = img
        rgb_images[name] = rgb_img
    
    # Calculate rows and columns dynamically
    n = len(transformations)
    cols = 3
    rows = (n + cols - 1) // cols  # Ceiling division
    
    # Create a figure of appropriate size - with more padding at the top for title
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Handle the case of a single row or column
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1)
    
    # Fill the subplots with images and add scales
    for i, (name, img) in enumerate(rgb_images.items()):
        row, col = divmod(i, cols)
        if rows > 1 and cols > 1:
            ax = axes[row, col]
        else:
            ax = axes[i]
        
        # Display image with scales
        im = ax.imshow(img)
        
        # Add title for each subplot but move it slightly higher
        ax.set_title(name.capitalize(), fontsize=12, pad=10)
        
        # Show axes with scales (like in Image 1)
        ax.axis('on')
        
        # Add grid for better visibility of scales
        ax.grid(False)  # Disable grid for cleaner look
        
        # Set ticks for better scale reading
        height, width = img.shape[:2]
        # Set x-ticks at every 50 pixels
        x_ticks = np.arange(0, width, 50)
        ax.set_xticks(x_ticks)
        
        # Set y-ticks at every 50 pixels
        y_ticks = np.arange(0, height, 50)
        ax.set_yticks(y_ticks)
    
    # Hide any empty subplots
    for i in range(len(transformations), rows * cols):
        row, col = divmod(i, cols)
        if rows > 1 and cols > 1:
            axes[row, col].axis('off')
        elif i < len(axes):
            axes[i].axis('off')
    
    # Add a main title with more spacing to avoid overlap with subplot titles
    plt.suptitle(f'Transformations: {os.path.basename(image_path)}', 
                fontsize=16, 
                y=0.98)  # Position title higher to avoid overlap
    
    # Adjust layout with more top padding
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Increase top margin to avoid overlap
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.abspath("./plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate filename for the plot
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    plot_filename = os.path.join(plots_dir, f"transformation_{base_name}.png")
    
    # Save the figure
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_filename}")
    
    # Display the plot
    plt.show()
    
    return plot_filename

def transform_image(image_path: str, output_dir: str, options: Dict[str, bool]) -> None:
    """
    Apply transformations to a single image and generate a plot.
    """
    try:
        # Get the base name of the image (preserving original name)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create a subdirectory for this image - fix the directory structure
        img_output_dir = os.path.join(output_dir, filename)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Load the image and apply transformations
        image = load_image(image_path)
        transformations = apply_transformations(image, options)
        
        # Save the transformed images directly to the image's subdirectory
        save_transformations(transformations, img_output_dir, filename)
        
        # Generate and save the plot with all transformations
        plot_filename = plot_transformations(transformations, image_path)
        
        print(f"{GREEN}Processed:{NC} {os.path.basename(image_path)}")
        print(f"All transformations saved to: {img_output_dir}")
        
        print(f"✅ Processing completed successfully for {os.path.basename(image_path)}")
        print(f"Results saved to: {img_output_dir}")
    except Exception as e:
        print(f"{RED}Error processing {image_path}:{NC} {e}")

def transform_directory(directory_path: str, output_dir: str, options: Dict[str, bool]) -> None:
    # Get list of image files in the directory
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(list(Path(directory_path).glob(f"*{ext}")))
        image_files.extend(list(Path(directory_path).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"{YELLOW}No image files found in {directory_path}{NC}")
        return
    
    print(f"{BLUE}Found {len(image_files)} image files. Processing...{NC}")
    
    # Process each image
    for image_path in image_files:
        transform_image(str(image_path), output_dir, options)
    
    print(f"\n{GREEN}All transformations saved to:{NC} {output_dir}")

def main():
    # path: path of the directory to transform
    # dest: destination path, where the transformations will be saved
    # options: which transformations to apply
    path, dest, options = parse_arguments()
    
    if os.path.isfile(path):
        transform_image(path, dest, options)
    elif os.path.isdir(path):
        transform_directory(path, dest, options)
    else:
        raise Exception(f"{RED}The path is not a file or a directory:{NC} {path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Program interrupted by user.{NC}")
        sys.exit(0)
    except Exception as error:
        print(f"{RED}Error:{NC} {error}")
        sys.exit(1)