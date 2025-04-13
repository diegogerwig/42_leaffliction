#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from pyfiglet import Figlet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def parse_argument():
    parser = argparse.ArgumentParser(
        description="Augment an image with various techniques."
    )
    
    parser.add_argument(
        "image", 
        type=str,
        help="Path to the image to augment"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output directory for augmented images",
        default=None
    )
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # Check if the file is an image (based on extension)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ext = os.path.splitext(args.image)[1].lower()
    
    if ext not in valid_extensions:
        print(f"{RED}Warning:{NC} File {args.image} does not have a standard image extension.")
    
    # If output is not specified, use default ./images_augmented
    if args.output is None:
        args.output = "./images_augmented"
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    print(f"{GREEN}Output directory:{NC} {args.output}")
    
    return args


def read_image(filename):
    return cv2.imread(filename)


def flip_image(image, flip_code=1):
    """
    Flip an image horizontally, vertically, or both
    """
    return cv2.flip(image, flip_code)


def rotate_image(image, angle=None):
    """
    Rotate an image by a random angle between 15 and 45 degrees (positive or negative)
    """
    if angle is None:
        # Generar un número aleatorio entre 15 y 45, luego multiplicar por un signo aleatorio
        magnitude = random.randint(15, 45)
        sign = random.choice([-1, 1])
        angle = magnitude * sign
        
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    return cv2.warpAffine(image, rotation_matrix, (width, height),
                         flags=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(255, 255, 255))


def distort_image(image, alpha=None, blur_strength=None, brightness_factor=None):
    """
    Apply elastic distortion to an image with blur and brightness adjustment
    """
    
    # Apply elastic distortion
    if alpha is None:
        alpha = random.randint(25, 50)
        
    # Set random blur strength if not provided
    if blur_strength is None:
        blur_strength = random.randint(1, 3)  # Valores entre 1 y 3 para un efecto visible
    
    # Set random brightness factor if not provided
    if brightness_factor is None:
        # Valores entre 0.7 y 1.3 para oscurecer o aclarar la imagen
        brightness_factor = random.uniform(0.7, 1.3)
        
    height, width = image.shape[:2]
    
    # Create displacement fields
    dx = np.random.uniform(-1, 1, (height, width)).astype(np.float32) * alpha
    dy = np.random.uniform(-1, 1, (height, width)).astype(np.float32) * alpha
    
    # Smooth displacement fields
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=7.0, sigmaY=7.0)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=7.0, sigmaY=7.0)
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply displacement fields
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # Remap image
    distorted = cv2.remap(image, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(255, 255, 255))
    
    # Apply blur if requested
    if blur_strength > 0:
        # Make sure blur_strength is odd
        blur_kernel = 2 * blur_strength + 1 if blur_strength > 0 else 1
        distorted = cv2.GaussianBlur(distorted, (blur_kernel, blur_kernel), 0)
    
    # Apply brightness adjustment if requested
    if brightness_factor != 1.0:
        distorted = cv2.convertScaleAbs(distorted, alpha=brightness_factor, beta=0)
    
    return distorted


def skew_image(image, intensity=None):
    """
    Apply skew transformation to an image
    """
    if intensity is None:
        intensity = random.uniform(0.2, 0.4)
        
    height, width = image.shape[:2]
    
    # Choose skew direction (horizontal or vertical or both)
    skew_type = random.choice(['horizontal', 'vertical', 'both'])
    
    # Define source points
    src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Define destination points based on skew type
    if skew_type == 'horizontal':
        offset = width * intensity
        dst_pts = np.float32([[offset, 0], [width - offset, 0], 
                             [0, height], [width, height]])
    elif skew_type == 'vertical':
        offset = height * intensity
        dst_pts = np.float32([[0, offset], [width, offset], 
                             [0, height - offset], [width, height - offset]])
    else:  # both
        offset_w = width * intensity
        offset_h = height * intensity
        dst_pts = np.float32([[offset_w, offset_h], [width - offset_w, offset_h], 
                             [0, height - offset_h], [width, height - offset_h]])
    
    # Get perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply transformation
    return cv2.warpPerspective(image, transform_matrix, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))


def shear_image(image, intensity=None):
    """
    Apply shear transformation to an image
    """
    if intensity is None:
        intensity = random.uniform(0.2, 0.4)
        
    height, width = image.shape[:2]
    
    # Choose shear direction (x or y)
    shear_direction = random.choice(['x', 'y'])
    
    if shear_direction == 'x':
        # Shear in x direction
        M = np.float32([
            [1, intensity, 0],
            [0, 1, 0]
        ])
    else:
        # Shear in y direction
        M = np.float32([
            [1, 0, 0],
            [intensity, 1, 0]
        ])
    
    # Apply affine transformation
    return cv2.warpAffine(image, M, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(255, 255, 255))


def crop_image(image, crop_percentage=None):
    """
    Crop an image and resize back to original dimensions
    """
    if crop_percentage is None:
        crop_percentage = random.uniform(0.2, 0.4)
        
    height, width = image.shape[:2]
    
    # Calculate crop dimensions
    crop_width = int(width * crop_percentage)
    crop_height = int(height * crop_percentage)
    
    # Get random crop position
    x1 = random.randint(0, crop_width)
    y1 = random.randint(0, crop_height)
    x2 = width - random.randint(0, crop_width)
    y2 = height - random.randint(0, crop_height)
    
    # Ensure valid crop dimensions
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    # Perform crop
    cropped = image[y1:y2, x1:x2]
    
    # Resize back to original dimensions
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def save_images(images, filename, output_dir):
    # Get the filename without extension
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    
    # Create subdirectory with the image name
    img_output_dir = os.path.join(output_dir, name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    print(f"{GREEN}Saving augmented images to: {img_output_dir}{NC}")
    
    # Skip the original image when saving (we don't need to save it again)
    for label, image in images.items():
        if label == "Original":
            continue
            
        # Construct the new filename with the augmentation type
        new_filename = os.path.join(img_output_dir, f"{name}_{label}{ext}")
        
        # Save the image
        cv2.imwrite(new_filename, image)
        print(f"Saved: {os.path.basename(new_filename)}")


def plot_images(images, filename):
    """
    Plot original and augmented images in a 3x3 grid and save to ./plots directory
    """
    # Convert BGR to RGB for display with matplotlib
    rgb_images = {label: cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                 for label, img in images.items()}
    
    # Fixed layout: 3 rows, 3 columns
    nrows, ncols = 3, 3
    
    # Create a figure with 3x3 grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    
    # Plot the original image in the top row, centered
    ax = axes[0, 1]  # Center position in the first row
    ax.imshow(rgb_images["Original"])
    ax.set_title("Original", fontsize=12)
    ax.axis('off')
    
    # Turn off the unused subplots in the first row
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    
    # Create a list of the augmented images (all except the original)
    augmented_images = [(label, img) for label, img in rgb_images.items() if label != "Original"]
    
    # Plot the first 3 augmented images in the second row
    for col, (label, image) in enumerate(augmented_images[:3]):
        ax = axes[1, col]
        ax.imshow(image)
        ax.set_title(label, fontsize=12)
        ax.axis('off')
    
    # Plot the remaining 3 augmented images in the third row
    for col, (label, image) in enumerate(augmented_images[3:]):
        ax = axes[2, col]
        ax.imshow(image)
        ax.set_title(label, fontsize=12)
        ax.axis('off')
    
    # Fill any remaining empty spots in the third row
    for col in range(len(augmented_images[3:]), 3):
        axes[2, col].axis('off')
    
    # Set the window title
    plt.suptitle(f'Augmentation: {os.path.basename(filename)}', fontsize=20, y=0.98)
    plt.tight_layout()

    plt.subplots_adjust(top=0.93)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.abspath("./plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate a filename for the saved plot
    base_name = os.path.splitext(os.path.basename(filename))[0]
    plot_filename = os.path.join(plots_dir, f"augmentation_{base_name}.png")
    
    # Save the figure
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    return plot_filename


def main():
    # Parse command line arguments to get the filename and output directory
    args = parse_argument()
    filename = args.image
    output_dir = args.output

    # Read the image
    image = read_image(filename)
    
    if image is None:
        print(f"{RED}Error:{NC} Could not read image {filename}")
        sys.exit(1)

    # Dictionary with the label of the image as key and the augmented image as value
    images = {
        "Original": image,
        "Flipped": flip_image(image),
        "Rotated": rotate_image(image),
        "Skew": skew_image(image),
        "Shear": shear_image(image),
        "Crop": crop_image(image),
        "Distortion": distort_image(image)
    }

    # Save the augmented images
    save_images(images, filename, output_dir)
    
    # Display the original and augmented images
    plot_images(images, filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{RED}Error:{NC} {error}")
        sys.exit(1)