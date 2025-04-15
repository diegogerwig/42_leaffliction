#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, run_command, run_progress_spinner, extract_source_category,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)

def get_dir_path_from_image(image_path):
    """
    Extract the directory path of the source category from an image path.
    
    Args:
        image_path: Path to the image
        
    Returns:
        str: Directory path in the format "./category"
    """
    from utils.utils import extract_source_category
    category = extract_source_category(image_path)
    
    # Return formatted path
    return f"./{category}"

def plot_image_set(images, filename, title_prefix="", max_cols=3, custom_title=None):
    """
    Plot a set of images with consistent formatting for both augmentation and transformation scripts.
    
    Parameters:
    -----------
    images : dict
        Dictionary with image labels as keys and image arrays as values.
        The first item is assumed to be the original image.
    filename : str
        Path to the original image file, used for output naming.
    title_prefix : str
        Prefix for the plot filename (e.g., "augmentation_" or "transformation_").
    max_cols : int
        Maximum number of columns per row (default: 3).
    custom_title : str, optional
        Custom title for the plot. If None, a default title is generated.
    
    Returns:
    --------
    str
        Path to the saved plot file.
    """
    # Convert BGR to RGB for display consistently
    rgb_images = {}
    for label, img in images.items():
        if len(img.shape) == 2:  # Grayscale image
            rgb_images[label] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:  # Color image (assume BGR from OpenCV)
            rgb_images[label] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_images[label] = img  # Keep as is for other formats
    
    # Extract the original image label (first key)
    original_label = list(rgb_images.keys())[0]
    
    # Get the rest of the images
    other_images = {k: v for k, v in rgb_images.items() if k != original_label}
    
    # Calculate the layout dimensions
    n_other = len(other_images)
    n_cols = min(max_cols, n_other)
    n_rows = (n_other + n_cols - 1) // n_cols  # Ceiling division
    
    # Add one row for the original image
    total_rows = n_rows + 1
    
    # Create figure with extra space for title
    fig = plt.figure(figsize=(15, 5 * total_rows + 1.5))
    
    # Get the directory path for the title
    dir_path = get_dir_path_from_image(filename)
    
    # Modify the custom title to include the directory path format
    if custom_title:
        # Extract the base filename from the custom title
        import re
        match = re.search(r'(.*?)(\(\w+,)', custom_title)
        if match:
            # Replace the category format
            custom_title = custom_title.replace(match.group(2), f"({dir_path}, ")
    
    # Add a main title with larger font at the top of the figure
    if custom_title:
        plt.figtext(0.5, 0.98, custom_title, fontsize=20, ha='center', va='top', fontweight='bold')
    
    # Create GridSpec to have more control over subplot placement
    gs = plt.GridSpec(total_rows, max_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # First row: Original image centered
    original_ax = None
    for i in range(max_cols):
        if i == max_cols // 2:  # Center position
            # Display original image
            original_ax = fig.add_subplot(gs[0, i])
            original_img = rgb_images[original_label]
            original_ax.imshow(original_img)
            original_ax.set_title(original_label.capitalize(), fontsize=12)
            
            # Show axes with scales
            original_ax.axis('on')
            
            # Set ticks for better scale reading
            height, width = original_img.shape[:2]
            # Set x-ticks at every 50 pixels
            x_ticks = np.arange(0, width, 50)
            original_ax.set_xticks(x_ticks)
            
            # Set y-ticks at every 50 pixels
            y_ticks = np.arange(0, height, 50)
            original_ax.set_yticks(y_ticks)
    
    # Remaining rows: Other images
    for idx, (label, image) in enumerate(other_images.items()):
        row = 1 + idx // max_cols
        col = idx % max_cols
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(image)
        ax.set_title(label.capitalize(), fontsize=12)
        
        # Show axes with scales
        ax.axis('on')
        
        # Set ticks for better scale reading
        height, width = image.shape[:2]
        # Set x-ticks at every 50 pixels
        x_ticks = np.arange(0, width, 50)
        ax.set_xticks(x_ticks)
        
        # Set y-ticks at every 50 pixels
        y_ticks = np.arange(0, height, 50)
        ax.set_yticks(y_ticks)
    
    # Adjust layout with padding to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for title
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.abspath("./plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate filename for the plot
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Extract source directory information
    from utils.utils import extract_source_category
    source_dir = extract_source_category(filename)
    
    # Use provided title prefix or default to generic name
    if not title_prefix:
        title_prefix = "image_processing_"
    
    # Include source directory in the filename
    plot_filename = os.path.join(plots_dir, f"{title_prefix}{base_name}_{source_dir}.png")
    
    # Save the figure with high resolution
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    return plot_filename