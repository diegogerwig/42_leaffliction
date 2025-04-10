#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# Color definitions
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def print_directory_structure(path, indent="", is_last=True):
    """Print directory structure recursively"""
    basename = os.path.basename(path)
    
    # Symbol to display in the tree
    connector = "└── " if is_last else "├── "
    
    print(f"{indent}{connector}{basename}")
    
    # Prepare indentation for child elements
    new_indent = indent + ("    " if is_last else "│   ")
    
    # Get directory items if it's a directory
    if os.path.isdir(path):
        items = sorted(os.listdir(path))
        
        # Filter hidden directories
        items = [item for item in items if not item.startswith('.')]
        
        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1)
            print_directory_structure(os.path.join(path, item), new_indent, is_last_item)

def count_files(path):
    """
    Count files in subdirectories only (exclude root directory)
    Returns a dictionary with directory name as key and 
    number of files as value
    """
    data = {}
    
    for root, dirs, files in os.walk(path):
        # Ignore hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Skip the root directory itself
        if root == path:
            continue
            
        # Get directory name relative to base directory
        rel_path = os.path.relpath(root, path)
        
        # Count only files (not directories)
        file_count = len(files)
        
        # Only add to dictionary if there are files
        if file_count > 0:
            data[rel_path] = file_count
    
    return data

def plot_bar(data, path, plots_dir):
    """Generate a bar chart with file distribution (improved version)"""
    # Sort data by number of files (from highest to lowest)
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    
    # Use a colormap for better visualization
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_data)))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        sorted_data.keys(), 
        sorted_data.values(), 
        color=colors, 
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'File Distribution in Subdirectories of {os.path.basename(path)}', fontsize=16, pad=20)
    plt.xlabel('Directories', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.tight_layout()
    
    # Save the chart to the plots directory
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(plots_dir, f"{os.path.basename(path)}_bar_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved as {output_file}")
    plt.close()

def plot_pie(data, path, plots_dir):
    """Generate a pie chart with file distribution (improved version)"""
    # Sort data by number of files (from highest to lowest)
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    
    # Custom colors for better visualization
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_data)))
    
    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        sorted_data.values(), 
        labels=None,  # We'll create a separate legend
        autopct='%1.1f%%', 
        startangle=90,
        shadow=False,  # Remove shadow
        colors=colors,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Set the opacity for better look
    for w in wedges:
        w.set_alpha(0.8)
    
    # Create legend with percentages
    total_files = sum(sorted_data.values())
    legend_labels = [f"{k} ({v} files)" for k, v in sorted_data.items()]
    plt.legend(
        wedges, 
        legend_labels,
        title="Directories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'File Distribution in Subdirectories of {os.path.basename(path)}', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the chart to the plots directory
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(plots_dir, f"{os.path.basename(path)}_pie_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pie chart saved as {output_file}")
    plt.close()

def main():
    # Hardcoded paths - automatically calculated based on script location
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    project_dir = os.path.dirname(current_dir)                # project root directory
    images_dir = os.path.join(project_dir, "images")          # directory to analyze
    plots_dir = os.path.join(project_dir, "plot")             # plots output directory
    
    print(f"Analyzing subdirectories in: {images_dir}")
    
    # Make sure the images directory exists
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory: {images_dir}")
    
    # Print the structure of the directory
    print("Directory structure:")
    print_directory_structure(images_dir)
    
    # Dictionary with the name of the directory as key and the number of files
    # in the directory as value
    data = count_files(images_dir)
    
    # Bar chart
    plot_bar(data, images_dir, plots_dir)
    
    # Pie chart
    plot_pie(data, images_dir, plots_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{RED}Error: {RESET}{error}")