#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def print_directory_structure(path, indent="", is_last=True):
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
            print_directory_structure(
                os.path.join(path, item),
                new_indent, is_last_item
            )


def count_files(path):
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
    if not data:
        print_colored("No files found in subdirectories to plot.", YELLOW)
        return False
        
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
    plt.title(
        f'File Distribution in Subdirectories of {os.path.basename(path)}',
        fontsize=16,
        pad=20
    )
    plt.xlabel('Directories', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.tight_layout()

    # Save the chart to the plots directory
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(
        plots_dir,
        f"{os.path.basename(path)}_bar_chart.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print_colored(f"Bar chart saved as {output_file}", GREEN)
    plt.close()
    return True


def plot_pie(data, path, plots_dir):
    if not data:
        print_colored("No files found in subdirectories to plot.", YELLOW)
        return False
        
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
    legend_labels = [f"{k} ({v} files)" for k, v in sorted_data.items()]
    plt.legend(
        wedges,
        legend_labels,
        title="Directories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.axis('equal')
    plt.title(
        f'File Distribution in Subdirectories of {os.path.basename(path)}',
        fontsize=16,
        pad=20
    )
    plt.tight_layout()

    # Save the chart to the plots directory
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(
        plots_dir,
        f"{os.path.basename(path)}_pie_chart.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print_colored(f"Pie chart saved as {output_file}", GREEN)
    plt.close()
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze file distribution in subdirectories'
    )
    
    # Get project directory (parent of the script directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    default_dir = os.path.join(project_dir, "images")
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default=default_dir,
        help=f'Directory to analyze (default: {default_dir})'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=os.path.join(project_dir, "plots"),
        help=f'Output directory for plots (default: {os.path.join(project_dir, "plots")})'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    target_dir = os.path.abspath(args.directory)
    plots_dir = os.path.abspath(args.output)
    
    print_colored(f"Analyzing subdirectories in: {target_dir}", GREEN)

    # Make sure the target directory exists
    if not os.path.exists(target_dir):
        print_colored(f"Creating directory: {target_dir}", YELLOW)
        os.makedirs(target_dir)
    
    if not os.path.isdir(target_dir):
        print_colored(f"Error: {target_dir} is not a directory", RED)
        return False

    # Print the structure of the directory
    print_colored("Directory structure:", GREEN)
    # print_directory_structure(target_dir)

    # Dictionary with the name of the directory as key and the number of files
    # in the directory as value
    data = count_files(target_dir)
    
    if not data:
        print_colored("No files found in subdirectories.", YELLOW)
    
    # Create plots
    bar_result = plot_bar(data, target_dir, plots_dir)
    pie_result = plot_pie(data, target_dir, plots_dir)
    
    if bar_result and pie_result:
        return True
    else:
        print_colored("No plots were generated.", YELLOW)
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print_colored("Analysis completed successfully.", GREEN)
        else:
            print_colored("Analysis completed with warnings.", YELLOW)
        sys.exit(0 if success else 1)
    except Exception as error:
        print_colored(f"Error: {error}", RED)
        sys.exit(1)