#!/usr/bin/env python3
import os
import sys
import subprocess
import random
import threading
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, run_command, run_progress_spinner, 
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def find_valid_images(images_dir):
    """Find all valid images in a directory and subdirectories"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']
    
    # Dictionary to organize images by subdirectory
    subdir_images = {}
    all_valid_images = []
    
    if os.path.exists(images_dir):
        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(images_dir):
            subdir = os.path.relpath(root, images_dir)
            subdir_images[subdir] = []
            
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in [ext.lower() for ext in valid_extensions]:
                    # Store relative path from images_dir for better display
                    rel_path = os.path.relpath(file_path, images_dir)
                    all_valid_images.append(rel_path)
                    subdir_images[subdir].append(rel_path)
    
    # Remove empty subdirectories
    subdir_images = {k: v for k, v in subdir_images.items() if v}
    
    return all_valid_images, subdir_images


def select_balanced_images(subdir_images, num_images=None):
    """
    Select a balanced sample of images from each subdirectory.
    
    Args:
        subdir_images: Dictionary with subdirectories as keys and lists of images as values
        num_images: Number of images to select, or 'all' for all images in a balanced way
    
    Returns:
        list: Selected image paths
    """
    if not subdir_images:
        return []
    
    num_subdirs = len(subdir_images)
    
    if num_images == 'all':
        # Find the minimum number of images in any subdirectory
        min_images_per_subdir = min([len(images) for subdir, images in subdir_images.items()])
        total_images = min_images_per_subdir * num_subdirs
        
        print_colored(f"Balanced selection: {min_images_per_subdir} images from each of the {num_subdirs} subdirectories", GREEN)
        print_colored(f"Total: {total_images} images will be processed", GREEN)
        
        images_to_process = []
        
        # Select equal number of images from each subdirectory
        for subdir, images in subdir_images.items():
            if images:
                selected = random.sample(images, min(min_images_per_subdir, len(images)))
                images_to_process.extend(selected)
                print_colored(f"Selected {len(selected)} images from {subdir}", YELLOW)
                
        return images_to_process
    else:
        try:
            requested_num_images = int(num_images)
            if requested_num_images <= 0:
                print_colored("❌ Number must be positive.", RED)
                return []
            
            # Calculate how many images to take from each subdirectory
            # We need to round up to ensure all subdirectories get at least one image
            import math
            images_per_subdir = math.ceil(requested_num_images / num_subdirs)
            
            # Calculate the total that we'll actually process with this balanced approach
            adjusted_total = images_per_subdir * num_subdirs
            
            if adjusted_total != requested_num_images:
                print_colored(f"For a balanced selection across {num_subdirs} subdirectories:", YELLOW)
                print_colored(f"Adjusting from {requested_num_images} to {adjusted_total} total images ({images_per_subdir} per subdirectory)", YELLOW)
            
            images_to_process = []
            
            # Select a balanced number of images from each subdirectory
            for subdir, images in subdir_images.items():
                if not images:
                    continue
                
                # Calculate how many to take from this subdirectory (minimum between available and calculated)
                to_select = min(images_per_subdir, len(images))
                
                # Randomly select the images
                selected = random.sample(images, to_select)
                images_to_process.extend(selected)
                
                print_colored(f"Selected {len(selected)} images from {subdir}", YELLOW)
            
            print_colored(f"Balanced selection completed: {len(images_to_process)} images selected", GREEN)
            return images_to_process
        except ValueError:
            print_colored("❌ Invalid input. Please enter a number or 'all'.", RED)
            return []


def validate_image_path(image_input, images_dir, valid_extensions):
    """Validate and resolve the image path input by the user"""
    if not image_input:
        return None
        
    if os.path.isabs(image_input) or image_input.startswith('./') or image_input.startswith('../'):
        # Input is a path
        image_path = image_input
    else:
        # First check if the input is a relative path from the images_dir
        potential_path = os.path.join(images_dir, image_input)
        if os.path.exists(potential_path):
            image_path = potential_path
        else:
            # Try to find the image in subdirectories
            found = False
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    if file == image_input:
                        image_path = os.path.join(root, file)
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print_colored(f"❌ Image file not found: {image_input}", RED)
                return None
    
    # Check if the file is an image (based on extension)
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext not in [e.lower() for e in valid_extensions]:
        print_colored(f"Warning: File {image_path} does not have a standard image extension.", YELLOW)
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print_colored("Operation cancelled.", RED)
            return None
    
    return image_path


def run_script_on_image(python_exe, script_path, image_path, output_dir=None, extra_args=None):
    """Run a Python script on a single image"""
    # Get absolute path to the project directory (root directory)
    script_dir = os.path.dirname(os.path.abspath(script_path))
    project_dir = os.path.dirname(script_dir)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set current working directory to project root
    original_dir = os.getcwd()
    os.chdir(project_dir)
    
    # Build the command
    command = [python_exe, script_path, image_path]
    
    # Process extra arguments before adding them to the command
    if extra_args:
        parsed_args = []
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            
            # Check if this is the --dest or --output argument followed by a path
            if arg in ["--dest", "--output"] and i + 1 < len(extra_args):
                # Include the argument itself
                parsed_args.append(arg)
                
                # Then add the output path (next item in extra_args)
                # Make sure we use the directory, not a path with the filename in it
                output_path = extra_args[i + 1]
                if os.path.isdir(output_path):
                    parsed_args.append(output_path)
                else:
                    # Extract just the directory part
                    output_dir_only = os.path.dirname(output_path)
                    if output_dir_only:
                        parsed_args.append(output_dir_only)
                    else:
                        parsed_args.append(".")  # Current directory as fallback
                
                i += 2  # Skip the next item as we've already processed it
            else:
                parsed_args.append(arg)
                i += 1
        
        # Add the processed arguments
        command.extend(parsed_args)
    
    try:
        print_colored(f"\nProcessing image: {os.path.basename(image_path)}...", GREEN)
        
        # Create a progress tracker
        stop_event = threading.Event()
        progress_message = f"Processing image {os.path.basename(image_path)}"
        progress_thread = threading.Thread(target=run_progress_spinner, args=(progress_message, stop_event))
        progress_thread.daemon = True
        progress_thread.start()
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stop the spinner before showing output
        stop_event.set()
        progress_thread.join(0.2)
        
        # Show the output
        print_colored("\nProcessing progress:", GREEN)
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            print_colored(f"Error running script:", RED)
            print(stderr)
            print_colored("\nScript failed to run. Check if all dependencies are installed.", RED)
            return False
        
        print_colored(f"\n✅ Processing completed successfully for {os.path.basename(image_path)}", GREEN)
        return True
    
    except Exception as e:
        print_colored(f"Error: {e}", RED)
        return False
    finally:
        # Restore original working directory
        os.chdir(original_dir)
        
def process_script_with_image(python_exe, script_path, image_path, output_dir, extra_args=None):
    # Get base filename of the image (without extension)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Determine script type
    is_transformation = "Transformation.py" in script_path
    is_augmentation = "Augmentation.py" in script_path
    
    # Create specific subdirectory for this image if needed
    # For both Transformation.py and Augmentation.py, use one subdirectory per image with original name
    if is_transformation or is_augmentation:
        # Create a subdirectory with the original image name
        image_output_dir = os.path.join(output_dir, base_filename)
    else:
        # For other scripts, use the provided directory
        image_output_dir = output_dir
    
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Get the project directory (root directory)
    script_dir = os.path.dirname(os.path.abspath(script_path))
    project_dir = os.path.dirname(script_dir)
    
    # Save the current working directory
    original_dir = os.getcwd()
    os.chdir(project_dir)  # Change to project directory
    
    # Build the base command
    command = [python_exe, script_path, image_path]
    
    # Add specific arguments according to script type
    if is_transformation:
        # For Transformation.py, add --dest and the specific output directory
        dest_args = ["--dest", image_output_dir]
        if extra_args:
            command.extend(extra_args)
        command.extend(dest_args)
    elif is_augmentation:
        # For Augmentation.py, add --output and the specific output directory
        output_args = ["--output", image_output_dir]
        if extra_args:
            command.extend(extra_args)
        command.extend(output_args)
    else:
        # For other scripts, add additional arguments if they exist
        if extra_args:
            command.extend(extra_args)
    
    try:
        print_colored(f"\nProcessing image: {os.path.basename(image_path)}...", GREEN)
        
        # Create progress animation
        stop_event = threading.Event()
        progress_message = f"Processing image {os.path.basename(image_path)}"
        progress_thread = threading.Thread(target=run_progress_spinner, args=(progress_message, stop_event))
        progress_thread.daemon = True
        progress_thread.start()
        
        # Execute the command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stop the spinner before showing output
        stop_event.set()
        progress_thread.join(0.2)
        
        # Show process output
        print_colored("\nProcessing progress:", GREEN)
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        # Wait for the process to finish
        process.wait()
        
        # Check for errors
        if process.returncode != 0:
            stderr = process.stderr.read()
            print_colored(f"Error running script:", RED)
            print(stderr)
            print_colored("\nScript failed. Verify that all dependencies are installed.", RED)
            return False
        
        print_colored(f"\n✅ Processing completed successfully for {os.path.basename(image_path)}", GREEN)
        print_colored(f"Results saved to: {image_output_dir}", GREEN)
        return True
    
    except Exception as e:
        print_colored(f"Error: {e}", RED)
        return False
    finally:
        # Restore the original working directory
        os.chdir(original_dir)
        
		
def process_images_batch(python_exe, script_path, images_dir, output_dir, image_selection, extra_args=None):
    """
    Process a batch of images with a specific script.
    """
    # Determine the type of script being executed
    script_type = ""
    if "Transformation.py" in script_path:
        script_type = "transformation"
    elif "Augmentation.py" in script_path:
        script_type = "augmentation"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print_colored(f"\nOutput directory: {output_dir}", GREEN)
    
    # Find all valid images in the specified directory
    all_valid_images, subdir_images = find_valid_images(images_dir)
    
    # If there are no images, exit
    if not all_valid_images:
        print_colored(f"\nNo valid images in {images_dir}", RED)
        return False
    
    # If the selected directory has a single file and it's an image
    if os.path.isfile(images_dir) and os.path.splitext(images_dir)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        print_colored(f"Processing single file: {images_dir}", GREEN)
        return process_script_with_image(python_exe, script_path, images_dir, output_dir, extra_args)
    
    # If 'all' or a specific number of images is selected
    if image_selection == 'all' or (isinstance(image_selection, str) and image_selection.isdigit()):
        print_colored(f"\nBalanced selection mode activated for {image_selection} images.", GREEN)
        
        # Select balanced images from each subdirectory
        images_to_process = select_balanced_images(subdir_images, image_selection)
        
        if not images_to_process:
            print_colored("❌ No images selected for processing.", RED)
            return False
        
        # Show the selected images
        print_colored("\nImages selected for processing:", GREEN)
        for i, img in enumerate(images_to_process):
            print(f"  {i+1}. {img}")
        
        # Process each selected image
        success_count = 0
        for img_file in images_to_process:
            # Build full path from relative path
            image_path = os.path.join(images_dir, img_file)
            print_colored(f"\nProcessing: {img_file}", BLUE)
            
            # Get the basename and create subdirectory with original image name
            image_basename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_basename)[0]

            # For both transformation and augmentation, create a subdirectory with original image name
            image_output_dir = os.path.join(output_dir, name_without_ext)
            
            os.makedirs(image_output_dir, exist_ok=True)
            
            success = process_script_with_image(python_exe, script_path, image_path, image_output_dir, extra_args)
            if success:
                success_count += 1
        
        print_colored(f"\n✅ Successfully processed {success_count} of {len(images_to_process)} images", GREEN)
        return success_count > 0
    
    # If a specific path is provided
    elif os.path.exists(image_selection):
        if os.path.isfile(image_selection):
            # Process a single file
            return process_script_with_image(python_exe, script_path, image_selection, output_dir, extra_args)
        elif os.path.isdir(image_selection):
            # Process all images in the directory
            dir_images, dir_subdir_images = find_valid_images(image_selection)
            
            if not dir_images:
                print_colored(f"❌ No images found in {image_selection}", RED)
                return False
            
            print_colored(f"\nProcessing all images in: {image_selection}", GREEN)
            
            success_count = 0
            for img_rel_path in dir_images:
                img_path = os.path.join(image_selection, img_rel_path)
                
                # Get the basename and create subdirectory with original image name
                image_basename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(image_basename)[0]
                
                # For both transformation and augmentation, create a subdirectory with original image name
                image_output_dir = os.path.join(output_dir, name_without_ext)
                
                os.makedirs(image_output_dir, exist_ok=True)
                
                success = process_script_with_image(python_exe, script_path, img_path, image_output_dir, extra_args)
                if success:
                    success_count += 1
            
            print_colored(f"\n✅ Successfully processed {success_count} of {len(dir_images)} images", GREEN)
            return success_count > 0
    
    # If the user input couldn't be processed
    else:
        print_colored(f"❌ Could not process input: {image_selection}", RED)
        print_colored("Please specify 'all', a number, or a valid path.", YELLOW)
        return False