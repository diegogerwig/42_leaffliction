#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import threading
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.venv_manager import setup_and_activate_environment, clean_directories
from utils.code_quality import run_flake8
from utils.utils import (
    print_colored, run_command, run_progress_spinner, 
    wait_for_confirmation, get_default_images_dir,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def run_augmentation(config, project_dir):
    """Run the image augmentation script with balanced sampling across subdirectories"""
    print_colored("\n=== Image Augmentation Tool ===", BLUE)
    
    # Check if the script exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    augmentation_script = os.path.join(script_dir, "Augmentation.py")
    
    if not os.path.exists(augmentation_script):
        print_colored(f"❌ Could not find Augmentation.py at {augmentation_script}", RED)
        return False
    
    if config['use_conda']:
        python_exe = os.path.join(config['env_path'], 'bin', 'python')
    else:
        python_exe = config['python_bin']
    
    # Set default images directory path
    images_dir = get_default_images_dir(project_dir)
    
    # Check if images directory exists and create it if needed
    if not os.path.exists(images_dir):
        print_colored(f"Images directory not found at {images_dir}. Creating it now...", YELLOW)
        os.makedirs(images_dir, exist_ok=True)
        print_colored(f"✅ Created images directory: {images_dir}", GREEN)
    
    # Get list of valid images in the directory and subdirectories
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
                if os.path.isfile(file_path) and os.path.splitext(file)[1] in valid_extensions:
                    # Store relative path from images_dir for better display
                    rel_path = os.path.relpath(file_path, images_dir)
                    all_valid_images.append(rel_path)
                    subdir_images[subdir].append(rel_path)
    
    # Remove empty subdirectories
    subdir_images = {k: v for k, v in subdir_images.items() if v}
    
    # Show available images in the images directory if any
    if all_valid_images:
        print_colored("\nAvailable images in the default directory:", GREEN)
        for i, file in enumerate(all_valid_images):
            print(f"  {i+1}. {file}")
        print_colored(f"  Total: {len(all_valid_images)} images found", YELLOW)
        
        if len(subdir_images) > 1:  # Only show subdirectory info if there are multiple subdirs
            print_colored("\nImages by subdirectory:", GREEN)
            for subdir, images in subdir_images.items():
                print(f"  {subdir}: {len(images)} images")
    else:
        print_colored(f"\nNo valid images found in {images_dir}", RED)
        print_colored("Please add some images to the directory and try again.", YELLOW)
        return False
    
    # Ask user for the image path to augment
    print_colored("\n🖼️ Image Augmentation Configuration", BLUE)
    print_colored(f"Enter the directory to augment (or press ENTER for default '{images_dir}' directory):", GREEN)
    image_input = input().strip()
    
    # Handle empty input - random selection mode
    if not image_input:
        if not all_valid_images:
            print_colored("❌ No valid images found in the directory. Cancelling augmentation.", RED)
            return False
            
        print_colored("\nBalanced random selection mode activated.", GREEN)
        print_colored("Enter the number of random images to process (or 'all' for all images):", GREEN)
        num_input = input().strip().lower()
        
        if num_input == 'all':
            # Find the subdirectory with the minimum number of images for balanced selection
            min_images_per_subdir = min([len(images) for subdir, images in subdir_images.items()])
            total_images = min_images_per_subdir * len(subdir_images)
            
            print_colored(f"Balanced selection: {min_images_per_subdir} images from each of the {len(subdir_images)} subdirectories", GREEN)
            print_colored(f"Total: {total_images} images will be processed", GREEN)
            
            images_to_process = []
            import random
            
            # Select equal number of images from each subdirectory
            for subdir, images in subdir_images.items():
                if images:
                    selected = random.sample(images, min(min_images_per_subdir, len(images)))
                    images_to_process.extend(selected)
                    print_colored(f"Selected {len(selected)} images from {subdir}", YELLOW)
        else:
            try:
                requested_num_images = int(num_input)
                if requested_num_images <= 0:
                    print_colored("❌ Number must be positive. Cancelling augmentation.", RED)
                    return False
                
                # Calculate images per subdirectory to achieve balance
                num_subdirs = len(subdir_images)
                
                if num_subdirs > 0:
                    # Calculate images needed per subdirectory, rounded up to the nearest integer
                    import math
                    images_per_subdir = math.ceil(requested_num_images / num_subdirs)
                    
                    # Calculate total after balancing
                    total_after_balancing = images_per_subdir * num_subdirs
                    
                    if total_after_balancing != requested_num_images:
                        print_colored(f"For balanced selection across {num_subdirs} subdirectories:", YELLOW)
                        print_colored(f"Adjusting from {requested_num_images} to {total_after_balancing} images total ({images_per_subdir} per subdirectory)", YELLOW)
                    
                    images_to_process = []
                    import random
                    
                    # Select balanced images from each subdirectory
                    for subdir, images in subdir_images.items():
                        if not images:
                            continue
                            
                        # Select minimum between calculated images_per_subdir and available images
                        to_select = min(images_per_subdir, len(images))
                        selected = random.sample(images, to_select)
                        images_to_process.extend(selected)
                        
                        print_colored(f"Selected {len(selected)} images from {subdir}", YELLOW)
                    
                    print_colored(f"Balanced selection complete: {len(images_to_process)} images selected", GREEN)
                    
                    # Display the selected images
                    print_colored("\nImages selected for processing:", GREEN)
                    for i, img in enumerate(images_to_process):
                        print(f"  {i+1}. {img}")
                else:
                    # Fallback if no subdirectories with images
                    print_colored(f"No subdirectories with images found. Using standard random selection.", YELLOW)
                    num_images = min(requested_num_images, len(all_valid_images))
                    images_to_process = random.sample(all_valid_images, num_images)
            except ValueError:
                print_colored("❌ Invalid input. Please enter a number or 'all'. Cancelling augmentation.", RED)
                return False
                
        # Process multiple images
        success_count = 0
        for img_file in images_to_process:
            # Construct full path from relative path
            image_path = os.path.join(images_dir, img_file)
            print_colored(f"\nProcessing: {img_file}", BLUE)
            success = process_single_image(python_exe, augmentation_script, image_path)
            if success:
                success_count += 1
                
        print_colored(f"\n✅ Successfully processed {success_count} out of {len(images_to_process)} images", GREEN)
        return success_count > 0
        
    # Regular single image processing
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
                return False
    
    # Check if the file is an image (based on extension)
    ext = os.path.splitext(image_path)[1]
    
    if ext not in valid_extensions:
        print_colored(f"Warning: File {image_path} does not have a standard image extension.", YELLOW)
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print_colored("Cancelling augmentation.", RED)
            return False
    
    # Process the single image
    return process_single_image(python_exe, augmentation_script, image_path)


def process_single_image(python_exe, augmentation_script, image_path):
    """Process a single image with the augmentation script"""
    # Get absolute path to the project directory (root directory)
    # First get src directory, then go up one level to reach project root
    script_dir = os.path.dirname(os.path.abspath(augmentation_script))
    project_dir = os.path.dirname(script_dir)
    
    # Create images_augmented directory
    output_dir = os.path.join(project_dir, "images_augmented")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set current working directory to project root
    # This makes relative paths in Augmentation.py start from project root
    original_dir = os.getcwd()
    os.chdir(project_dir)
    
    # Build the command to run the augmentation script with output directory
    command = [python_exe, augmentation_script, image_path, "--output", output_dir]
    
    try:
        print_colored(f"\nRunning augmentation on {os.path.basename(image_path)}...", GREEN)
        
        # Create a progress tracker with steps
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
        print_colored("\nAugmentation progress:", GREEN)
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            print_colored(f"Error running Augmentation.py:", RED)
            print(stderr)
            print_colored(
                "\nAugmentation.py failed to run. Check if all dependencies are installed.",
                RED
            )
            return False
        
        print_colored(f"\n✅ Augmentation completed successfully for {os.path.basename(image_path)}", GREEN)
        return True
    
    except Exception as e:
        print_colored(f"Error: {e}", RED)
        return False
    finally:
        # Restore original working directory
        os.chdir(original_dir)


def run_distribution(config, project_dir):
    """Run the Distribution.py script to analyze file distribution"""
    distribution_script = os.path.join(project_dir, "src", "Distribution.py")
    plots_dir = os.path.join(project_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    print_colored("\n=== File Distribution Analysis ===", BLUE)

    if config['use_conda']:
        python_exe = os.path.join(config['env_path'], 'bin', 'python')
    else:
        python_exe = config['python_bin']

    if not os.path.exists(distribution_script):
        print_colored(f"❌ Distribution.py not found at {distribution_script}", RED)
        sys.exit(1)

    # Default images directory
    images_dir = get_default_images_dir(project_dir)
    
    # Create the images directory if it doesn't exist
    if not os.path.exists(images_dir):
        create_dir = input(f"Default directory {images_dir} does not exist. Create it? (y/n): ").strip().lower()
        if create_dir == 'y':
            os.makedirs(images_dir, exist_ok=True)
            print_colored(f"Created directory: {images_dir}", GREEN)
    
    # Count images in the directory and subdirectories
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']
    image_count = 0
    subdirs = []
    
    if os.path.exists(images_dir):
        for root, dirs, files in os.walk(images_dir):
            has_images = False
            for file in files:
                if os.path.splitext(file)[1].lower() in [ext.lower() for ext in valid_extensions]:
                    has_images = True
                    image_count += 1
            
            if has_images and root != images_dir:
                subdirs.append(os.path.relpath(root, images_dir))
    
    if image_count > 0:
        print_colored(f"\nFound {image_count} images in directory and subdirectories", GREEN)
        if subdirs:
            print_colored("Subdirectories with images:", GREEN)
            for i, subdir in enumerate(subdirs):
                print(f"  {i+1}. {subdir}")

    # Ask user for target directory
    print_colored("\n📁 Target Directory Configuration", BLUE)
    print_colored(f"Enter the directory to analyze (or press ENTER for default '{images_dir}' directory):", GREEN)
    print_colored("You can also specify a subdirectory by name.", YELLOW)
    target_dir = input().strip()
    
    if not target_dir:
        target_dir = images_dir
        print_colored(f"Using default directory: {target_dir}", YELLOW)
    elif not os.path.isabs(target_dir):
        # Check if it's a subdirectory name
        potential_subdir = os.path.join(images_dir, target_dir)
        if os.path.exists(potential_subdir) and os.path.isdir(potential_subdir):
            target_dir = potential_subdir
            print_colored(f"Using subdirectory: {target_dir}", YELLOW)
        else:
            # It might be an absolute path from the user's perspective
            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                print_colored(f"Using directory: {target_dir}", YELLOW)
            else:
                create_dir = input(f"Directory {target_dir} does not exist. Create it? (y/n): ").strip().lower()
                if create_dir == 'y':
                    os.makedirs(target_dir, exist_ok=True)
                    print_colored(f"Created directory: {target_dir}", GREEN)
                else:
                    print_colored("Exiting as directory doesn't exist.", RED)
                    return False

    # Build the command with the target directory as argument
    command = [python_exe, distribution_script, '--directory', target_dir, '--output', plots_dir]

    try:
        # Create a progress spinner
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=run_progress_spinner, 
            args=(f"Analyzing directory {os.path.basename(target_dir)}", stop_event)
        )
        progress_thread.daemon = True
        
        # Create a variable to track line count for progress updates
        line_count = 0
        last_progress_update = time.time()
        
        print_colored("\nStarting directory analysis...", GREEN)
        
        # Show initial progress
        progress_thread.start()
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Process output line by line with progress updates
        for line in iter(process.stdout.readline, ''):
            line_count += 1
            
            # Show progress every second
            current_time = time.time()
            if current_time - last_progress_update >= 1.0:
                stop_event.set()
                progress_thread.join(0.2)
                
                # Print the line and reset progress spinner
                print(line.rstrip())
                
                # Create a new progress spinner
                stop_event = threading.Event()
                progress_thread = threading.Thread(
                    target=run_progress_spinner, 
                    args=(f"Processing data (lines: {line_count})", stop_event)
                )
                progress_thread.daemon = True
                progress_thread.start()
                
                last_progress_update = current_time
            elif "plot_" in line or "saved as" in line or "directory structure" in line:
                # Always show important messages immediately
                stop_event.set()
                progress_thread.join(0.2)
                print(line.rstrip())
                
                # Create a new progress spinner
                stop_event = threading.Event()
                progress_thread = threading.Thread(
                    target=run_progress_spinner, 
                    args=(f"Processing data (lines: {line_count})", stop_event)
                )
                progress_thread.daemon = True
                progress_thread.start()
                
                last_progress_update = current_time

        # Stop the spinner before checking process result
        stop_event.set()
        progress_thread.join(0.2)
        
        process.wait()
        
        # Check for any errors
        if process.returncode != 0:
            stderr = process.stderr.read()
            print_colored("Error running Distribution.py:", RED)
            print(stderr)
            print_colored(
                "\nDistribution.py failed to run. Check if all dependencies are installed.",
                RED
            )
            return False

    except Exception as e:
        # Make sure to stop the spinner thread if there's an exception
        stop_event.set()
        if 'progress_thread' in locals() and progress_thread.is_alive():
            progress_thread.join(0.2)
        
        print_colored(f"Error: {e}", RED)
        return False

    print_colored("\nChecking for generated plots...", GREEN)
    if os.path.isdir(plots_dir):
        plots = [
            os.path.join(plots_dir, f) for f in os.listdir(plots_dir)
            if f.endswith('.png')
        ]

        if not plots:
            print_colored(f"No plots were generated in {plots_dir}", RED)
            return False

        print_colored(f"Found {len(plots)} plots. Opening...", GREEN)
        
        # Open plots using Linux's xdg-open
        for plot in plots:
            try:
                subprocess.Popen(['xdg-open', plot])
                print_colored(f"Opened: {os.path.basename(plot)}", GREEN)
            except Exception as e:
                print_colored(f"Could not open {plot}: {e}", RED)
                print_colored(f"You can manually view the plots at: {plots_dir}", GREEN)
        
        return True
    else:
        print_colored(f"Plots directory not found at {plots_dir}", RED)
        return False


def show_menu(config, project_dir):
    """Show the main menu and handle user selection"""
    while True:
        try:
            print_colored("\n=== Machine Learning Project Tools ===", BLUE)
            print_colored("1. Run code quality check (flake8)", GREEN)
            print_colored("2. Run data distribution analysis", GREEN)
            print_colored("3. Run image augmentation", GREEN)
            print_colored("0. Exit", GREEN)
            
            choice = input("\nEnter your choice (0-3): ").strip()
            
            if choice == '0':
                print_colored("Exiting program. Goodbye!", BLUE)
                sys.exit(0)
            elif choice == '1':
                run_flake8(config, project_dir)
                wait_for_confirmation()
            elif choice == '2':
                run_distribution(config, project_dir)
                wait_for_confirmation()
            elif choice == '3':
                run_augmentation(config, project_dir)
                wait_for_confirmation()
            else:
                print_colored("Invalid choice. Please try again.", RED)
                input("Press ENTER to continue...")
        except KeyboardInterrupt:
            print("\n")  # Add a newline for better formatting
            print_colored("CTRL+C detected. Exiting program. Goodbye!", BLUE)
            sys.exit(0)


def main():
    try:
        os.system('clear')
        
        # Setup and activate the environment
        env_ready, config = setup_and_activate_environment()

        if not env_ready:
            print_colored("Failed to set up environment. Exiting.", RED)
            sys.exit(1)

        # Continue with the rest of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)

        # Ensure the images directory exists
        images_dir = get_default_images_dir(project_dir)
        if not os.path.exists(images_dir):
            print_colored(f"Creating default images directory: {images_dir}", YELLOW)
            os.makedirs(images_dir, exist_ok=True)
            
        # Clean plot and images_augmented directories
        print_colored("\nCleaning output directories after environment setup...", BLUE)
        clean_directories(project_dir)

        # Show the main menu and handle user selection
        show_menu(config, project_dir)
    
    except KeyboardInterrupt:
        print("\n")  # Add a newline for better formatting
        print_colored("CTRL+C detected. Exiting program. Goodbye!", BLUE)
        sys.exit(0)


if __name__ == "__main__":
    main()