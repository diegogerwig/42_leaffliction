#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import threading
import shutil
from pathlib import Path


def clear_screen():
    os.system('clear')


# Colors for better output
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color

# Spinner animation characters
SPINNER_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


def print_colored(text, color):
    print(f"{color}{text}{NC}")


def get_config():
    """Get configuration for virtual environment"""
    home = os.environ.get('HOME', os.path.expanduser('~'))
    
    if os.path.exists(f"{home}/sgoinfre"):
        conda_path = f"{home}/sgoinfre/miniforge"
        return {
            'conda_path': conda_path,
            'conda_bin': f"{conda_path}/bin/conda",
            'env_name': "leaf_env",
            'env_path': f"{conda_path}/envs/leaf_env",
            'use_conda': True
        }
    else:
        venv_path = os.path.join(home, "leaf_env")
        return {
            'venv_path': venv_path,
            'env_name': "leaf_env",
            'python_bin': os.path.join(venv_path, 'bin', 'python'),
            'pip_bin': os.path.join(venv_path, 'bin', 'pip'),
            'use_conda': False
        }


def run_command(command, shell=False, capture_output=False):
    try:
        if shell:
            if capture_output:
                return subprocess.run(
                    command, shell=True, check=True, capture_output=True, text=True
                )
            return subprocess.run(command, shell=True, check=True)
        else:
            if capture_output:
                return subprocess.run(
                    command.split(), check=True, capture_output=True, text=True
                )
            return subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Error executing: {command}", RED)
        if capture_output and e.stderr:
            print_colored(f"Error details: {e.stderr}", RED)
        return False


def create_venv(config):
    print_colored('\n🔆 Creating new virtual environment', BLUE)
    
    if config['use_conda']:
        conda_cmd = f"{config['conda_bin']} create -y -n {config['env_name']} python=3.11"
        return run_command(conda_cmd, shell=True)
    else:
        if not run_command("python3 -m venv --help", shell=True, capture_output=True):
            print_colored("❌ Python venv module not available. Please install it first:", RED)
            print_colored("    sudo apt-get install python3-venv  # For Debian/Ubuntu", YELLOW)
            print_colored("    brew install python3  # For macOS with Homebrew", YELLOW)
            print_colored(
                "    python -m pip install virtualenv  # Alternative approach", 
                YELLOW
            )
            return False
            
        os.makedirs(os.path.dirname(config['venv_path']), exist_ok=True)
        result = run_command(f"python3 -m venv {config['venv_path']}", shell=True)
        
        if result:
            print_colored(f"✅ Created virtual environment at {config['venv_path']}", GREEN)
        
        return result


def venv_exists(config):
    """Check if the virtual environment already exists"""
    if config['use_conda']:
        env_check = run_command("conda info --envs", shell=True, capture_output=True)
        return env_check and config['env_name'] in env_check.stdout
    else:
        return os.path.exists(config['python_bin'])


def activate_venv(config):
    print_colored('\n🐍 Activating virtual environment', BLUE)
    
    if config['use_conda']:
        activate_cmd = (
            f". {config['conda_path']}/etc/profile.d/conda.sh && "
            f"conda activate {config['env_name']}"
        )
        run_command(activate_cmd, shell=True)
        
        env_check = run_command("conda info --envs", shell=True, capture_output=True)
        if env_check and config['env_name'] in env_check.stdout:
            print_colored(f"✅ Activated conda environment: {config['env_name']}", GREEN)
            return True
        else:
            print_colored("❌ Failed to activate conda environment", RED)
            return False
    else:
        if os.path.exists(config['python_bin']):
            print_colored(f"✅ Virtual environment found at: {config['venv_path']}", GREEN)
            return True
        else:
            print_colored(f"❌ Virtual environment not found at {config['venv_path']}", RED)
            return False


def install_dependencies(config):
    print_colored('\n🔗 Installing dependencies', BLUE)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    requirements_file = os.path.join(project_dir, "requirements.txt")
    
    # Check if requirements.txt exists, otherwise show a warning
    if not os.path.exists(requirements_file):
        print_colored("⚠️ requirements.txt not found! Dependencies may not be installed correctly.", YELLOW)
        return False
    
    if config['use_conda']:
        pip_command = (
            f"{config['env_path']}/bin/pip install -r {requirements_file}"
        )
    else:
        pip_command = f"{config['pip_bin']} install -r {requirements_file}"
    
    result = run_command(pip_command, shell=True)
    if result:
        print_colored("✅ Dependencies installed successfully", GREEN)
    else:
        print_colored("⚠️ Some dependencies might not have installed correctly", YELLOW)
    
    return True


def clean_directories(project_dir):
    """Clean the plot and images_augmented directories while preserving .gitkeep files"""
    print_colored('\n🧹 Cleaning output directories', BLUE)
    
    dirs_to_clean = [
        os.path.join(project_dir, "plot"),
        os.path.join(project_dir, "images_augmented")
    ]
    
    for dir_path in dirs_to_clean:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print_colored(f"✅ Created directory: {dir_path}", GREEN)
            # Create .gitkeep file
            gitkeep_path = os.path.join(dir_path, ".gitkeep")
            with open(gitkeep_path, 'w') as f:
                pass
            continue
            
        cleaned = 0
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            # Skip .gitkeep files
            if item == ".gitkeep":
                continue
                
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                    cleaned += 1
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    cleaned += 1
            except Exception as e:
                print_colored(f"⚠️ Failed to remove {item_path}: {e}", YELLOW)
        
        # Create .gitkeep if it doesn't exist
        gitkeep_path = os.path.join(dir_path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                pass
                
        print_colored(f"✅ Cleaned {cleaned} items from {dir_path}", GREEN)
    
    return True


def setup_and_activate_environment():
    """Setup (if needed) and activate the virtual environment"""
    config = get_config()
    
    # Check if we're already in the correct virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    env_path = os.environ.get('CONDA_PREFIX') or os.environ.get('VIRTUAL_ENV')
    conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')
    
    if in_venv or env_path:
        env_name = conda_env_name or (os.path.basename(env_path) if env_path else "unknown")
        if (config['use_conda'] and conda_env_name == config['env_name']) or \
           (not config['use_conda'] and env_path == config['venv_path']):
            print_colored(f"✅ Already in the correct virtual environment: {env_name}", GREEN)
            
            # Even if we're in the correct environment, ensure dependencies are installed
            print_colored("Ensuring all dependencies are installed...", YELLOW)
            install_dependencies(config)
            
            return True, config
        else:
            print_colored(f"Warning: Currently in a different virtual environment: {env_name}", YELLOW)
            print_colored(
                f"Will attempt to switch to the required environment: {config['env_name']}",
                YELLOW
            )
    
    # Check if the environment exists
    if venv_exists(config):
        # Environment exists, just activate it
        activated = activate_venv(config)
        if activated:
            # Install dependencies after activation
            print_colored("Installing dependencies after activation...", YELLOW)
            install_dependencies(config)
        return activated, config
    else:
        # Environment doesn't exist, create and activate it
        print_colored(
            f"Virtual environment {config['env_name']} not found. Creating it now...",
            YELLOW
        )
        if create_venv(config) and activate_venv(config) and install_dependencies(config):
            print_colored("✅ Virtual environment setup complete!", GREEN)
            return True, config
        else:
            print_colored("❌ Failed to set up the virtual environment", RED)
            return False, config


def run_progress_spinner(message, stop_event):
    """Run a spinning animation to indicate progress"""
    i = 0
    steps = 0
    
    while not stop_event.is_set():
        steps_indicator = "." * (steps % 4)
        print(f"\r{CYAN}{message} {SPINNER_CHARS[i % len(SPINNER_CHARS)]} {steps_indicator:<3}{NC}", end="")
        i += 1
        if i % 10 == 0:
            steps += 1
        time.sleep(0.1)
    
    # Clear the line when done
    print("\r" + " " * 100 + "\r", end="")


def run_flake8(config, project_dir):
    """Run flake8 to check code quality and show a summary"""
    print_colored('\n🔍 Running flake8 code quality check', BLUE)
    
    # Install flake8 if not already installed
    if config['use_conda']:
        pip_bin = os.path.join(config['env_path'], 'bin', 'pip')
        flake8_bin = os.path.join(config['env_path'], 'bin', 'flake8')
    else:
        pip_bin = config['pip_bin']
        flake8_bin = os.path.join(os.path.dirname(config['pip_bin']), 'flake8')
    
    # Check if flake8 is installed
    if not os.path.exists(flake8_bin):
        print_colored("Installing flake8...", YELLOW)
        run_command(f"{pip_bin} install flake8", shell=True)
    
    # Run flake8 on the project directory
    print_colored(f"Running flake8 on {project_dir}...", GREEN)
    
    # Start a spinner animation for the scanning process
    stop_spinner = False
    
    def run_spinner():
        i = 0
        while not stop_spinner:
            print(f"\r{CYAN}Scanning Python files... {SPINNER_CHARS[i % len(SPINNER_CHARS)]}{NC}", end="")
            i += 1
            time.sleep(0.1)
    
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(target=run_spinner)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    # Get stats about the code base
    code_stats_cmd = f"find {project_dir} -name '*.py' | wc -l"
    files_count_result = run_command(code_stats_cmd, shell=True, capture_output=True)
    files_count = int(files_count_result.stdout.strip()) if files_count_result and files_count_result.stdout else 0
    
    lines_count_cmd = f"find {project_dir} -name '*.py' -exec cat {{}} \\; | wc -l"
    lines_count_result = run_command(lines_count_cmd, shell=True, capture_output=True)
    lines_count = int(lines_count_result.stdout.strip()) if lines_count_result and lines_count_result.stdout else 0
    
    # Stop the spinner
    stop_spinner = True
    spinner_thread.join(0.2)
    print("\r" + " " * 50 + "\r", end="")  # Clear the spinner line
    
    has_critical_issues = False
    has_style_issues = False
    
    # Run detailed style check
    print_colored("\nDetailed style review:", BLUE)
    
    # Start a spinner for detailed style review
    stop_spinner = False
    spinner_thread = threading.Thread(target=run_spinner)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    detailed_cmd = f"{flake8_bin} {project_dir} --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics"
    detailed_result = run_command(detailed_cmd, shell=True, capture_output=True)
    
    # Stop the spinner
    stop_spinner = True
    spinner_thread.join(0.2)
    print("\r" + " " * 50 + "\r", end="")  # Clear the spinner line
    
    if detailed_result and detailed_result.stdout:
        # Check if there are actual issues by looking for line numbers in the output
        if ":" in detailed_result.stdout:
            has_style_issues = True
            print(detailed_result.stdout)
            
            # Count issues by type
            style_issues = detailed_result.stdout.strip().split('\n')
            issue_count = len([line for line in style_issues if ":" in line])
            
            # Summary at the end
            print_colored(f"\n📊 Analysis Summary:", BLUE)
            print_colored(f"   - Files analyzed: {files_count}", GREEN)
            print_colored(f"   - Lines analyzed: {lines_count}", GREEN)
            print_colored(f"   - Critical errors: {'YES' if has_critical_issues else 'No'}", RED if has_critical_issues else GREEN)
            print_colored(f"   - Style issues: {issue_count}", YELLOW if issue_count > 0 else GREEN)
            
            # Calculate error density
            if lines_count > 0:
                error_density = (issue_count / lines_count) * 1000  # Issues per 1000 lines
                print_colored(f"   - Error density: {error_density:.2f} per 1000 lines", 
                             YELLOW if error_density > 5 else GREEN)
        else:
            print_colored("✅ No style issues found", GREEN)
    else:
        print_colored("❌ Failed to run flake8 for style review", RED)
    
    return True


def wait_for_confirmation():
    """Wait for user to press ENTER to continue"""
    print_colored("\n⏸️ Task completed", BLUE)
    print_colored("   Press ENTER to return to main menu or Ctrl+C to exit...", YELLOW)
    input()


def get_default_images_dir(project_dir):
    """Get the default images directory path"""
    return os.path.join(project_dir, "images")


def run_augmentation(config, project_dir):
    """Run the image augmentation script"""
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
    valid_images = []
    
    if os.path.exists(images_dir):
        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and os.path.splitext(file)[1] in valid_extensions:
                    # Store relative path from images_dir for better display
                    rel_path = os.path.relpath(file_path, images_dir)
                    valid_images.append(rel_path)
    
    # Show available images in the images directory if any
    if valid_images:
        print_colored("\nAvailable images in the default directory:", GREEN)
        for i, file in enumerate(valid_images):
            print(f"  {i+1}. {file}")
        print_colored(f"  Total: {len(valid_images)} images found", YELLOW)
    else:
        print_colored(f"\nNo valid images found in {images_dir}", RED)
        print_colored("Please add some images to the directory and try again.", YELLOW)
        return False
    
    # Ask user for the image path to augment
    print_colored("\n🖼️ Image Augmentation Configuration", BLUE)
    print_colored("Enter the path to the image you want to augment", GREEN)
    print_colored(f"(or just the image name if it's in the default {images_dir} directory):", GREEN)
    print_colored("Leave empty to select random images.", YELLOW)
    image_input = input().strip()
    
    # Handle empty input - random selection mode
    if not image_input:
        if not valid_images:
            print_colored("❌ No valid images found in the directory. Cancelling augmentation.", RED)
            return False
            
        print_colored("\nRandom selection mode activated.", GREEN)
        print_colored("Enter the number of random images to process (or 'all' for all images):", GREEN)
        num_input = input().strip().lower()
        
        if num_input == 'all':
            images_to_process = valid_images
            print_colored(f"Processing all {len(images_to_process)} images", GREEN)
        else:
            try:
                num_images = int(num_input)
                if num_images <= 0:
                    print_colored("❌ Number must be positive. Cancelling augmentation.", RED)
                    return False
                    
                if num_images > len(valid_images):
                    print_colored(f"Requested {num_images} images but only {len(valid_images)} are available.", YELLOW)
                    print_colored(f"Will process all {len(valid_images)} available images instead.", YELLOW)
                    num_images = len(valid_images)
                    
                # Randomly select images
                import random
                images_to_process = random.sample(valid_images, num_images)
                print_colored(f"Randomly selected {len(images_to_process)} images for processing:", GREEN)
                for i, img in enumerate(images_to_process):
                    print(f"  {i+1}. {img}")
                    
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
    # Get project directory to define the output directory path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(augmentation_script)))
    
    # Create images_augmented directory
    output_dir = os.path.join(project_dir, "images_augmented")
    os.makedirs(output_dir, exist_ok=True)
    
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


def run_distribution(config, project_dir):
    """Run the Distribution.py script to analyze file distribution"""
    distribution_script = os.path.join(project_dir, "src", "Distribution.py")
    plots_dir = os.path.join(project_dir, "plot")

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
        if progress_thread.is_alive():
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


def main():
    clear_screen()
    
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


if __name__ == "__main__":
    main()
  