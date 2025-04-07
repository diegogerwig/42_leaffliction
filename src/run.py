#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

# Clear terminal screen at the beginning
def clear_screen():
    """Clear the terminal screen based on the OS"""
    if platform.system() == "Windows":
        os.system('cls')
    else:  # Linux and macOS
        os.system('clear')

# Clear the screen immediately
clear_screen()

# Colors for better output
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
NC = '\033[0m'  # No Color

def print_colored(text, color):
    """Print colored text"""
    print(f"{color}{text}{NC}")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# Define paths
distribution_script = os.path.join(script_dir, "Distribution.py")
plots_dir = os.path.join(project_dir, "plot")

print_colored("=== Starting Distribution Analysis ===", BLUE)

# Check if running in a virtual environment - multiple detection methods
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

# Additional checks for conda environments
env_path = os.environ.get('CONDA_PREFIX') or os.environ.get('VIRTUAL_ENV')
conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')

if in_venv or env_path or conda_env_name:
    env_name = conda_env_name or (os.path.basename(env_path) if env_path else "unknown")
    print_colored(f"Virtual environment detected: {env_name}", GREEN)
else:
    print_colored("ERROR: No virtual environment detected!", RED)
    print_colored("This script requires a virtual environment with the necessary dependencies.", RED)
    print_colored("Please activate your virtual environment and try again:", YELLOW)
    print_colored("    conda activate mlp_env", YELLOW)
    print_colored("Or:", YELLOW)
    print_colored("    source ~/mlp_env/bin/activate", YELLOW)
    sys.exit(1)

# Run the Distribution.py script
print_colored("Running Distribution.py script...", GREEN)
try:
    # Run the script directly with the current Python interpreter
    process = subprocess.Popen([sys.executable, distribution_script], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check for errors
    if process.returncode != 0:
        stderr = process.stderr.read()
        print_colored("Error running Distribution.py:", RED)
        print(stderr)
        print_colored("\nDistribution.py failed to run. Please check if all dependencies are installed.", RED)
        print_colored("Try activating your virtual environment: conda activate mlp_env", YELLOW)
        sys.exit(1)
    
except Exception as e:
    print_colored(f"Error: {e}", RED)
    sys.exit(1)

# Check if any plots were generated
print_colored("Checking for generated plots...", GREEN)
if os.path.isdir(plots_dir):
    # Get all PNG files in the plots directory
    plots = [os.path.join(plots_dir, f) for f in os.listdir(plots_dir) 
             if f.endswith('.png')]
    
    if not plots:
        print_colored(f"No plots were generated in {plots_dir}", RED)
        sys.exit(1)
    
    print_colored(f"Found {len(plots)} plots. Opening...", GREEN)
    
    # Linux-specific way to open plots
    for plot in plots:
        try:
            subprocess.Popen(['xdg-open', plot])
            print_colored(f"Opened: {os.path.basename(plot)}", GREEN)
        except Exception as e:
            print_colored(f"Could not open {plot}: {e}", RED)
            print_colored(f"You can manually view the plots at: {plots_dir}", GREEN)
else:
    print_colored(f"Plots directory not found at {plots_dir}", RED)

print_colored("=== Distribution Analysis Complete ===", BLUE)