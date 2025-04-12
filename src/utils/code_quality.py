#!/usr/bin/env python3
import os
import sys
import time
import threading


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, run_command, run_progress_spinner, 
    wait_for_confirmation, get_default_images_dir,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


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


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    print_colored("This script is intended to be imported, not run directly.", YELLOW)
    print_colored("For testing, please import this module in another script.", YELLOW)