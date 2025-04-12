#!/usr/bin/env python3
import os
import time
import subprocess

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
    """Print text with color"""
    print(f"{color}{text}{NC}")


def run_command(command, shell=False, capture_output=False):
    """Run a shell command and handle errors"""
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


def wait_for_confirmation():
    """Wait for user to press ENTER to continue"""
    print_colored("\n⏸️ Task completed", BLUE)
    print_colored("   Press ENTER to return to main menu or Ctrl+C to exit...", YELLOW)
    input()


def get_default_images_dir(project_dir):
    """Get the default images directory path"""
    return os.path.join(project_dir, "images")