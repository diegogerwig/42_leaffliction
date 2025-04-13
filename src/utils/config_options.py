import os
import sys
import subprocess
import threading
import random
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, run_command, run_progress_spinner, 
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def get_script_options(script_type):
    """
    Gets and processes options for scripts according to their type.
    
    Args:
        script_type: Script type ('transformation' or 'augmentation')
        
    Returns:
        list: List of additional arguments for the script
    """
    if script_type == 'transformation':
        print_colored("\n🔧 Image Transformation Configuration", BLUE)
        print_colored("Select transformation options:", GREEN)
        print_colored("1. Grayscale", YELLOW)
        print_colored("2. Edge detection", YELLOW)
        print_colored("3. Blur", YELLOW)
        print_colored("4. Sharpen", YELLOW)
        print_colored("5. Binary (Black and White)", YELLOW)
        print_colored("6. Contrast enhancement", YELLOW)
        print_colored("7. Leaf mask", YELLOW)
        print_colored("8. ROI objects", YELLOW)
        print_colored("9. Leaf analysis", YELLOW)
        print_colored("10. Pseudolandmarks", YELLOW)
        print_colored("0. All transformations", YELLOW)
        
        transform_options = input("Enter transformation numbers to apply (e.g. 1,3,5) or 0 for all: ").strip()
        
        # Process transformation options
        extra_args = []
        if transform_options == "0":
            extra_args.append("--all")
        else:
            # Map input numbers to transformation options
            option_map = {
                "1": "--grayscale",
                "2": "--edges",
                "3": "--blur",
                "4": "--sharpen",
                "5": "--binary",
                "6": "--contrast",
                "7": "--mask",
                "8": "--roi",
                "9": "--analyze",
                "10": "--landmarks"
            }
            
            selected_options = transform_options.split(",")
            for opt in selected_options:
                opt = opt.strip()
                if opt in option_map:
                    extra_args.append(option_map[opt])
        
        if not extra_args:
            print_colored("❌ No valid options selected. Using defaults (all).", YELLOW)
            extra_args.append("--all")
            
        return extra_args
        
    elif script_type == 'augmentation':
        # For the augmentation script, there are currently no additional options
        return []
    
    return []