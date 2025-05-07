#!/usr/bin/env python3
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, run_command, run_progress_spinner, 
    wait_for_confirmation, get_default_images_dir,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)


def get_config():
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


def create_venv(config):
    print_colored('\n🔆 Creating new virtual environment', BLUE)
    
    if config['use_conda']:
        conda_cmd = f"{config['conda_bin']} create -y -n {config['env_name']} python=3.11"
        return run_command(conda_cmd, shell=True)
    else:
        if not run_command("python3 -m venv --help", shell=True, capture_output=True):
            print_colored("❌ Python venv module not available.", RED)
            return False
            
        os.makedirs(os.path.dirname(config['venv_path']), exist_ok=True)
        result = run_command(f"python3 -m venv {config['venv_path']}", shell=True)
        
        if result:
            print_colored(f"✅ Created virtual environment at {config['venv_path']}", GREEN)
        
        return result


def venv_exists(config):
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
    project_dir = os.path.dirname(os.path.dirname(script_dir))
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
    print_colored('\n🧹 Cleaning output directories', BLUE)
    
    dirs_to_clean = [
        os.path.join(project_dir, "plots"),
        os.path.join(project_dir, "images_augmented"),
        os.path.join(project_dir, "images_transformed"),
        # os.path.join(project_dir, "temp_data"),
        os.path.join(project_dir, "temp_train_data"),
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
        # Modified part: Use os.walk to recursively go through all subdirectories
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                # Skip .gitkeep files
                if name == ".gitkeep":
                    continue
                    
                file_path = os.path.join(root, name)
                try:
                    os.unlink(file_path)
                    cleaned += 1
                except Exception as e:
                    print_colored(f"⚠️ Failed to remove file {file_path}: {e}", YELLOW)
            
            for name in dirs:
                dir_to_remove = os.path.join(root, name)
                try:
                    # Check if there's a .gitkeep file in this directory
                    gitkeep_in_subdir = os.path.join(dir_to_remove, ".gitkeep")
                    if os.path.exists(gitkeep_in_subdir):
                        # If .gitkeep exists, just remove all other files
                        for item in os.listdir(dir_to_remove):
                            if item != ".gitkeep":
                                item_path = os.path.join(dir_to_remove, item)
                                if os.path.isfile(item_path):
                                    os.unlink(item_path)
                                    cleaned += 1
                    else:
                        # If no .gitkeep, remove the entire directory
                        shutil.rmtree(dir_to_remove)
                        cleaned += 1
                except Exception as e:
                    print_colored(f"⚠️ Failed to remove directory {dir_to_remove}: {e}", YELLOW)
        
        # Ensure there's a .gitkeep file in the main directory
        gitkeep_path = os.path.join(dir_path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                pass
                
        print_colored(f"✅ Cleaned {cleaned} items from {dir_path}", GREEN)
    
    return True


def setup_and_activate_environment():
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


if __name__ == "__main__":
    config = get_config()
    print_colored("Testing virtual environment management...", BLUE)
    result, config = setup_and_activate_environment()
    if result:
        print_colored("Virtual environment setup successful!", GREEN)
    else:
        print_colored("Virtual environment setup failed!", RED)