#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import threading
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.venv_manager import setup_and_activate_environment, clean_directories
from utils.code_quality import run_flake8
from utils.utils import (
    print_colored, run_command, run_progress_spinner, 
    wait_for_confirmation, get_default_images_dir,
    GREEN, BLUE, RED, YELLOW, CYAN, NC
)
from utils.image_utils import find_valid_images, process_images_batch
from utils.config_options import get_script_options


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


# def get_script_options(script_type):
#     """
#     Obtiene y procesa opciones para los scripts según su tipo.
    
#     Args:
#         script_type: Tipo de script ('transformation' o 'augmentation')
        
#     Returns:
#         list: Lista de argumentos adicionales para el script
#     """
#     if script_type == 'transformation':
#         print_colored("\n🔧 Configuración de Transformación de Imágenes", BLUE)
#         print_colored("Seleccione opciones de transformación:", GREEN)
#         print_colored("1. Escala de grises", YELLOW)
#         print_colored("2. Detección de bordes", YELLOW)
#         print_colored("3. Desenfoque", YELLOW)
#         print_colored("4. Enfoque", YELLOW)
#         print_colored("5. Binario (Blanco y Negro)", YELLOW)
#         print_colored("6. Mejora de contraste", YELLOW)
#         print_colored("7. Máscara de hoja", YELLOW)
#         print_colored("8. Objetos ROI", YELLOW)
#         print_colored("9. Análisis de hojas", YELLOW)
#         print_colored("10. Pseudolandmarks", YELLOW)
#         print_colored("0. Todas las transformaciones", YELLOW)
        
#         transform_options = input("Ingrese números de transformaciones a aplicar (ej. 1,3,5) o 0 para todas: ").strip()
        
#         # Procesar opciones de transformación
#         extra_args = []
#         if transform_options == "0":
#             extra_args.append("--all")
#         else:
#             # Mapear números de entrada con opciones de transformación
#             option_map = {
#                 "1": "--grayscale",
#                 "2": "--edges",
#                 "3": "--blur",
#                 "4": "--sharpen",
#                 "5": "--binary",
#                 "6": "--contrast",
#                 "7": "--mask",
#                 "8": "--roi",
#                 "9": "--analyze",
#                 "10": "--landmarks"
#             }
            
#             selected_options = transform_options.split(",")
#             for opt in selected_options:
#                 opt = opt.strip()
#                 if opt in option_map:
#                     extra_args.append(option_map[opt])
        
#         if not extra_args:
#             print_colored("❌ No se seleccionaron opciones válidas. Usando predeterminadas (todas).", YELLOW)
#             extra_args.append("--all")
            
#         return extra_args
        
#     elif script_type == 'augmentation':
#         # Para el script de augmentación no hay opciones adicionales actualmente
#         return []
    
#     return []


def run_transformation(config, project_dir):
    """
    Execute the transformation script on selected images.
    
    Args:
        config: Environment configuration
        project_dir: Project directory
        
    Returns:
        bool: True if execution was successful, False otherwise
    """
    print_colored("\n=== Image Transformation Tool ===", BLUE)
    
    # Check if the script exists
    script_dir = os.path.join(project_dir, "src")
    transformation_script = os.path.join(script_dir, "Transformation.py")
    
    if not os.path.exists(transformation_script):
        print_colored(f"❌ Transformation.py not found at {transformation_script}", RED)
        return False
    
    # Configure Python executable according to environment
    if config['use_conda']:
        python_exe = os.path.join(config['env_path'], 'bin', 'python')
    else:
        python_exe = config['python_bin']
    
    # Base image directory
    images_dir = os.path.join(project_dir, "images")
    
    # Create transformed images directory
    transformed_dir = os.path.join(project_dir, "images_transformed")
    os.makedirs(transformed_dir, exist_ok=True)
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print_colored(f"Images directory not found at {images_dir}. Creating now...", YELLOW)
        os.makedirs(images_dir, exist_ok=True)
        print_colored(f"✅ Images directory created: {images_dir}", GREEN)
    
    # Get list of valid images
    all_valid_images, subdir_images = find_valid_images(images_dir)
    
    # Show available images
    if all_valid_images:
        print_colored("\nAvailable images in default directory:", GREEN)
        for i, file in enumerate(all_valid_images):
            print(f"  {i+1}. {file}")
        print_colored(f"  Total: {len(all_valid_images)} images found", YELLOW)
        
        if len(subdir_images) > 1:  # Only show subdirectory info if there are multiple
            print_colored("\nImages by subdirectory:", GREEN)
            for subdir, images in subdir_images.items():
                print(f"  {subdir}: {len(images)} images")
    else:
        print_colored(f"\nNo valid images found in {images_dir}", RED)
        print_colored("Please add some images to the directory and try again.", YELLOW)
        return False
    
    # Ask user for directory to use
    print_colored(f"\nEnter directory to transform (or press ENTER to use default '{images_dir}'):", GREEN)
    image_dir_input = input().strip()
    
    # Determine directory to use
    target_dir = images_dir
    if image_dir_input:
        if os.path.isabs(image_dir_input) and os.path.exists(image_dir_input):
            target_dir = image_dir_input
        elif os.path.exists(os.path.join(images_dir, image_dir_input)):
            target_dir = os.path.join(images_dir, image_dir_input)
        else:
            print_colored(f"❌ Directory not found: {image_dir_input}. Using default directory: {images_dir}", YELLOW)
    
    print_colored(f"Using directory: {target_dir}", GREEN)
    
    # Find images in selected directory again
    target_images, target_subdir_images = find_valid_images(target_dir)
    
    if not target_images:
        print_colored(f"❌ No images found in {target_dir}", RED)
        return False
    
    # Show number of images in subdirectories
    if len(target_subdir_images) > 1:
        print_colored("\nImage distribution in selected directory:", GREEN)
        for subdir, images in target_subdir_images.items():
            print(f"  {subdir}: {len(images)} images")
    
    # Ask how many images to process
    print_colored("\nEnter the number of images to process (or 'all' to process all):", GREEN)
    num_images_input = input().strip().lower()
    
    # Get transformation options
    extra_args = get_script_options('transformation')
    
    # Process the batch of images
    return process_images_batch(
        python_exe,
        transformation_script,
        target_dir,
        transformed_dir,
        num_images_input if num_images_input else 'all',
        extra_args
    )


def run_augmentation(config, project_dir):
    """
    Ejecuta el script de augmentación en imágenes seleccionadas.
    
    Args:
        config: Configuración del entorno
        project_dir: Directorio del proyecto
        
    Returns:
        bool: True si la ejecución fue exitosa, False en caso contrario
    """
    print_colored("\n=== Herramienta de Augmentación de Imágenes ===", BLUE)
    
    # Verificar si existe el script
    script_dir = os.path.join(project_dir, "src")
    augmentation_script = os.path.join(script_dir, "Augmentation.py")
    
    if not os.path.exists(augmentation_script):
        print_colored(f"❌ No se encontró Augmentation.py en {augmentation_script}", RED)
        return False
    
    # Configurar el ejecutable de Python según el entorno
    if config['use_conda']:
        python_exe = os.path.join(config['env_path'], 'bin', 'python')
    else:
        python_exe = config['python_bin']
    
    # Directorio base de imágenes
    images_dir = os.path.join(project_dir, "images")
    
    # Crear directorio de imágenes aumentadas
    augmented_dir = os.path.join(project_dir, "images_augmented")
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Verificar si existe el directorio de imágenes
    if not os.path.exists(images_dir):
        print_colored(f"Directorio de imágenes no encontrado en {images_dir}. Creándolo ahora...", YELLOW)
        os.makedirs(images_dir, exist_ok=True)
        print_colored(f"✅ Directorio de imágenes creado: {images_dir}", GREEN)
    
    # Obtener lista de imágenes válidas
    all_valid_images, subdir_images = find_valid_images(images_dir)
    
    # Mostrar imágenes disponibles
    if all_valid_images:
        print_colored("\nImágenes disponibles en el directorio predeterminado:", GREEN)
        for i, file in enumerate(all_valid_images):
            print(f"  {i+1}. {file}")
        print_colored(f"  Total: {len(all_valid_images)} imágenes encontradas", YELLOW)
        
        if len(subdir_images) > 1:  # Solo mostrar info de subdirectorios si hay múltiples
            print_colored("\nImágenes por subdirectorio:", GREEN)
            for subdir, images in subdir_images.items():
                print(f"  {subdir}: {len(images)} imágenes")
    else:
        print_colored(f"\nNo se encontraron imágenes válidas en {images_dir}", RED)
        print_colored("Por favor, añada algunas imágenes al directorio e intente de nuevo.", YELLOW)
        return False
    
    # Solicitar al usuario el directorio a utilizar
    print_colored(f"\nIngrese el directorio a aumentar (o presione ENTER para usar el predeterminado '{images_dir}'):", GREEN)
    image_dir_input = input().strip()
    
    # Determinar el directorio a usar
    target_dir = images_dir
    if image_dir_input:
        if os.path.isabs(image_dir_input) and os.path.exists(image_dir_input):
            target_dir = image_dir_input
        elif os.path.exists(os.path.join(images_dir, image_dir_input)):
            target_dir = os.path.join(images_dir, image_dir_input)
        else:
            print_colored(f"❌ Directorio no encontrado: {image_dir_input}. Usando directorio predeterminado: {images_dir}", YELLOW)
    
    print_colored(f"Usando directorio: {target_dir}", GREEN)
    
    # Volver a buscar imágenes en el directorio seleccionado
    target_images, target_subdir_images = find_valid_images(target_dir)
    
    if not target_images:
        print_colored(f"❌ No se encontraron imágenes en {target_dir}", RED)
        return False
    
    # Mostrar cantidad de imágenes en subdirectorios
    if len(target_subdir_images) > 1:
        print_colored("\nDistribución de imágenes en el directorio seleccionado:", GREEN)
        for subdir, images in target_subdir_images.items():
            print(f"  {subdir}: {len(images)} imágenes")
    
    # Preguntar cuántas imágenes procesar
    print_colored("\nIngrese el número de imágenes a procesar (o 'all' para procesar todas):", GREEN)
    num_images_input = input().strip().lower()
    
    # Procesar el lote de imágenes
    return process_images_batch(
        python_exe,
        augmentation_script,
        target_dir,
        augmented_dir,
        num_images_input if num_images_input else 'all',
        []
    )


def show_menu(config, project_dir):
    """Show the main menu and handle user selection"""
    while True:
        try:
            print_colored("\n=== Machine Learning Project Tools ===", BLUE)
            print_colored("1. Run code quality check (flake8)", GREEN)
            print_colored("2. Run data distribution analysis", GREEN)
            print_colored("3. Run image augmentation", GREEN)
            print_colored("4. Run image transformation", GREEN)
            print_colored("0. Exit", GREEN)
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
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
            elif choice == '4':
                run_transformation(config, project_dir)
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
            
        # Clean output directories (plots, images_augmented, images_transformed)
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