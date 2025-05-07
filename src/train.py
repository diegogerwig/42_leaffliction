#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import shutil
import zipfile
import datetime
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import utility functions
from utils.utils import (
    print_colored, extract_source_category, GREEN, BLUE, RED, YELLOW, CYAN, NC
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a leaf disease classification model')
    parser.add_argument('directory', type=str, help='Directory containing leaf disease images')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--output', type=str, default='leaffliction_model', help='Output zip file name without extension')
    parser.add_argument('--use_tqdm', type=str, default='False', help='Use tqdm for progress display')
    
    return parser.parse_args()

def create_class_directories(base_dir, train_dir, val_dir):
    """Create directories for training and validation datasets."""
    # Get all subdirectories (classes) in the base directory
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Create directories for each class in train and validation directories
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    return classes

def split_data(base_dir, train_dir, val_dir, validation_split=0.2):
    """Split data into training and validation sets."""
    # Get all subdirectories (classes) in the base directory
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    dataset_info = {
        "classes": classes,
        "distribution": {},
        "total_images": 0,
        "training_images": 0,
        "validation_images": 0
    }
    
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        # Count total images per class
        dataset_info["distribution"][class_name] = len(images)
        dataset_info["total_images"] += len(images)
        
        # Split images into training and validation sets
        train_images, val_images = train_test_split(images, test_size=validation_split, random_state=42)
        
        # Count images in each split
        dataset_info["training_images"] += len(train_images)
        dataset_info["validation_images"] += len(val_images)
        
        # Copy images to their respective directories
        for img in train_images:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(train_dir, class_name, img)
            )
        
        for img in val_images:
            shutil.copy(
                os.path.join(class_dir, img),  # Corregido: Eliminado el class_name duplicado
                os.path.join(val_dir, class_name, img)
            )
    
    return dataset_info

def augment_image(image):
    """Apply augmentation to a single image."""
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random rotation between -45 and 45 degrees
    angle = np.random.uniform(-45, 45)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height),
                          flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    
    # Random brightness change
    brightness_factor = np.random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    return image

def create_augmented_images(train_dir, augmented_dir, n_augmented=5):
    """Create augmented images for each original image."""
    # Get all classes in the training directory
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Create augmented directories
    for class_name in classes:
        os.makedirs(os.path.join(augmented_dir, class_name), exist_ok=True)
        
        # Get all images in the class directory
        class_dir = os.path.join(train_dir, class_name)
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            # Read the image
            img = cv2.imread(img_path)
            
            if img is None:
                print_colored(f"Warning: Could not read {img_path}", YELLOW)
                continue
            
            # Generate augmented images
            for i in range(n_augmented):
                # Apply augmentation
                augmented_img = augment_image(img.copy())
                
                # Save the augmented image
                name, ext = os.path.splitext(img_file)
                aug_filename = f"{name}_aug_{i}{ext}"
                aug_path = os.path.join(augmented_dir, class_name, aug_filename)
                cv2.imwrite(aug_path, augmented_img)
                
    # Copy original images to augmented directory as well
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        aug_class_dir = os.path.join(augmented_dir, class_name)
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                shutil.copy(
                    os.path.join(class_dir, img_file),
                    os.path.join(aug_class_dir, img_file)
                )

def create_model(input_shape, num_classes):
    """Create a convolutional neural network model."""
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten layer
        Flatten(),
        
        # Fully connected layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_dir, val_dir, epochs=50, batch_size=32, use_tqdm='False'):
    """Train the model using the prepared datasets."""
    # Create data generators with data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Calculate steps per epoch for tqdm
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    # Determine if we should use tqdm
    use_tqdm_progress = use_tqdm.lower() == 'true'
    
    if use_tqdm_progress:
        try:
            from tqdm import tqdm
            import tensorflow as tf
            from tensorflow import keras
            
            # Custom callback for tqdm progress bar
            class TQDMProgressBar(keras.callbacks.Callback):
                def on_train_begin(self, logs=None):
                    print("Training started...")
                    
                def on_epoch_begin(self, epoch, logs=None):
                    print(f"\nEpoch {epoch+1}/{epochs}")
                    self.train_progbar = tqdm(total=steps_per_epoch, desc="Training", 
                                            position=0, leave=True, 
                                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
                    
                def on_train_batch_end(self, batch, logs=None):
                    # Update the progress bar with batch metrics
                    self.train_progbar.update(1)
                    self.train_progbar.set_postfix({
                        'loss': f"{logs.get('loss', 0):.4f}",
                        'acc': f"{logs.get('accuracy', 0):.4f}"
                    })
                    
                def on_epoch_end(self, epoch, logs=None):
                    # Close the progress bar
                    self.train_progbar.close()
                    
                    # Print epoch results
                    print(f"Epoch {epoch+1}/{epochs} - loss: {logs.get('loss', 0):.4f} - "
                          f"accuracy: {logs.get('accuracy', 0):.4f} - "
                          f"val_loss: {logs.get('val_loss', 0):.4f} - "
                          f"val_accuracy: {logs.get('val_accuracy', 0):.4f}")
                    
                    # Check validation set size and accuracy
                    if logs.get('val_accuracy', 0) < 0.90:
                        print("Warning: Validation accuracy is below 90%. Project requires at least 90% accuracy.")
                    
                def on_train_end(self, logs=None):
                    print("\nTraining completed!")
            
            # Use our custom callback
            callbacks = [early_stopping, TQDMProgressBar()]
            verbose_mode = 0  # Suppress default progress bar
        except ImportError:
            print("Warning: tqdm package not found. Using default progress display.")
            callbacks = [early_stopping]
            verbose_mode = 1  # Use default progress bar
    else:
        callbacks = [early_stopping]
        verbose_mode = 1  # Use default progress bar
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose_mode
    )
    
    # Create class mapping
    class_indices = train_generator.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    
    # Check the validation set size
    val_count = 0
    for _, _, files in os.walk(val_dir):
        val_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
    
    if val_count < 100:
        print(f"Warning: Validation set contains only {val_count} images. Project requires at least 100 validation images.")
    
    return history, class_mapping

def evaluate_model(model, val_dir, batch_size=32):
    """Evaluate the model on the validation set."""
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate the model
    evaluation = model.evaluate(val_generator)
    
    # Get predictions
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = val_generator.classes
    
    # Calculate accuracy
    accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
    
    return {
        'loss': evaluation[0],
        'accuracy': evaluation[1],
        'calculated_accuracy': accuracy,
        'correct_predictions': np.sum(predicted_classes == true_classes),
        'total_predictions': len(true_classes)
    }

def create_zip_archive(model, class_mapping, dataset_info, evaluation_results, model_summary, train_dir, augmented_dir, output_name):
    """Create a zip archive containing the model and augmented images."""
    # Create a temporary directory to store files
    temp_dir = os.path.join(os.getcwd(), 'temp_model_files')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(temp_dir, 'leaf_disease_model.h5')
    model.save(model_path)
    
    # Save class mapping
    class_mapping_path = os.path.join(temp_dir, 'class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(class_mapping, f)
    
    # Save dataset info
    dataset_info_path = os.path.join(temp_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f)
    
    # Save evaluation results
    evaluation_path = os.path.join(temp_dir, 'evaluation_results.json')
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_results, f)
    
    # Save model summary
    summary_path = os.path.join(temp_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(model_summary)
    
    # Save date and time of training
    timestamp_path = os.path.join(temp_dir, 'timestamp.txt')
    with open(timestamp_path, 'w') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Create a readme file
    readme_path = os.path.join(temp_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Leaf Disease Classification Model\n\n")
        f.write("## Model Information\n")
        f.write(f"- Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Number of Classes: {len(class_mapping)}\n")
        f.write(f"- Classes: {', '.join(dataset_info['classes'])}\n")
        f.write(f"- Validation Accuracy: {evaluation_results['accuracy'] * 100:.2f}%\n\n")
        f.write("## Files\n")
        f.write("- `leaf_disease_model.h5`: Trained model file\n")
        f.write("- `class_mapping.json`: Mapping between class indices and class names\n")
        f.write("- `dataset_info.json`: Information about the dataset used for training\n")
        f.write("- `evaluation_results.json`: Results of model evaluation on validation set\n")
        f.write("- `model_summary.txt`: Summary of the model architecture\n")
        f.write("- `augmented_images/`: Directory containing augmented images used for training\n\n")
        f.write("## Usage\n")
        f.write("Use `predict.py` to predict diseases from new leaf images.")
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a zip file with the model files and augmented images
    output_zip = f"{output_name}.zip"
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model files
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
        
        # Add augmented images
        for root, _, files in os.walk(augmented_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join('augmented_images', os.path.relpath(file_path, augmented_dir))
                zipf.write(file_path, arcname)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return output_zip

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Get the input directory
    input_dir = os.path.abspath(args.directory)
    
    print_colored(f"Starting training with data from: {input_dir}", GREEN)
    
    # Check if the directory exists
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print_colored(f"Error: Directory {input_dir} does not exist!", RED)
        return 1
    
    # Create directories for training, validation, and augmented data
    temp_base_dir = os.path.join(os.getcwd(), 'temp_data')
    train_dir = os.path.join(temp_base_dir, 'train')
    val_dir = os.path.join(temp_base_dir, 'validation')
    augmented_dir = os.path.join(temp_base_dir, 'augmented')
    
    # Clean up existing directories if they exist
    for dir_path in [train_dir, val_dir, augmented_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # Create class directories
    classes = create_class_directories(input_dir, train_dir, val_dir)
    print_colored(f"Found {len(classes)} classes: {', '.join(classes)}", GREEN)
    
    # Split data into training and validation sets
    print_colored("Splitting data into training and validation sets...", BLUE)
    dataset_info = split_data(input_dir, train_dir, val_dir, args.validation_split)
    
    print_colored(f"Total images: {dataset_info['total_images']}", GREEN)
    print_colored(f"Training images: {dataset_info['training_images']}", GREEN)
    print_colored(f"Validation images: {dataset_info['validation_images']}", GREEN)
    
    # Create augmented images
    print_colored("Creating augmented images...", BLUE)
    create_augmented_images(train_dir, augmented_dir)
    
    # Create the model
    print_colored("Creating model...", BLUE)
    input_shape = (224, 224, 3)  # Standard input shape for model
    num_classes = len(classes)
    model = create_model(input_shape, num_classes)
    
    # Get model summary as string
    model_summary = ""
    model.summary(print_fn=lambda x: model_summary + x + "\n")
    
    # Train the model
    print_colored(f"Training model for {args.epochs} epochs with batch size {args.batch_size}...", BLUE)
    history, class_mapping = train_model(
        model, augmented_dir, val_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        use_tqdm=args.use_tqdm
    )
    
    # Evaluate the model
    print_colored("Evaluating model on validation set...", BLUE)
    evaluation_results = evaluate_model(model, val_dir, args.batch_size)
    
    print_colored(f"Validation accuracy: {evaluation_results['accuracy'] * 100:.2f}%", GREEN)
    
    if evaluation_results['accuracy'] < 0.9:
        print_colored("Warning: Validation accuracy is below 90%!", YELLOW)
    
    # Create a zip archive with model and augmented images
    print_colored("Creating zip archive...", BLUE)
    output_zip = create_zip_archive(
        model, 
        class_mapping, 
        dataset_info, 
        evaluation_results,
        model_summary,
        train_dir, 
        augmented_dir,
        args.output
    )
    
    print_colored(f"Training completed! Files saved to: {output_zip}", GREEN)
    
    # Clean up
    shutil.rmtree(temp_base_dir)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print_colored(f"Error: {e}", RED)
        sys.exit(1)