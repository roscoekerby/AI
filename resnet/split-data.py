import os
import shutil
import random

# Define paths for the original cat and dog images
cat_image_dir = r'C:\Users\RoscoeKerby\PycharmProjects\resnet\cats&dogs\cat'
dog_image_dir = r'C:\Users\RoscoeKerby\PycharmProjects\resnet\cats&dogs\dog'

# Define base output directories for train, val, and test splits
base_output_dir = r'C:\Users\RoscoeKerby\PycharmProjects\resnet\cats&dogs'
output_train_dir = os.path.join(base_output_dir, 'train')
output_val_dir = os.path.join(base_output_dir, 'val')
output_test_dir = os.path.join(base_output_dir, 'test')

# Create cat and dog subdirectories within train, val, and test directories
for split_dir in [output_train_dir, output_val_dir, output_test_dir]:
    os.makedirs(os.path.join(split_dir, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'dog'), exist_ok=True)

# Function to split and move files into train, val, and test directories
def split_and_move_files(image_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Get list of all images
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(images)

    # Calculate exact counts for each split
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)
    test_count = len(images) - train_count - val_count  # Ensures total sum matches

    # Split images into train, val, and test sets
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Move images to respective directories
    for image_file in train_images:
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(train_dir, image_file))
    for image_file in val_images:
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(val_dir, image_file))
    for image_file in test_images:
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(test_dir, image_file))

    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")

# Split and move files for cats and dogs, organizing them into train/val/test folders with cat/dog subfolders
print("Processing cats:")
split_and_move_files(cat_image_dir,
                     os.path.join(output_train_dir, 'cat'),
                     os.path.join(output_val_dir, 'cat'),
                     os.path.join(output_test_dir, 'cat'))

print("\nProcessing dogs:")
split_and_move_files(dog_image_dir,
                     os.path.join(output_train_dir, 'dog'),
                     os.path.join(output_val_dir, 'dog'),
                     os.path.join(output_test_dir, 'dog'))
