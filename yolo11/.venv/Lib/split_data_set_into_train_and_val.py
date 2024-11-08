import os
import shutil
import random

# Define paths for images and labels
image_dir = r'C:\Users\RoscoeKerby\PycharmProjects\yolo11\.venv\Lib\BostonDogs'
label_dir = r'C:\Users\RoscoeKerby\PycharmProjects\yolo11\.venv\Lib\labels_dog-muzzle_2024-10-17-01-46-51'
output_train_dir = r'C:\Users\RoscoeKerby\PycharmProjects\yolo11\.venv\Lib\train'
output_val_dir = r'C:\Users\RoscoeKerby\PycharmProjects\yolo11\.venv\Lib\val'

# Create output directories if they don't exist
os.makedirs(output_train_dir + '/images', exist_ok=True)
os.makedirs(output_train_dir + '/labels', exist_ok=True)
os.makedirs(output_val_dir + '/images', exist_ok=True)
os.makedirs(output_val_dir + '/labels', exist_ok=True)

# Get list of all images
images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(images)

# Define split ratio
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

# Split images into train and val sets
train_images = images[:split_index]
val_images = images[split_index:]

# Function to move files
def move_files(file_list, source_dir, dest_dir_images, dest_dir_labels):
    for image_file in file_list:
        image_path = os.path.join(source_dir, image_file)
        label_file = image_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        if os.path.exists(label_path):
            # Move image and label to respective train or val directories
            shutil.copy(image_path, os.path.join(dest_dir_images, image_file))
            shutil.copy(label_path, os.path.join(dest_dir_labels, label_file))

# Move train images and labels
move_files(train_images, image_dir, output_train_dir + '/images', output_train_dir + '/labels')

# Move val images and labels
move_files(val_images, image_dir, output_val_dir + '/images', output_val_dir + '/labels')

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
