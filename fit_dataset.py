import os
from opencv_face_detection import *
from PIL import Image

def resize_and_save_image(src, dst, size=(64, 64)):
    # Open the image
    with Image.open(src) as img:
        # Resize the image
        img_resized = img.resize(size, Image.ANTIALIAS)
        # Save the resized image
        img_resized.save(dst)

def count_face_images(folder_path):
    face_counts = {}

    print("Counting face images in each subfolder...")
    # Iterate through each subfolder in the main folder
    for name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, name)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            count = 0

            # Count the number of JPG files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.lower().endswith('.jpg'):
                    count += 1

            face_counts[name] = count

    return face_counts

# Example usage
folder_path = 'lfw_funneled'
face_counts = count_face_images(folder_path)
print("Face counts in each subfolder:", face_counts)

# Filter the dictionary to include only those with 2 or more photos
filtered_face_counts = {name: count for name, count in face_counts.items() if count >= 2}
print("Filtered counts (2 or more photos):", filtered_face_counts)

keys = filtered_face_counts.keys()

# Iterate over keys and process files
for key in keys:
    source_folder = f"lfw_funneled/{key}"
    destination_folder = f"lfw_filtered/{key}"

    # Check if source folder exists
    if os.path.exists(source_folder):
        # Create the destination folder
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created folder: {destination_folder}")
for key in keys:
    source_folder = f"lfw_funneled/{key}"
    destination_folder = f"lfw_filtered/{key}"

    # Check if source folder exists
    if os.path.exists(source_folder):
        # Create the destination folder
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created folder: {destination_folder}")

        # Iterate over all files in the source folder
        for file_name in os.listdir(source_folder):
            src_path = os.path.join(source_folder, file_name)
            dst_path = os.path.join(destination_folder, file_name)

            # Resize and save each file
            detect_and_save_face(src_path, "./tmp/tmp_img.jpg")
            resize_and_save_image("./tmp/tmp_img.jpg", dst_path)
            print(f"Processed and resized: {dst_path}")
    else:
        print(f"Folder {source_folder} does not exist")