import os

def count_face_images(folder_path):
    face_counts = {}

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
print(face_counts)

# Filter the dictionary to include only those with 2 or more photos
filtered_face_counts = {name: count for name, count in face_counts.items() if count >= 2}

# Print the number of people in the filtered dictionary
print("Number of people with at least 2 photos:", len(filtered_face_counts))