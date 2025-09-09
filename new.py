import os

# Path to the main folder
root_dir = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\dev\\Beyond_words\\keypoints\\keypoints"

# Traverse all subfolders in the keypoints directory
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    # Only process if it's a folder
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)
        
        # Sort files for consistency (optional)
        files.sort()

        # Keep the first file, delete the rest
        for file in files[1:]:
            file_path = os.path.join(subfolder_path, file)
            os.remove(file_path)

print("Cleanup complete: kept 1 file per subfolder in 'keypoints/'")
