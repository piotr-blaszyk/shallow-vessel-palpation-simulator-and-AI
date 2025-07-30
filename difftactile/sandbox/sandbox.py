import os
import re

markers_folder = "difftactile/output/training_data/markers"
masks_folder = "difftactile/output/training_data/segmentation_mask"


def rename_files_in_folder(folder_path, old_pattern, new_pattern):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    files = os.listdir(folder_path)
    pattern = re.compile(old_pattern.format(r"(\d+)", r"(\d+)"))
    for file in files:
        match = pattern.match(file)
        if match:
            num1, num2 = match.groups()
            new_name = new_pattern.format(num1, num2)
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {file}: {str(e)}")


print("\nProcessing markers folder...")
rename_files_in_folder(markers_folder, "markers_{}_{}.png", "image_{}_{}.png")
print("\nProcessing masks folder...")
rename_files_in_folder(masks_folder, "segmentation_mask_{}_{}.png", "image_{}_{}.png")
print("\nRenaming process completed!")
