import os

# Path to train directory
TRAIN_DIR = "/home/wfscontrol/Downloads/used_cars/train"

# Path to the list of image filenames to delete
DELETE_LIST_PATH = "delete_list.txt"

# Read the filenames from the list
with open(DELETE_LIST_PATH, 'r', encoding='utf-8') as f:
    filenames = [line.strip() for line in f if line.strip()]

deleted = []
not_found = []

# Process each filename
for filename in filenames:
    if not filename.endswith('.jpg'):
        continue

    try:
        # Parse class folder and image number
        class_name = "_".join(filename.split("_")[:-1])
        class_folder = os.path.join(TRAIN_DIR, class_name)
        image_path = os.path.join(class_folder, filename)

        if os.path.exists(image_path):
            os.remove(image_path)
            deleted.append(image_path)
        else:
            not_found.append(image_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"✅ Deleted {len(deleted)} images.")
if not_found:
    print(f"❗ Could not find {len(not_found)} images:")
    for nf in not_found:
        print(nf)
