import os

# Define merging rules
MERGE_CLASS_MAP = {
    'K5_하이브리드_3세대_2020_2023': 'K5_3세대_하이브리드_2020_2022',
    '디_올_뉴_니로_2022_2025': '디_올뉴니로_2022_2025',
    '박스터_718_2017_2024': '718_박스터_2017_2024',
    '라브4_4세대_2013_2018': 'RAV4_2016_2018',
    '라브4_5세대_2019_2024': 'RAV4_5세대_2019_2024',
}

def get_class_name_from_folder(folder_name):
    return MERGE_CLASS_MAP.get(folder_name, folder_name)

def rename_images(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    class_name = get_class_name_from_folder(folder_name)

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

    for idx, fname in enumerate(image_files):
        old_path = os.path.join(folder_path, fname)
        new_name = f"{class_name}_{idx:04d}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"{fname} → {new_name}")

# Example usage
rename_images("/home/wfscontrol/Downloads/used_cars/train/아반떼_N_2022_2023")