"""
Builds tryon_mapping.csv for RealTryonDataset from local folders.

Minimum required subfolders under root_dir: person_images/, garment_images/.
Optional subfolders (used directly if present, auto-derived otherwise -
see tryondiffusion/preprocessing.py): ca_images/, person_pose_path/.

Every subfolder present under root_dir becomes a column; files are matched
across folders by shared base filename (a "_keypoints" suffix, as produced
by OpenPose, is stripped before matching).
"""
import os
import pandas as pd

def generate_structured_csv(root_dir, output_csv='tryon_mapping.csv', use_full_path=False):
    folder_file_map = {}
    base_filenames = set()

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        file_map = {}
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path):
                continue

            name, ext = os.path.splitext(file)

            # Special handling for _keypoints.json files
            if name.endswith('_keypoints'):
                base_name = name.replace('_keypoints', '')
            else:
                base_name = name

            value = file_path if use_full_path else file
            file_map[base_name] = value
            base_filenames.add(base_name)

        folder_file_map[folder_name] = file_map

    rows = []
    sorted_base_names = sorted(base_filenames)

    for base_name in sorted_base_names:
        row = {}
        for folder_name, files in folder_file_map.items():
            row[folder_name] = files.get(base_name, '')
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(root_dir, output_csv)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to: {csv_path}")


# Example usage: python csv_mapping.py
# (only needed for the local-folder data path; the Hugging Face data path
# in trainer.py needs no CSV at all)
if '__main__' == __name__:
    generate_structured_csv(root_dir="./data", use_full_path= False)