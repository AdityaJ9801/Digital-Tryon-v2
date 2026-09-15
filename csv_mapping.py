"""
Builds tryon_mapping.csv for RealTryonDataset from local folders.

Minimum required subfolders under root_dir: person_images/, garment_images/
(or whatever names you pass to RealTryonDataset's person_folder=/
garment_folder=, e.g. `image`/`cloth` for Kaggle VITON-HD-style dumps).
Optional subfolders (used directly if present, auto-derived otherwise -
see tryondiffusion/preprocessing.py): ca_images/, person_pose_path/.

By default every subfolder present under root_dir becomes a column; pass
include_folders=[...] to only map specific subfolders and ignore the rest
(e.g. a dataset that also ships cloth-mask/, image-parse-v3/,
openpose_img/, agnostic-v3.2/ etc. that this project doesn't need, since
tryondiffusion/preprocessing.py derives the clothing-agnostic image and
pose automatically instead of requiring them precomputed).

Files are matched across folders by shared base filename (a "_keypoints"
suffix, as produced by OpenPose, is stripped before matching).
"""
import os
import pandas as pd

def generate_structured_csv(root_dir, output_csv='tryon_mapping.csv', use_full_path=False, include_folders=None):
    folder_file_map = {}
    base_filenames = set()

    for folder_name in sorted(os.listdir(root_dir)):
        if include_folders is not None and folder_name not in include_folders:
            continue

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
    print(f"CSV saved to: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")


# Example usage:
#   python csv_mapping.py --root ./data
#   python csv_mapping.py --root /path/to/train --include image cloth
# (only needed for the local-folder data path; the Hugging Face data path
# in trainer.py needs no CSV at all)
if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="./data", help="Dataset root directory containing the image subfolders")
    parser.add_argument("--output", default="tryon_mapping.csv", help="Output CSV filename (written inside --root)")
    parser.add_argument(
        "--include", nargs="+", default=None,
        help="Only map these subfolder names (e.g. --include image cloth), ignoring every other "
             "subfolder under --root. Omit to include every subfolder found."
    )
    parser.add_argument("--full-path", action="store_true", help="Store absolute file paths instead of filenames")
    args = parser.parse_args()

    generate_structured_csv(
        root_dir=args.root, output_csv=args.output, use_full_path=args.full_path, include_folders=args.include
    )