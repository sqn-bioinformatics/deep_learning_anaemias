from pathlib import Path
import shutil
from tqdm import tqdm
import csv
from datetime import datetime


def collect_files(folder, file_set):
    for item in folder.rglob("*"):
        if item.is_file():
            file_set.add(item)
    print(f"Total number of files in {folder.name}: {len(file_set)}")


def main(path_in:str, path_out:str) --> DataFrame:
    path_in = Path(path_in)
    path_out = Path(path_out)
    path_out.mkdir(exist_ok=True)

    cell_types = ['discocyte', "echinocyte", "granular", "holly_leaf", "sickle"]
    all_sets = {cell_type: set() for cell_type in cell_types}

    # Collect all the files in the subfolders
    for cell_type in cell_types:
        # Ensure the output directory for the cell type exists
        output_dir = path_out / cell_type
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cell type is {cell_type}")
        for folder in tqdm(path_in.rglob("*_sorted")):
            print(f"Folder name: {folder.name}")
            for subfolder in tqdm(folder.iterdir()):
                if subfolder.is_dir():
                    if cell_type in subfolder.name:
                        collect_files(subfolder, all_sets[cell_type])

    # Write the total counts to a csv file
    current_date = datetime.now().strftime("%y%m%d")
    output_csv_path = p_out / "output"/ f"{current_date}_total_counts.csv"
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Cell Type", "Total Count"])
        for cell_type, files in all_sets:
            print(f"Total number of {cell_type} files: {len(files)}")
            writer.writerow([cell_type, len(files)])

    # Copy files into the combined folder
    for cell_type, files in tqdm(all_sets):
        for image_file in tqdm(files):
            output_dir = path_out / cell_type
            for image_file in tqdm(files):
                shutil.copy(
                image_file,
                output_dir / image_file.name,
                )

if __name__ == "__main__":
    main()
