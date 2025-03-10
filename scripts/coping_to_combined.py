from pathlib import Path
import shutil
from tqdm import tqdm
import random
import csv


def collect_files(folder, file_set):
    for item in folder.rglob("*"):
        if item.is_file():
            file_set.add(item)
    print(f"Total number of files in {folder.name}: {len(file_set)}")


def main():

    p = Path(
        "/home/t.afanasyeva/research_storage/Processing/Lab - Van Dam/datasets/srf_anaemias/CytPix/processed"
    )
    p_out = Path("/home/t.afanasyeva/deep_learning_anaemias")
    combined_folder = p_out / "resources/cytpix/combined"
    combined_folder.mkdir(exist_ok=True)

    for folder in tqdm(p.rglob("24-711122_sorted")):
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                new_combined_folder = combined_folder / subfolder.name
                new_combined_folder.mkdir(exist_ok=True)

    discocyte_set = set()
    echinocyte_set = set()
    granular_set = set()
    holly_leaf_set = set()
    sickle_set = set()
    all_sets = [
        ("discocyte", discocyte_set),
        ("echinocyte", echinocyte_set),
        ("granular", granular_set),
        ("holly_leaf", holly_leaf_set),
        ("sickle", sickle_set),
    ]

    # Collect all the files in the subfolders
    for folder in tqdm(p.rglob("*_sorted")):
        print(f"Folder name: {folder.name}")
        for subfolder in tqdm(folder.iterdir()):
            if subfolder.is_dir():
                print(f"Subfolder name: {subfolder.name}")
                if "discocyte" in subfolder.name:
                    collect_files(subfolder, discocyte_set)
                elif "echinocyte" in subfolder.name:
                    collect_files(subfolder, echinocyte_set)
                elif "granular" in subfolder.name:
                    collect_files(subfolder, granular_set)
                elif "holly_leaf" in subfolder.name:
                    collect_files(subfolder, holly_leaf_set)
                elif "sickle" in subfolder.name:
                    collect_files(subfolder, sickle_set)

    # # Write the total counts to a csv file
    # output_csv_path = p_out / "output/250205_total_counts.csv"
    # with open(output_csv_path, mode="w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Cell Type", "Total Count"])
    #     for cell_type, files in all_sets:
    #         print(f"Total number of {cell_type} files: {len(files)}")
    #         writer.writerow([cell_type, len(files)])

    # Shuffle the files in the sets and copy them to the combined folder
    for cell_type, files in tqdm(all_sets):
        for image_file in tqdm(files):
            shutil.copy(
                image_file,
                combined_folder / cell_type / image_file.name,
            )


if __name__ == "__main__":
    main()
