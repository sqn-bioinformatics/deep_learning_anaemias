from pathlib import Path
import shutil
from tqdm import tqdm
import time

p = Path(
    "/home/t.afanasyeva/research_storage/Processing/Lab - Van Dam/datasets/srf_anaemias/CytPix/processed"
)

names_to_check = ["24-Ramos-4122"]

start_time = time.time()
paths = [
    x
    for x in p.iterdir()
    if x.is_dir() and any(name in x.name for name in names_to_check)
]
print(f"Time taken to collect paths: {time.time() - start_time:.2f} seconds")

# Check folder "name" and put all paths in set and printout the number
name_folder = p / names_to_check[0]
print(f"Checking folder: {name_folder}")

start_time = time.time()
name_files = set()
for file in tqdm(name_folder.glob("**/*"), desc="Collecting files"):
    if file.is_file():
        name_files.add(file.name)
print(f"Number of files in {name_folder.name}: {len(name_files)}")
print(
    f"Time taken to collect files in {name_folder.name}: {time.time() - start_time:.2f} seconds"
)

# Check remaining folders in path get all the names and put them in a set
start_time = time.time()
remaining_files = set()


def collect_files(folder, file_set):
    for item in folder.rglob("*"):
        if item.is_file():
            file_set.add(item.name)
    print(f"Total number of files in {folder.name}: {len(file_set)}")


for folder in tqdm(p.iterdir(), desc="Checking subfolders"):
    if (
        folder.is_dir()
        and any(name in folder.name for name in names_to_check)
        and folder != name_folder
    ):
        print(f"Checking subfolder: {folder.name}")
        collect_files(folder, remaining_files)

print(f"Number of remaining files: {len(remaining_files)}")
print(f"Time taken to collect remaining files: {time.time() - start_time:.2f} seconds")

# Remove the second set from the first one this is the thirst set
start_time = time.time()
missing_files = name_files - remaining_files
print(f"Number of missing files: {len(missing_files)}")
print(f"Time taken to find missing files: {time.time() - start_time:.2f} seconds")

# Copy all the files of the fird set to the folder called "name_discocytes"
destination_folder = p / f"{names_to_check[0]}_discocytes"
destination_folder.mkdir(exist_ok=True)

start_time = time.time()
for file_name in tqdm(missing_files, desc="Copying files"):
    source_file = name_folder / file_name
    destination_file = destination_folder / file_name
    shutil.copy2(source_file, destination_file)
print(f"Copied {len(missing_files)} missing files to {destination_folder}")
print(f"Time taken to copy missing files: {time.time() - start_time:.2f} seconds")
