import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]
CATEGORY = "can"
BASE_PATH = ROOT / "datasets" / "MVTecAD2" / CATEGORY

def create_manifest(dir_path: Path, save_path: Path, mode: str):
    if not dir_path.exists():
        raise FileNotFoundError(f"directory {dir_path} not found")

    data = {}

    if mode == 'train' or mode == 'validation':
        files = list(dir_path.glob("*.png"))
        total_files = len(files)

        if total_files == 0:
            raise FileNotFoundError("the directory is empty!")

        file_paths = [f"datasets/MVTecAD2/{CATEGORY}/{mode}/good/{count:03d}_regular.png" for count in range(total_files)]
        data = {
                'Image_ID': list(range(total_files)),
                'Path_to_PNG': file_paths,
                'Label': [0] * total_files,
                'Domain': ['Normal'] * total_files
            }

    if mode == 'test_public':
        all_file_paths, all_labels, all_domains = [], [], []

        for subfolder in ['good', 'bad']:
            subfolder_path = dir_path / subfolder
            if not subfolder_path.exists():
                raise FileNotFoundError(f"subfolder path {subfolder_path} not found")

            files = list(subfolder_path.glob("*.png"))
            total_files = len(files)
            if total_files == 0:
                raise FileNotFoundError(f"the directory {subfolder_path} is empty!")
            
            total_sets = total_files // 6

            base = f"datasets/MVTecAD2/{CATEGORY}/{mode}/{subfolder}"

            domain_mapping = {
                'Regular': [f"{base}/{count:03d}_regular.png" for count in range(total_sets)],
                'Overexposed': [f"{base}/{count:03d}_overexposed.png" for count in range(total_sets)],
                'Underexposed': [f"{base}/{count:03d}_underexposed.png" for count in range(total_sets)],
                'Shift_1': [f"{base}/{count:03d}_shift_1.png" for count in range(total_sets)],
                'Shift_2': [f"{base}/{count:03d}_shift_2.png" for count in range(total_sets)],
                'Shift_3': [f"{base}/{count:03d}_shift_3.png" for count in range(total_sets)]
            }


            for domain, file in domain_mapping.items():
                all_file_paths.extend(file)
                all_domains.extend([domain] * len(file))
                all_labels.extend([0 if subfolder == 'good' else 1] * len(file))

        data = {
            'Image_ID': list(range(len(all_file_paths))),
            'Path_to_PNG': all_file_paths,
            'Label': all_labels,
            'Domain': all_domains
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"{mode} manifest created for {CATEGORY} at path {save_path}")

if __name__ == '__main__':

    for i in ['train', 'validation', 'test_public']:
        dir = BASE_PATH / i / "good" if i == 'train' or i == 'validation' else BASE_PATH / i
        save = ROOT / "results" / "artifacts" / f"{i}_manifest.csv"
        create_manifest(dir, save, i)

    