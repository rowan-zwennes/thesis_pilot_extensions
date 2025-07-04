import pathlib
import joblib
import sys

from benchmark_config import NEW_DATASET_NAMES
from benchmark_util_new import load_data

DATAFOLDER = pathlib.Path(__file__).parent / "Data"
DATAFOLDER.mkdir(exist_ok=True)

# Downloads the data and stores it in 'Data' folder
for repo_id in NEW_DATASET_NAMES:
    filename = DATAFOLDER / f"other_{repo_id}.pkl"
    if filename.exists():
        print(f"{filename.name} already exists, skipping")
        continue
    dataset = load_data(
        repo_id,
        use_download=False,
        kind="other"
    )
    
    print(f"Storing data in {filename}")
    joblib.dump(dataset, filename)
    

