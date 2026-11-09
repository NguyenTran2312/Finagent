from pathlib import Path

from huggingface_hub import snapshot_download

from config.settings import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = CONFIG["dataset"]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / DATASET_CONFIG["cache_dir"]


def download_dataset(local_dir: Path = DATA_DIR) -> str:
    return snapshot_download(
        repo_id=DATASET_CONFIG["repo_id"],
        repo_type="dataset",
        cache_dir=CACHE_DIR,
        local_dir=local_dir,
    )


if __name__ == "__main__":
    downloaded_path = download_dataset()
    print(f"ViFinQA dataset downloaded to: {downloaded_path}")
