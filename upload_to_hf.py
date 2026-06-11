import os
from huggingface_hub import HfApi, create_repo, login

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")

login(HF_TOKEN)
api = HfApi()

DATASET_REPO = "research-simulation26/cicids2017-gnn"
MODEL_REPO = "research-simulation26/cybersage-cicids2017"
LOCAL_DIR = "/content/drive/MyDrive/csir"

dataset_files = [
    "cicids2017_final.parquet",
    "scaler.pkl",
    "label_encoder_grouped.pkl",
    "cicids_graph.pt",
    "X_explain.pkl",
    "y_explain.pkl",
    "shap_values.pkl",
]

model_files = [
    "best_model_v2.pt",
]

def upload_repo(repo_id, files, repo_type):
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repo {repo_id} ready")
    except Exception as e:
        print(f"Repo creation: {e}")

    for fname in files:
        path = os.path.join(LOCAL_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"  Done ({os.path.getsize(path) / 1e6:.1f} MB)")

if __name__ == "__main__":
    print("=== Uploading to Dataset Repo ===")
    upload_repo(DATASET_REPO, dataset_files, "dataset")

    print("\n=== Uploading to Model Repo ===")
    upload_repo(MODEL_REPO, model_files, "model")

    print("\nAll uploads complete!")
