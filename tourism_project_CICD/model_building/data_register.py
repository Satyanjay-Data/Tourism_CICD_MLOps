from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "Satyanjay/tourism-package-prediction-CICD"
repo_type = "dataset"

# 🔹 Get token
token = os.getenv("HF_TOKEN")

if token is None:
    raise ValueError("❌ HF_TOKEN is missing! Add it to GitHub Secrets.")

# 🔹 Initialize API
api = HfApi(token=token)

# 🔹 Check if dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset '{repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f"⚠️ Dataset '{repo_id}' not found. Creating...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=token   # ✅ CRITICAL FIX
    )

    print(f"✅ Dataset '{repo_id}' created.")

# 🔹 Upload folder
api.upload_folder(
    folder_path="tourism_project_CICD/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("✅ Upload completed successfully.")
