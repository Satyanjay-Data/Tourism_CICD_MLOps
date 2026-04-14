import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

folder_path = "tourism_project_CICD/deployment"

api.upload_folder(
    folder_path=folder_path,
    repo_id="Satyanjay/Tourism-Package-Prediction-CICD",
    repo_type="space",
)
