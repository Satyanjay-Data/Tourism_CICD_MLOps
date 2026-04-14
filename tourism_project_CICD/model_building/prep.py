# for data manipulation
import pandas as pd
import os

# for splitting data
from sklearn.model_selection import train_test_split

# for encoding categorical variables
from sklearn.preprocessing import LabelEncoder

# for hugging face upload
from huggingface_hub import HfApi

# -------------------------
# Load dataset
# -------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/Satyanjay/tourism-package-prediction-CICD/tourism.csv"
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Columns:", df.columns)

# -------------------------
# Drop identifier column
# -------------------------
df.drop(columns=['CustomerID'], inplace=True)

# -------------------------
# Handle missing values (important for real data)
# -------------------------
df = df.dropna()

# -------------------------
# Encode categorical columns (from data dictionary)
# -------------------------
label_encoder = LabelEncoder()

categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# -------------------------
# Target column (FROM DATA DICTIONARY)
# -------------------------
target_col = 'ProdTaken'

# -------------------------
# Split features and target
# -------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------
# Train-test split
# -------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Save datasets
# -------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# -------------------------
# Upload to Hugging Face Dataset Repo
# -------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="Satyanjay/tourism-package-prediction-CICD",
        repo_type="dataset",
    )
