import pandas as pd
import os
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------
# Load dataset from HF
# ----------------------------
Xtrain = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/ytrain.csv")
ytest = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/ytest.csv")

ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

# ----------------------------
# Identify columns
# ----------------------------
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ----------------------------
# Model
# ----------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

pipeline = make_pipeline(preprocessor, model)

# ----------------------------
# Train
# ----------------------------
pipeline.fit(Xtrain, ytrain)

# ----------------------------
# Evaluate
# ----------------------------
train_pred = pipeline.predict(Xtrain)
test_pred = pipeline.predict(Xtest)

print("TRAIN REPORT")
print(classification_report(ytrain, train_pred))

print("TEST REPORT")
print(classification_report(ytest, test_pred))

# ----------------------------
# Save model
# ----------------------------
model_path = "tourism_model.joblib"
joblib.dump(pipeline, model_path)

# ----------------------------
# Upload to Hugging Face Model Hub
# ----------------------------
repo_id = "Satyanjay/tourism-package-prediction-CICD-model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Model repo exists")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Model repo created")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
