import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------
# MLflow Setup
# ----------------------------
mlflow.set_experiment("Tourism_Prediction_Experiment")

# ----------------------------
# Load dataset
# ----------------------------
Xtrain = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/ytrain.csv").values.ravel()
ytest = pd.read_csv("hf://datasets/Satyanjay/tourism-package-prediction-CICD/ytest.csv").values.ravel()

# ----------------------------
# Feature groups
# ----------------------------
numeric_features = [
    'Age','CityTier','DurationOfPitch','NumberOfPersonVisiting',
    'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips',
    'Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    'TypeofContact','Occupation','Gender',
    'ProductPitched','MaritalStatus','Designation'
]

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ----------------------------
# Model + Hyperparameter Grid
# ----------------------------
model = xgb.XGBClassifier(random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1]
}

pipeline = make_pipeline(preprocessor, model)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# ----------------------------
# MLflow Run
# ----------------------------
with mlflow.start_run():

    # Train
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions
    train_pred = best_model.predict(Xtrain)
    test_pred = best_model.predict(Xtest)

    # Metrics
    train_acc = accuracy_score(ytrain, train_pred)
    test_acc = accuracy_score(ytest, test_pred)

    # Log params
    mlflow.log_params(grid_search.best_params_)

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    # Log classification report
    report = classification_report(ytest, test_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # ----------------------------
    # Model Versioning
    # ----------------------------
    model_version = f"tourism_model_v{mlflow.active_run().info.run_id[:5]}"
    model_path = f"{model_version}.joblib"

    joblib.dump(best_model, model_path)

    # ----------------------------
    # Upload to Hugging Face
    # ----------------------------
    repo_id = "Satyanjay/tourism-package-prediction-CICD-model"
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print("Best Params:", grid_search.best_params_)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
