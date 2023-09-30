from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_training import train_model
from steps.model_evaluation import evaluate_model


@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
