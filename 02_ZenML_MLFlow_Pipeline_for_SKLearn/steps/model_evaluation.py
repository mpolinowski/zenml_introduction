import logging
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from model.model_eval import MSE, RMSE, R2

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "rmse"],
    Annotated[float, "r2"]
]:
    """
    Evaluate the trained model on test dataset
    
    Args:
        df: pd.DataFrame of ingested data
    """
    
    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)

        return mse, rmse, r2

    except Exception as e:
        logging.error("ERROR :: Calculating eval scores failed: {}".format(e))
        raise e