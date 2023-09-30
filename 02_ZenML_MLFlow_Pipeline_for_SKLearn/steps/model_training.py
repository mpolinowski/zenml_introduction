import logging
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step

from model.model_dev import LinearRegressionModel
from .config import ModelNameConfig

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Perform model training
    
    Args:
        X_train = pd.DataFrame of cleaned training features
        y_train = pd.Series of training labels
    Returns:
        Trained model
    """
    
    try:
        model = None
            
        # debugging
        logging.debug("DEBUG :: Model Name: {}".format(config.model_name))

        if config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)

            return trained_model

        else:
            raise ValueError("Model {} not supported".format(config.model_name))
        
    except Exception as e:
        logging.error("ERROR :: Model training failed: {}".format(e))
        raise e

# import logging

# #import mlflow
# import pandas as pd
# from model.model_dev import (
#     HyperparameterTuner,
#     #LightGBMModel,
#     LinearRegressionModel,
#     RandomForestModel,
#     #XGBoostModel,
# )
# from sklearn.base import RegressorMixin
# from zenml import step
# # from zenml.client import Client

# from .config import ModelNameConfig

# # experiment_tracker = Client().active_stack.experiment_tracker


# # @step(experiment_tracker=experiment_tracker.name)
# @step
# def train_model(
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_test: pd.DataFrame,
#     y_test: pd.Series,
#     config: ModelNameConfig,
# ) -> RegressorMixin:
#     """
#     Args:
#         X_train: pd.DataFrame
#         X_test: pd.DataFrame
#         y_train: pd.Series
#         y_test: pd.Series
#     Returns:
#         model: RegressorMixin
#     """
#     try:
#         model = None
#         tuner = None

#         # debug
#         logging.info("DEBUG :: Select model: {}".format(config.model_name))

#         if config.model_name == "linear_regression":
#             # mlflow.sklearn.autolog()
#             model = LinearRegressionModel()
#         elif config.model_name == "randomforest":
#             #mlflow.sklearn.autolog()
#             model = RandomForestModel()
#         elif config.model_name == "xgboost":
#             #mlflow.xgboost.autolog()
#             model = XGBoostModel()
#         elif config.model_name == "lightgbm":
#             #mlflow.lightgbm.autolog()
#             model = LightGBMModel()
#         else:
#             raise ValueError("ERROR :: Model {} not supported".format(config.model_name))

#         # tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)

#         if config.fine_tuning:
#             # best_params = tuner.optimize()
#             trained_model = model.train(X_train, y_train, **best_params)
#         else:
#             # debug
#             logging.info("DEBUG :: Start training model: {}".format(config.model_name))
#             logging.info("DEBUG :: X_train dtype: {}".format(type(X_train)))
#             logging.info("DEBUG :: y_train dtype: {}".format(type(y_train)))

#             trained_model = model.train(X_train, y_train)

#         return trained_model

#     except Exception as e:
#         logging.error("ERROR :: Model training failed: {}".format(e))
#         raise e
