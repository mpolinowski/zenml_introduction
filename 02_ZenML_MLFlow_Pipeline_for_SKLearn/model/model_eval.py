from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Class for model evaluation
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate model training scores
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Return:
            None
        """

        pass

class MSE(Evaluation):
    """
    Evaluation strategy using the Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("INFO :: Calculating MSE score")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("INFO :: MSE score: {}".format(mse))
            return mse

        except Exception as e:
            logging.error("ERROR :: Calculating MSE failed: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy using the Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("INFO :: Calculating RMSE score")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("INFO :: RMSE score: {}".format(rmse))
            return rmse

        except Exception as e:
            logging.error("ERROR :: Calculating RMSE failed: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation strategy using the R2 score
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("INFO :: Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("INFO :: R2 score: {}".format(r2))
            return r2

        except Exception as e:
            logging.error("ERROR :: Calculating R2 score failed: {}".format(e))
            raise e