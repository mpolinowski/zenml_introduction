from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Strategy for handling of data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for pre-processing of data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process data
        """

        try:
            # drop columns of low feature importance
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "customer_zip_code_prefix",
                    "order_item_id"
                ],
                axis=1
            )

            # fill missing data
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            # drop or encode categorical columns -> drop
            data = data.select_dtypes(include=[np.number])
            # there are still a few missing values - I fill them with zeros for now
            data = data.fillna(value=0)
            # for debugging
            data.to_csv('datasets/_data_cleaned.csv')
            return data

        except Exception as e:
            logging.error("ERROR :: Pre-processing data failed: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for splitting of data
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Split data
        """
        try:
            # divide features from labels
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            
            # do a train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
            
            # debugging
            logging.debug("DEBUG :: X_train shape: {}".format(X_train.shape))
            logging.debug("DEBUG :: X_test shape: {}".format(X_test.shape))
            logging.debug("DEBUG :: y_train shape: {}".format(y_train.shape))
            logging.debug("DEBUG :: y_test shape: {}".format(y_test.shape))

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("ERROR :: Train/Test Split failed: {}".format(e))
            raise e


class DataCleaning(DataStrategy):
    """
    Pre-process data and perform a train/test split
    """

    def __init__ (self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("ERROR :: Handling data failed: {}".format(e))
            raise(e)