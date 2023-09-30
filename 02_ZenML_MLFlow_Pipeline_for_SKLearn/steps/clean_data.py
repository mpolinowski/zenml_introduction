import logging
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
from zenml import step

from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Perform data pre-processing and perform train/test split

    Args:
        df: pd.DataFrame of raw data
    Returns:
        X_train = pd.DataFrame of cleaned training features
        X_test = pd.DataFrame of cleaned testing features
        y_train = pd.Series of training labels
        y_test = pd.Series testing of labels
    """
    
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_df = data_cleaning.handle_data()
        logging.info("INFO :: Data cleaning completed")

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_df, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("INFO :: Data split completed")
        
        # debugging
        logging.debug("DEBUG :: X_train dtype: {}".format(type(X_train)))
        logging.debug("DEBUG :: X_test dtype: {}".format(type(X_test)))
        logging.debug("DEBUG :: y_train dtype: {}".format(type(y_train)))
        logging.debug("DEBUG :: y_test dtype: {}".format(type(y_test)))

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error("ERROR :: Data cleaning failed: {}".format(e))
        raise e

@step
def only_split(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Perform train/test split

    Args:
        df: pd.DataFrame of raw data
    Returns:
        X_train = pd.DataFrame of cleaned training features
        X_test = pd.DataFrame of cleaned testing features
        y_train = pd.Series of training labels
        y_test = pd.Series testing of labels
    """
    
    try:
        process_strategy = DataPreprocessStrategy()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(df, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("INFO :: Data split completed")
        
        # debugging
        logging.debug("DEBUG :: X_train dtype: {}".format(type(X_train)))
        logging.debug("DEBUG :: X_test dtype: {}".format(type(X_test)))
        logging.debug("DEBUG :: y_train dtype: {}".format(type(y_train)))
        logging.debug("DEBUG :: y_test dtype: {}".format(type(y_test)))

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error("ERROR :: Data splitting failed: {}".format(e))
        raise e