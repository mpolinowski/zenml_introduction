import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from data_path.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to csv datafile
        Returns:
            pd.DataFrame of ingested data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from data_path.
        """
        logging.info(f'INFO :: Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from data_path.

    Args:
        data_path: path to csv datafile
    Returns:
        pd.DataFrame of ingested data
    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        # debugging
        logging.debug("DEBUG :: Data input shape: {}".format(df.shape))
        return df
    except Exception as e:
        logging.error(f'ERROR :: Failed to ingest data: {e}')
        raise(e)