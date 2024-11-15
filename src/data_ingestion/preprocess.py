# src/data_ingestion/preprocess.py

import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT  # Import the project root path

logger = get_logger(__name__)


def clean_stock_data(df, coverage_threshold=0.5):
    """
    Cleans the stock data by handling missing values.

    Parameters:
    - df (DataFrame): Raw stock data with missing values.
    - coverage_threshold (float): Minimum percentage of non-NA values for each stock to be retained (0.0 to 1.0).

    Returns:
    - DataFrame: Cleaned stock data with minimal missing values.
    """
    logger.info("Cleaning stock data.")

    # Drop stocks with too many missing values based on coverage threshold
    min_non_na_count = int(coverage_threshold * len(df))
    df = df.dropna(thresh=min_non_na_count, axis=1)
    logger.info(f"Stocks retained after applying coverage threshold of {coverage_threshold}: {df.shape[1]}")

    # Fill remaining missing values with forward and backward filling
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Reset the index for consistent output
    df = df.reset_index()

    logger.info(f"Stock data cleaned. Final shape: {df.shape}")
    return df


def save_processed_data(df, filename='stock_prices_cleaned.csv'):
    """
    Saves processed data to the data/processed directory.
    """
    processed_data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    processed_data_path = os.path.join(processed_data_dir, filename)
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")


def load_processed_data(filename='stock_prices_cleaned.csv'):
    """
    Loads processed data from the data/processed directory.
    """
    processed_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', filename)
    if not os.path.exists(processed_data_path):
        logger.error(f"File {processed_data_path} does not exist.")
        return None
    df = pd.read_csv(processed_data_path, parse_dates=['Date'])
    logger.info(f"Processed data loaded from {processed_data_path}")
    return df
