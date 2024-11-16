# src/data_ingestion/preprocess.py

import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT  # Import the project root path

logger = get_logger(__name__)


def clean_stock_data(data, coverage_threshold=0.5):
    """
    Cleans the stock data by handling missing values and filtering stocks based on coverage threshold.

    Parameters:
    - data (DataFrame): Raw stock price data.
    - coverage_threshold (float): Minimum data coverage required to retain a stock.

    Returns:
    - df (DataFrame): Cleaned stock data.
    """
    logger = get_logger(__name__)
    logger.info("Cleaning stock data.")

    # Calculate the coverage (non-NA values) for each stock
    coverage = data.notna().mean()

    # Filter stocks based on coverage threshold
    filtered_stocks = coverage[coverage >= coverage_threshold].index.tolist()
    logger.info(f"Stocks retained after applying coverage threshold of {coverage_threshold}: {len(filtered_stocks)}")

    # Select the filtered stocks
    df = data[filtered_stocks]

    # Fill missing values using forward and backward fill
    df = df.ffill().bfill()

    # Reset index to ensure 'Date' is a column if needed
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
