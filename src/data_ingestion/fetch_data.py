# src/data_ingestion/fetch_data.py

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT
from src.utils.config_loader import load_config  # Load the configuration file

logger = get_logger(__name__)


def get_sp500_tickers_with_sector():
    """
    Scrapes the S&P 500 tickers and their respective sectors from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 tickers and sectors from {url}")

    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch S&P 500 data. Status code: {response.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})

    if table:
        from io import StringIO  # Import StringIO inside the function or at the top
        df = pd.read_html(StringIO(str(table)))[0]
        df = df[['Symbol', 'GICS Sector']]
        df.rename(columns={'Symbol': 'Ticker', 'GICS Sector': 'Sector'}, inplace=True)
    else:
        logger.error("Failed to find the S&P 500 table on Wikipedia.")
        df = pd.DataFrame()

    logger.info(f"Retrieved {len(df)} S&P 500 tickers with sectors.")
    return df


def fetch_historical_data(tickers_df=None, start_date=None, end_date=None):
    """
    Fetches historical adjusted close prices for the given tickers within the date range.
    """
    # Load configuration dates if not provided
    if start_date is None or end_date is None:
        config = load_config()
        start_date = start_date or config['data']['start_date']
        end_date = end_date or config['data']['end_date']

    # Fetch tickers from Wikipedia if not provided
    if tickers_df is None:
        tickers_df = get_sp500_tickers_with_sector()

    # Extract tickers from the DataFrame
    tickers_list = tickers_df['Ticker'].tolist()

    # Filter tickers to include only valid strings
    filtered_tickers = [ticker for ticker in tickers_list if isinstance(ticker, str)]

    # Log any non-string tickers that were removed
    removed_tickers = [ticker for ticker in tickers_list if not isinstance(ticker, str)]
    if removed_tickers:
        logger.warning(f"Removed non-string tickers: {removed_tickers}")

    # Define the directory and file path using PROJECT_ROOT
    filename = f"historical_data_{start_date}_{end_date}.csv"
    raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    filepath = os.path.join(raw_data_dir, filename)

    # Check if the data file already exists
    if os.path.exists(filepath):
        logger.info(f"Loading historical data from {filepath}")
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return data, tickers_df  # Ensure consistent return values

    # Fetch data from Yahoo Finance
    logger.info(f"Fetching historical data from Yahoo Finance for tickers: {filtered_tickers}")
    data = yf.download(filtered_tickers, start=start_date, end=end_date)['Adj Close']

    # Ensure the directory exists before saving
    os.makedirs(raw_data_dir, exist_ok=True)
    data.to_csv(filepath)
    logger.info(f"Historical data saved to {filepath}")

    return data, tickers_df


def save_raw_data(data, filename='stock_prices_raw.csv'):
    """
    Saves raw data to the data/raw directory.
    """
    # Define the directory and file path using PROJECT_ROOT
    raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = os.path.join(raw_data_dir, filename)

    # Save the data
    data.to_csv(file_path)
    logger.info(f"Raw data saved to {file_path}")

def load_raw_data(filename='stock_prices_raw.csv'):
    """
    Loads raw data from the data/raw directory.
    """
    # Define the file path using PROJECT_ROOT
    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', filename)

    # Check if the file exists before loading
    if not os.path.exists(raw_data_path):
        logger.error(f"File {raw_data_path} does not exist.")
        return None

    # Load and return the data
    data = pd.read_csv(raw_data_path, index_col='Date', parse_dates=True)
    logger.info(f"Raw data loaded from {raw_data_path}")
    return data
