# src/data_ingestion/fetch_data.py

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import time
from io import StringIO
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

def collect_stocks_data(tickers):
    """
    Collects comprehensive financial data for a list of tickers.

    Parameters:
    - tickers (list): List of ticker symbols.

    Returns:
    - stocks_df (DataFrame): DataFrame containing financial data for the tickers.
    """
    logger.info("Starting collection of financial data for tickers.")
    stocks_data = []

    for ticker in tickers:
        try:
            logger.debug(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.info

            # Extract desired financial metrics
            stock_data = {
                'Ticker': ticker,
                'Sector': data.get('sector', None),
                'Industry': data.get('industry', None),
                'Market Cap': data.get('marketCap', None),
                'Enterprise Value': data.get('enterpriseValue', None),
                'Trailing P/E': data.get('trailingPE', None),
                'Forward P/E': data.get('forwardPE', None),
                'PEG Ratio': data.get('pegRatio', None),
                'Price to Sales Ratio': data.get('priceToSalesTrailing12Months', None),
                'Price to Book Ratio': data.get('priceToBook', None),
                'Enterprise Value/Revenue': data.get('enterpriseToRevenue', None),
                'Enterprise Value/EBITDA': data.get('enterpriseToEbitda', None),
                'Profit Margin': data.get('profitMargins', None),
                'Operating Margin': data.get('operatingMargins', None),
                'Return on Assets': data.get('returnOnAssets', None),
                'Return on Equity': data.get('returnOnEquity', None),
                'Revenue': data.get('totalRevenue', None),
                'Revenue Per Share': data.get('revenuePerShare', None),
                'Quarterly Revenue Growth': data.get('revenueGrowth', None),
                'Gross Profit': data.get('grossProfits', None),
                'EBITDA': data.get('ebitda', None),
                'Net Income': data.get('netIncomeToCommon', None),
                'Diluted EPS': data.get('trailingEps', None),
                'Quarterly Earnings Growth': data.get('earningsGrowth', None),
                'Total Cash': data.get('totalCash', None),
                'Total Cash Per Share': data.get('totalCashPerShare', None),
                'Total Debt': data.get('totalDebt', None),
                'Debt to Equity': data.get('debtToEquity', None),
                'Current Ratio': data.get('currentRatio', None),
                'Book Value Per Share': data.get('bookValue', None),
                'Operating Cash Flow': data.get('operatingCashflow', None),
                'Levered Free Cash Flow': data.get('freeCashflow', None),
                'Beta': data.get('beta', None),
                '52 Week High': data.get('fiftyTwoWeekHigh', None),
                '52 Week Low': data.get('fiftyTwoWeekLow', None),
                '50 Day Moving Average': data.get('fiftyDayAverage', None),
                '200 Day Moving Average': data.get('twoHundredDayAverage', None),
                'Shares Outstanding': data.get('sharesOutstanding', None),
                'Dividend Yield': data.get('dividendYield', None),
                'Payout Ratio': data.get('payoutRatio', None),
                'Ex-Dividend Date': data.get('exDividendDate', None),
                'Short Ratio': data.get('shortRatio', None),
                'Float Shares': data.get('floatShares', None),
                'Avg 10 Day Volume': data.get('averageVolume10days', None),
                'Avg 3 Month Volume': data.get('averageVolume', None),
            }

            # Convert Unix timestamp to datetime for 'Ex-Dividend Date'
            ex_dividend_date = stock_data['Ex-Dividend Date']
            if ex_dividend_date:
                stock_data['Ex-Dividend Date'] = pd.to_datetime(ex_dividend_date, unit='s').date()

            stocks_data.append(stock_data)

            # Sleep to avoid hitting rate limits
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")

    stocks_df = pd.DataFrame(stocks_data)
    logger.info("Completed collection of financial data.")
    return stocks_df

def save_financial_data_to_excel(stocks_df, filename='financial_data.xlsx'):
    """
    Saves the financial data DataFrame to an Excel file.

    Parameters:
    - stocks_df (DataFrame): DataFrame containing financial data.
    - filename (str): Name of the Excel file.
    """
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    stocks_df.to_excel(file_path, index=False)
    logger.info(f"Financial data saved to {file_path}")
