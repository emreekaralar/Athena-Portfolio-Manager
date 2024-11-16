# src/feature_engineering/screening.py

import pandas as pd
from src.utils.logger import get_logger
from src.data_ingestion.fetch_data import get_sp500_tickers_with_sector
import yfinance as yf
import numpy as np

logger = get_logger(__name__)

# Mapping of screening types to their corresponding functions
SCREENING_FUNCTIONS = {
    "avg_return": "screen_avg_return",
    "momentum": "screen_momentum",
    "valuation": "screen_valuation",
    "dividend_yield": "screen_dividend_yield",
    "volatility": "screen_volatility",
    "earnings_growth": "screen_earnings_growth"
}


def sector_specific_screening(price_data, tickers_df, criteria_config):
    """
    Applies sector-specific screening criteria based on configuration.

    Parameters:
    - price_data (DataFrame): Historical price data with Date as index and tickers as columns.
    - tickers_df (DataFrame): DataFrame containing 'Ticker' and 'Sector' columns.
    - criteria_config (dict): Configuration for screening criteria per sector.

    Returns:
    - filtered_tickers (DataFrame): DataFrame of tickers that pass the screening criteria.
    """
    logger = get_logger(__name__)
    logger.info("Applying sector-specific screening criteria based on configuration.")

    filtered_rows = []

    for _, row in tickers_df.iterrows():
        ticker = row['Ticker']
        sector = row['Sector']
        if sector in criteria_config:
            criterion = criteria_config[sector]
            screening_type = criterion.get('type')
            threshold = criterion.get('threshold')
            screening_function_name = SCREENING_FUNCTIONS.get(screening_type)

            if screening_function_name and screening_function_name in globals():
                screening_function = globals()[screening_function_name]
                metric = screening_function(ticker, price_data)
                if metric is not None and metric > threshold:
                    filtered_rows.append({
                        'Ticker': ticker,
                        'Sector': sector,
                        'Metric': metric
                    })
            else:
                logger.warning(f"Unsupported or undefined screening type '{screening_type}' for sector: {sector}")
        else:
            logger.warning(f"No screening criteria defined for sector: {sector}")

    # Create DataFrame from the list of filtered rows
    filtered_tickers = pd.DataFrame(filtered_rows, columns=['Ticker', 'Sector', 'Metric'])
    logger.info(f"Screening complete. {len(filtered_tickers)} tickers passed the criteria.")
    return filtered_tickers


def screen_avg_return(ticker, price_data):
    """
    Screens based on average return over the last 30 days.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - price_data (DataFrame): Historical price data.

    Returns:
    - avg_return (float): Average return or None.
    """
    try:
        avg_return = price_data[ticker].pct_change().dropna().tail(30).mean()
        return avg_return
    except Exception as e:
        logger.error(f"Error calculating average return for {ticker}: {e}")
        return None


def screen_momentum(ticker, price_data):
    """
    Screens based on 90-day momentum.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - price_data (DataFrame): Historical price data.

    Returns:
    - momentum (float): Momentum or None.
    """
    try:
        momentum = price_data[ticker].pct_change(periods=90).mean()
        return momentum
    except Exception as e:
        logger.error(f"Error calculating momentum for {ticker}: {e}")
        return None


def screen_valuation(ticker, price_data):
    """
    Screens based on P/E ratio.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - price_data (DataFrame): Historical price data.

    Returns:
    - pe_ratio (float): P/E ratio or None.
    """
    try:
        stock = yf.Ticker(ticker)
        pe_ratio = stock.info.get('trailingPE', None)
        if pe_ratio is not None:
            return pe_ratio
        else:
            logger.warning(f"P/E ratio not available for {ticker}.")
            return None
    except Exception as e:
        logger.error(f"Error fetching P/E ratio for {ticker}: {e}")
        return None


def screen_dividend_yield(ticker, price_data):
    """
    Screens based on dividend yield.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - price_data (DataFrame): Historical price data.

    Returns:
    - dividend_yield (float): Dividend yield or None.
    """
    try:
        stock = yf.Ticker(ticker)
        dividend_yield = stock.info.get('dividendYield', None)
        if dividend_yield is not None:
            return dividend_yield
        else:
            logger.warning(f"Dividend yield not available for {ticker}.")
            return None
    except Exception as e:
        logger.error(f"Error fetching dividend yield for {ticker}: {e}")
        return None


def screen_volatility(ticker, price_data):
    """
    Screens based on annualized volatility.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - price_data (DataFrame): Historical price data.

    Returns:
    - annual_volatility (float): Annualized volatility or None.
    """
    try:
        daily_std = price_data[ticker].pct_change().dropna().std()
        annual_volatility = daily_std * np.sqrt(252)
        return annual_volatility
    except Exception as e:
        logger.error(f"Error calculating volatility for {ticker}: {e}")
        return None


def screen_earnings_growth(ticker, price_data):
    """
    Screens based on earnings growth.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - price_data (DataFrame): Historical price data.

    Returns:
    - earnings_growth (float): Earnings growth rate or None.
    """
    try:
        stock = yf.Ticker(ticker)
        earnings_growth = stock.info.get('earningsGrowth', None)
        if earnings_growth is not None:
            return earnings_growth
        else:
            logger.warning(f"Earnings growth not available for {ticker}.")
            return None
    except Exception as e:
        logger.error(f"Error fetching earnings growth for {ticker}: {e}")
        return None
