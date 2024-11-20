# src/feature_engineering/screening.py

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def sector_specific_screening(financial_data, scores_df, tickers_df, num_long, num_short, available_tickers):
    """
    Assigns scores to stocks and selects stocks for long and short positions.

    Parameters:
    - financial_data (DataFrame): DataFrame containing financial metrics.
    - scores_df (DataFrame): DataFrame containing Ticker and Composite Score.
    - tickers_df (DataFrame): DataFrame containing 'Ticker' and 'Sector' columns.
    - num_long (int): Number of stocks to long in each sector.
    - num_short (int): Number of stocks to short in each sector.
    - available_tickers (set): Set of tickers available in the cleaned price data.

    Returns:
    - long_tickers (DataFrame): DataFrame of tickers selected for long positions.
    - short_tickers (DataFrame): DataFrame of tickers selected for short positions.
    """
    logger.info("Ranking stocks within each sector based on composite scores.")

    # Filter scores_df to include only available tickers
    scores_df = scores_df[scores_df['Ticker'].isin(available_tickers)]

    # Merge scores with sector information
    scores_with_sector = scores_df.merge(tickers_df[['Ticker', 'Sector']], on='Ticker')

    long_positions = []
    short_positions = []

    sectors = scores_with_sector['Sector'].unique()

    for sector in sectors:
        sector_stocks = scores_with_sector[scores_with_sector['Sector'] == sector]

        # Check if there are enough stocks in the sector
        if len(sector_stocks) < (num_long + num_short):
            logger.warning(f"Not enough stocks in sector '{sector}' for the specified number of long and short positions.")
            # Adjust num_long and num_short proportionally
            total_positions = len(sector_stocks)
            adjusted_num_long = int((num_long / (num_long + num_short)) * total_positions)
            adjusted_num_short = total_positions - adjusted_num_long
            num_long_sector = max(adjusted_num_long, 1)
            num_short_sector = max(adjusted_num_short, 1)
        else:
            num_long_sector = num_long
            num_short_sector = num_short

        sector_stocks = sector_stocks.sort_values(by='Composite Score', ascending=False)

        # Select top N stocks for long positions
        top_stocks = sector_stocks.head(num_long_sector)
        long_positions.append(top_stocks)

        # Select bottom N stocks for short positions
        bottom_stocks = sector_stocks.tail(num_short_sector)
        short_positions.append(bottom_stocks)

    # Concatenate the results
    long_tickers = pd.concat(long_positions).reset_index(drop=True)
    short_tickers = pd.concat(short_positions).reset_index(drop=True)

    logger.info(f"Selected {len(long_tickers)} stocks for long positions.")
    logger.info(f"Selected {len(short_tickers)} stocks for short positions.")

    return long_tickers, short_tickers
