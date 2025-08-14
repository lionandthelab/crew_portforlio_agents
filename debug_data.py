#!/usr/bin/env python3
"""
Debug script to check the structure of downloaded market data
"""

import pandas as pd
from src.tools.market_data import load_real_market_data


def debug_data():
    print("=== Debugging Market Data ===")

    # Load real market data
    start_date = "2023-01-01"
    end_date = "2025-07-31"

    try:
        prices, fundamentals = load_real_market_data(
            start_date, end_date, use_real_data=True
        )

        print(f"\n=== Price Data ===")
        print(f"Shape: {prices.shape}")
        print(f"Columns: {prices.columns.tolist()}")
        print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
        print(f"Unique tickers: {len(prices['ticker'].unique())}")
        print(f"Sample tickers: {prices['ticker'].unique()[:5]}")

        # Check for missing values
        print(f"\nMissing values in prices:")
        print(prices.isnull().sum())

        # Sample data
        print(f"\nSample price data:")
        print(prices.head())

        print(f"\n=== Fundamental Data ===")
        print(f"Shape: {fundamentals.shape}")
        print(f"Columns: {fundamentals.columns.tolist()}")
        print(
            f"Date range: {fundamentals['date'].min()} to {fundamentals['date'].max()}"
        )
        print(f"Unique tickers: {len(fundamentals['ticker'].unique())}")

        # Check for missing values
        print(f"\nMissing values in fundamentals:")
        print(fundamentals.isnull().sum())

        # Sample data
        print(f"\nSample fundamental data:")
        print(fundamentals.head())

        # Check sectors
        print(f"\n=== Sector Information ===")
        sectors = fundamentals[["ticker", "sector"]].drop_duplicates()
        print(f"Total sectors: {len(sectors['sector'].unique())}")
        print(f"Sectors: {sorted(sectors['sector'].unique())}")

        # Check data quality
        print(f"\n=== Data Quality Check ===")

        # Check if we have enough data for portfolio construction
        min_date = prices["date"].min()
        max_date = prices["date"].max()
        print(f"Price data covers: {min_date} to {max_date}")

        # Check if we have fundamentals for all tickers
        price_tickers = set(prices["ticker"].unique())
        fund_tickers = set(fundamentals["ticker"].unique())
        missing_fundamentals = price_tickers - fund_tickers
        print(f"Tickers with prices but no fundamentals: {len(missing_fundamentals)}")
        if missing_fundamentals:
            print(f"Missing: {list(missing_fundamentals)[:5]}")

        # Check if we have enough historical data
        ticker_counts = prices.groupby("ticker").size()
        print(f"Average data points per ticker: {ticker_counts.mean():.1f}")
        print(f"Min data points per ticker: {ticker_counts.min()}")
        print(f"Max data points per ticker: {ticker_counts.max()}")

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_data()
