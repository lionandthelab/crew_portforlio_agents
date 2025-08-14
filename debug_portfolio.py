#!/usr/bin/env python3
"""
Debug script to test portfolio construction logic
"""

import pandas as pd
import numpy as np
from src.tools.pipeline_tool_impl import (
    load_prices,
    load_fundamentals,
    compute_signals,
    build_portfolio,
    execute_trades,
    run_backtest,
)
from src.tools.market_data import load_real_market_data


def debug_portfolio():
    print("=== Debugging Portfolio Construction ===")

    start_date = "2023-01-01"
    end_date = "2025-07-31"

    try:
        # Load data
        print("1. Loading data...")
        prices = load_prices(start_date, end_date, use_real_data=True)
        fundamentals = load_fundamentals(start_date, end_date, use_real_data=True)

        print(f"   Prices shape: {prices.shape}")
        print(f"   Fundamentals shape: {fundamentals.shape}")

        # Load config
        print("\n2. Loading config...")
        import yaml

        with open("config/constraints.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        print(f"   Config: {cfg}")

        # Get sectors
        print("\n3. Processing sectors...")
        sectors = fundamentals[["ticker", "sector"]].drop_duplicates().copy()
        print(f"   Sectors shape: {sectors.shape}")
        print(f"   Sample sectors: {sectors.head()}")

        # Compute signals
        print("\n4. Computing signals...")
        scores = compute_signals(prices, fundamentals, cfg)
        print(f"   Scores shape: {scores.shape}")
        print(f"   Scores date range: {scores['date'].min()} to {scores['date'].max()}")
        print(f"   Sample scores: {scores.head()}")

        # Check if scores have valid values
        print(f"   Score statistics:")
        print(f"   - Min: {scores['score'].min()}")
        print(f"   - Max: {scores['score'].max()}")
        print(f"   - Mean: {scores['score'].mean()}")
        print(f"   - Std: {scores['score'].std()}")

        # Build portfolio
        print("\n5. Building portfolio...")
        target_weights = build_portfolio(scores, sectors, cfg)
        print(f"   Target weights shape: {target_weights.shape}")
        print(f"   Target weights sample: {target_weights.head()}")

        if len(target_weights) > 0:
            print(f"   Weight statistics:")
            print(f"   - Min: {target_weights['weight'].min()}")
            print(f"   - Max: {target_weights['weight'].max()}")
            print(f"   - Sum: {target_weights['weight'].sum()}")

            # Execute trades
            print("\n6. Executing trades...")
            weights, trades = execute_trades(target_weights, cfg)
            print(f"   Weights shape: {weights.shape}")
            print(f"   Trades shape: {trades.shape}")

            if len(weights) > 0:
                print(f"   Final weight statistics:")
                print(f"   - Min: {weights['weight'].min()}")
                print(f"   - Max: {weights['weight'].max()}")
                print(f"   - Sum: {weights['weight'].sum()}")

                # Run backtest
                print("\n7. Running backtest...")
                equity, metrics = run_backtest(weights, trades, prices, cfg)
                print(f"   Equity shape: {equity.shape}")
                print(f"   Metrics: {metrics}")
            else:
                print("   ERROR: No weights generated!")
        else:
            print("   ERROR: No target weights generated!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_portfolio()
