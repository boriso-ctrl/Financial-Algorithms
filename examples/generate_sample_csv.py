"""
Generate Sample CSV Data for Testing

This script creates a sample CSV file with realistic market data
that can be used to test the CSV backtest functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from examples.generate_sample_data import generate_intraday_data, add_realistic_patterns


def main():
    """Generate and save sample CSV data."""
    print("Generating sample CSV data...")
    
    # Generate 60 days of 5-minute data
    df = generate_intraday_data(
        start_date='2023-11-01',
        num_days=60,
        initial_price=450.0,
        volatility=0.015
    )
    
    df = add_realistic_patterns(df)
    
    # Reset index to have timestamp as column
    df_csv = df.reset_index()
    
    # Save to CSV
    output_path = 'examples/sample_data/sample_5min_data.csv'
    df_csv.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df_csv)} bars")
    print(f"  Date range: {df_csv['timestamp'].min()} to {df_csv['timestamp'].max()}")
    print(f"  Sessions: {df_csv['session'].nunique()}")
    print(f"  Saved to: {output_path}")
    print()
    print("You can now test with:")
    print(f"  python examples/run_csv_backtest.py --csv {output_path}")


if __name__ == "__main__":
    main()
