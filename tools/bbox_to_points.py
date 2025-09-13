#!/usr/bin/env python3

__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: September 13, 2025
    """

import pandas as pd
import sys
import argparse
from pathlib import Path

def bbox_to_points(input_csv: str, output_csv: str) -> None:
    """Convert bounding box CSV to points CSV by calculating centers.
    
    Args:
        input_csv (str): Path to input CSV file with bbox annotations
        output_csv (str): Path to output CSV file for point annotations
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_csv}' is empty.")
        sys.exit(1)
    
    # Verify required columns exist
    required_cols = ['images', 'x_min', 'y_min', 'x_max', 'y_max', 'labels']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Converting {len(df)} bounding box annotations to points...")
    
    # Calculate center points
    df['x'] = (df['x_min'] + df['x_max']) / 2
    df['y'] = (df['y_min'] + df['y_max']) / 2
    
    # Keep only required columns for points format
    points_df = df[['images', 'x', 'y', 'labels']].copy()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    points_df.to_csv(output_csv, index=False)
    print(f"Successfully converted {len(points_df)} annotations to points format")
    print(f"Output saved to: {output_csv}")
    
    # Show sample of converted data
    print("\nSample of converted data:")
    print(points_df.head())

def main():
    parser = argparse.ArgumentParser(
        description="Convert bounding box annotations to point annotations by calculating bbox centers"
    )
    parser.add_argument(
        "input_csv", 
        help="Path to input CSV file with bounding box annotations"
    )
    parser.add_argument(
        "output_csv", 
        help="Path to output CSV file for point annotations"
    )
    
    args = parser.parse_args()
    
    bbox_to_points(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()