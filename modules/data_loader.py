# modules/data_loader.py

import os                      # For file path operations (optional, useful if extending)
import pandas as pd            # For reading and managing tabular data

def load_datasets(file_paths):
    """
    Load multiple CSV files and return a single concatenated DataFrame.

    This function:
    - Reads each file path from the input list.
    - Loads each CSV into a pandas DataFrame.
    - Logs key information such as head, info, statistical summary, and missing values.
    - Handles errors gracefully and continues loading remaining files.
    - Concatenates all DataFrames into one.

    Parameters:
    -----------
    file_paths : list of str
        List of paths to CSV files.

    Returns:
    --------
    pd.DataFrame
        A single pandas DataFrame containing all the rows from the input files.
        Files that fail to load are skipped with an error message.
    
    Example:
    --------
    >>> load_datasets(['data/file1.csv', 'data/file2.csv'])
    """

    dataframes = []

    # Iterate over each file path in the list
    for file in file_paths:
        try:
            df = pd.read_csv(file)  # Attempt to read CSV
            dataframes.append(df)

            # Log dataset overview for debugging or EDA purposes
            print(f"\n--- {file} ---")
            print(df.head())                   # First few rows
            print(df.info())                   # Column info and data types
            print(df.describe())               # Summary statistics
            print("Missing Values:\n", df.isnull().sum())  # Count of NaNs per column

        except Exception as e:
            # Log the error and continue with next file
            print(f"Error reading {file}: {e}")

    # Concatenate all successfully loaded DataFrames
    return pd.concat(dataframes, ignore_index=True)
