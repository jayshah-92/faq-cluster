# modules/data_cleaner.py

import re                      # For regular expression operations
import pandas as pd            # For handling pandas Series

def clean_text_column(series):
    """
    Clean and normalize text data in a pandas Series.
    This helps reduce noise for models and clustering later on.


    This function performs the following cleaning steps:
    1. Replaces NaN values with empty strings.
    2. Converts text to lowercase.
    3. Removes non-ASCII characters (e.g., emojis, accents).
    4. Removes punctuation and special characters (only a-z, 0-9, and whitespace are retained).
    5. Collapses multiple spaces into a single space.
    6. Strips leading and trailing whitespace.

    Parameters:
    -----------
    series : pd.Series
        A pandas Series containing raw text data.

    Returns:
    --------
    pd.Series
        Cleaned and normalized text data.
    
    Example:
    --------
    >>> clean_text_column(pd.Series([" Hello!! ", "Tést🔥Data"]))
    0     hello
    1    test data
    dtype: object
    """
    return (
        series.fillna('')  # Replace NaNs with empty string
              .str.lower()  # Convert all text to lowercase
              .str.encode('ascii', 'ignore').str.decode('utf-8')  # Remove non-ASCII characters
              .str.replace(r'[^a-z0-9\s]', '', regex=True)  # Remove punctuation and symbols
              .str.replace(r'\s+', ' ', regex=True)  # Collapse multiple spaces
              .str.strip()  # Remove leading/trailing whitespace
    )
