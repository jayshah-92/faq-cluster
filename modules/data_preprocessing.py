# modules/data_preprocessing.py

import re
import pandas as pd
from nltk.corpus import stopwords       # NLTK stopword list
from nltk.tokenize import word_tokenize # Tokenizer for splitting text into words

# Initialize English stopword set once for efficiency
stop_words = set(stopwords.words('english'))

def get_question_type(text):
    """
    Categorize a question into one of several predefined types based on leading keywords.
    This adds structure to the dataset and gives us an idea of the user’s intent behind each question.

    Parameters:
    -----------
    text : str
        A question string.

    Returns:
    --------
    str
        One of the types: 'instructional', 'reasoning', 'informational', 
        'temporal', 'locational', 'personal', 'boolean', 'decision', 
        'comparative', or 'other'.
    """
    if not isinstance(text, str) or not text.strip():
        return 'other'
    
    text = text.lower().strip()

    # Rule-based keyword matching to identify question intent
    if re.match(r'^how\b', text): return 'instructional'
    if re.match(r'^why\b', text): return 'reasoning'
    if re.match(r'^what\b', text): return 'informational'
    if re.match(r'^when\b', text): return 'temporal'
    if re.match(r'^where\b', text): return 'locational'
    if re.match(r'^(who|whom)\b', text): return 'personal'
    if re.match(r'^(is|are|was|were|do|does|did|can|could|will|would|should|am|have|has)\b', text): return 'boolean'
    if re.match(r'^(should|would)\b', text): return 'decision'
    if re.search(r'\b(which|better|best|vs)\b', text): return 'comparative'

    return 'other'


def filtered_stopwords(text):
    """
    Remove English stopwords from input text using NLTK's stopword list.
    This step keeps only meaningful words that might help distinguish one question from another, which is especially useful before vectorizing the text.

    Parameters:
    -----------
    text : str
        Raw text string.

    Returns:
    --------
    str
        Cleaned string with stopwords removed.
    """
    if pd.isnull(text):
        return ''
    
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w.lower() not in stop_words])


def is_question(text):
    """
    Check if a given text is likely a question based on structure or punctuation.
    This filters out statements or malformed data and helps ensure we're only working with actual questions.


    Parameters:
    -----------
    text : str

    Returns:
    --------
    str or None
        Returns the original text if it's a question, else returns None.
    """
    question_starters = (
        'what', 'why', 'how', 'where', 'when', 'who', 'whom', 'which',
        'is', 'are', 'can', 'could', 'do', 'does', 'did', 'will',
        'would', 'shall', 'should', 'may', 'might', 'am'
    )

    if not text:
        return None

    words = text.strip().split()

    # If it ends with "?" or starts with a known interrogative, consider it a question
    return text if (text.endswith('?') or words[0].lower() in question_starters) else None


def drop_duplicates(df, subset_cols=['keyword_cleaned', 'question_text_validated']):
    """
    Drop duplicate rows in a DataFrame based on specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
    subset_cols : list of str
        Columns to consider when identifying duplicates.

    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset_cols)


def add_question_length(df, question_col='question_text_validated'):
    """
    Add a new column to the DataFrame representing question length in word count.

    Parameters:
    -----------
    df : pd.DataFrame
    question_col : str
        Name of the column containing the question text.

    Returns:
    --------
    pd.DataFrame
        Original DataFrame with an additional 'question_length' column.
    """
    df['question_length'] = df[question_col].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )
    return df
