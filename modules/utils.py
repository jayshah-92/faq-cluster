# modules/utils.py

import os
import random
import logging
import time
import numpy as np
import torch
import nltk
import spacy.cli
import pandas as pd
import sklearn
import transformers

def setup_environment(seed=42, log_path="data/output/error_log.txt"):
    """
    Set up a reproducible and debuggable runtime environment.

    This function:
    - Sets global random seeds for reproducibility.
    - Ensures output/log directory exists.
    - Configures error logging to a specified file.

    Parameters:
    -----------
    seed : int
        Seed value for random number generators (default: 42).
    log_path : str
        File path to write error logs.

    Returns:
    --------
    None
    """
    # Set random seeds for reproducibility across random, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def download_nlp_resources():
    """
    Download required NLTK and spaCy resources for NLP processing.

    Resources downloaded:
    - NLTK tokenizer ('punkt')
    - NLTK stopword list
    - spaCy's small English model ('en_core_web_sm')

    Returns:
    --------
    None
    """
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        spacy.cli.download("en_core_web_sm")
    except Exception as e:
        # Log and print error if resource download fails
        logging.error(f"NLP resource download failed: {e}")
        print(f"NLP resource download failed: {e}")


def print_library_versions():
    """
    Print the versions of key libraries used in the NLP/ML pipeline.

    Useful for debugging, reproducibility, and documenting environment setups.

    Output:
    -------
    Prints version info for:
    - pandas, numpy, sklearn, spaCy, transformers, torch, nltk
    """
    print("\n📦 Library Versions:")
    print(f"Pandas: {pd.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"spaCy: {spacy.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Torch: {torch.__version__}")
    print(f"Nltk: {nltk.__version__}")
