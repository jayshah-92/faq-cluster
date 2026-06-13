# modules/classifier.py

# Import necessary libraries
from transformers import pipeline         # Hugging Face pipeline for NLP tasks
import pandas as pd                       # For working with tabular data
from tqdm import tqdm                     # For displaying progress bars
import logging                            # For logging warnings and errors

# Initialize a zero-shot classification pipeline using a pre-trained BART model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the candidate labels for classification
labels = ["TOFU", "MOFU", "BOFU"]  # TOFU: Top of Funnel, MOFU: Middle, BOFU: Bottom

def batch_zero_shot_predict(texts, batch_size=8):
    """
    Perform zero-shot classification on a list of input texts using a specified batch size.
    We use Hugging Face's pipeline with the labels above and run it in batches for efficiency. This step gives marketing insight into where each question fits in the customer journey.
    
    Parameters:
    -----------
    texts : list of str
        A list of text inputs to be classified.
    batch_size : int, optional (default=8)
        Number of texts to classify in each batch for efficient processing.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with predicted 'funnel_stage' and associated 'confidence' score for each input text.
        If a text is invalid or classification fails, returns a placeholder entry with 'skipped' or 'error'.
    """

    results = []

    # Iterate over the input texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classification"):
        batch_raw = texts[i:i + batch_size]

        # Clean and validate batch inputs (remove empty strings or non-string types)
        batch = [text for text in batch_raw if isinstance(text, str) and text.strip()]
        
        if not batch:
            # Log and skip empty or invalid batch
            logging.warning(f"Skipped empty batch at index {i}")
            results.extend([{"funnel_stage": "skipped", "confidence": 0}] * len(batch_raw))
            continue
        
        try:
            # Run zero-shot classification
            outputs = classifier(batch, labels)

            # Ensure consistent formatting: output could be a dict (for single input) or list (for multiple)
            outputs = [outputs] if isinstance(outputs, dict) else outputs

            # Parse and store the top prediction and its confidence for each text
            for res in outputs:
                results.append({
                    "funnel_stage": res['labels'][0],    # Label with highest score
                    "confidence": res['scores'][0]       # Corresponding confidence score
                })
        
        except Exception as e:
            # Log any errors during classification and return placeholder results
            logging.error(f"Zero-shot batch failed: {e}")
            results.extend([{"funnel_stage": "error", "confidence": 0}] * len(batch_raw))
    
    # Return all results as a structured DataFrame
    return pd.DataFrame(results)
