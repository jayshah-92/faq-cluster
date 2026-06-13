# modules/ner.py

import spacy                      # For natural language processing
import logging                    # For error logging

# Load small English language model (includes tokenization, tagging, NER, etc.)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from a given text using spaCy's pre-trained model.
    Entities include things like people, organizations, dates, and more. We also add descriptions for each label using spaCy's built-in definitions.
    This can help with tagging, content personalization, or filtering later.


    Parameters:
    -----------
    text : str
        Input text from which to extract named entities.

    Returns:
    --------
    List[dict]
        A list of dictionaries, each representing an entity with:
            - text: Entity text as it appears in input
            - start: Start character index
            - end: End character index
            - label: Entity type (e.g., ORG, PERSON, DATE)
            - label_desc: Human-readable explanation of the entity label

    In case of failure, logs the error and returns an empty list.

    Example:
    --------
    >>> extract_entities("Apple was founded in 1976 by Steve Jobs.")
    [{'text': 'Apple', 'start': 0, 'end': 5, 'label': 'ORG', 'label_desc': 'Companies, agencies, institutions, etc.'},
     {'text': '1976', 'start': 21, 'end': 25, 'label': 'DATE', 'label_desc': 'Absolute or relative dates or periods'},
     {'text': 'Steve Jobs', 'start': 29, 'end': 40, 'label': 'PERSON', 'label_desc': 'People, including fictional'}]
    """
    try:
        # Preprocess text and apply NLP pipeline
        doc = nlp(str(text).strip())

        # Extract and return structured entity data
        return [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "label_desc": spacy.explain(ent.label_) or "Unknown"
            }
            for ent in doc.ents
        ]

    except Exception as e:
        # Log any unexpected error and return empty result
        logging.error(f"NER failed for text: {text} | Error: {e}")
        return []
