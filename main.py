# main.py
'''
### `Importing Libraries`

Importing all the libraries needed across the project.

- os, re, logging, time, random — for file handling, regex operations, logging errors, and reproducibility.
- pandas & numpy — core data handling libraries.
- nltk & spacy — for tokenization, stopword removal, and entity recognition.
- transformers — using Hugging Face's pipeline for zero-shot classification.
- sklearn — vectorization (TF-IDF), clustering (KMeans), dimensionality reduction (SVD).
- matplotlib & seaborn — for plotting insights.
- tqdm — adds progress bars to loops (especially model predictions).
- torch — ensures reproducibility when using transformer-based models.

'''
from dotenv import load_dotenv
import os
import pandas as pd


from modules import (
    utils, data_loader, data_cleaner, data_preprocessing,
    ner, classifier, visualizer, evaluation, insights
)


# ========== 1. Environment Setup ==========

'''### `Setup: Downloads, Seeding, Logging`

```
This section:

- Downloads required NLP resources from NLTK and spaCy.
- Seeds all random number generators (random, numpy, torch) for reproducibility.
- Sets up logging to capture any errors throughout the process in a text file.
```

'''

load_dotenv()                           # Load .env variables into the environment
utils.setup_environment()               # Set random seed, logging
utils.download_nlp_resources()         # Download NLTK + spaCy resources if not already present
utils.print_library_versions()         # Print versions for reproducibility/debugging


# ========== 2. Load Raw Datasets ==========

# Get folder path from environment variable
folder_path = os.getenv("INPUT_FOLDER")

# Ensure folder exists
if not folder_path or not os.path.isdir(folder_path):
    raise ValueError(f"Folder path '{folder_path}' is not valid.")

# List all files in the folder (you can filter CSVs if needed)
files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')
]

df = data_loader.load_datasets(files)  # Load and concatenate datasets


# ========== 3. Clean & Preprocess Text Columns ==========

df['title_cleaned'] = data_cleaner.clean_text_column(df['title'])       # Clean 'title' column
df['keyword_cleaned'] = data_cleaner.clean_text_column(df['keyword'])   # Clean 'keyword' column

# Token filtering and question detection

df['title_filtered_tokens'] = df['title_cleaned'].apply(data_preprocessing.filtered_stopwords)  # Remove stopwords
df['question_text_validated'] = df['title_cleaned'].apply(data_preprocessing.is_question)       # Keep valid questions
df['question_type'] = df['question_text_validated'].apply(data_preprocessing.get_question_type) # Get question type


# ========== 4. Refine Dataset ==========
df = data_preprocessing.drop_duplicates(df)       # Remove duplicate questions
df = data_preprocessing.add_question_length(df)   # Add a new feature: question length


# ========== 5. Named Entity Recognition ==========
df['extracted_entities'] = df['title'].apply(ner.extract_entities)  # Extract entities using spaCy


# ========== 6. Funnel Stage Classification ==========
texts = df['title_cleaned'].fillna("").tolist()
result_df = classifier.batch_zero_shot_predict(texts)               # Predict funnel stage + confidence score
df[['funnel_stage', 'confidence']] = result_df                      # Assign prediction results


# ========== 7. Save Cleaned Dataset ==========
output_file = os.getenv("OUTPUT_FILE")
df.to_csv(output_file, index=False)

# df.to_csv('data/output/questions_final.csv', index=False)           # Persist output


# ========== 8. Visualize Funnel Insights ==========
visualizer.plot_confidence_distribution(df, save=True)              # Histogram of confidence
visualizer.plot_funnel_stage_distribution(df, save=True)            # Pie chart of funnel stages


# ========== 9. Text Clustering & Visualization ==========
df, components = evaluation.perform_clustering(df, text_column='title_cleaned', n_clusters=3)  # Dimensionality reduction + KMeans
evaluation.plot_cluster_scatter(df, components, save=True)           # 2D cluster scatter
evaluation.plot_cluster_distribution(df, save=True)                 # Bar chart: Cluster count


# ========== 10. Insight Reporting ==========
insights.print_question_type_by_cluster(df)                         # Print question-type breakdown by cluster
insights.print_sample_questions_per_cluster(df, n=3)                # Show sample FAQs per cluster
insights.print_cluster_stage_suggestions()                          # Interpret clusters in marketing funnel context
