# modules/insights.py

import pandas as pd

def print_question_type_by_cluster(df):
    """
    Display the distribution of question types across different text clusters.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'text_cluster' and 'question_type' columns.

    Output:
    -------
    Prints a grouped frequency count of question types within each cluster.
    """
    print("\n--- Cluster-wise Question Type Distribution ---")
    print(df.groupby('text_cluster')['question_type'].value_counts())


def print_sample_questions_per_cluster(df, n=3):
    """
    Print a sample of validated questions from each text cluster.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'text_cluster' and 'question_text_validated'.
    n : int
        Number of questions to print per cluster.

    Output:
    -------
    Prints `n` random sample questions from each cluster.
    """
    print("\n--- Sample FAQs per Cluster ---")
    for c in sorted(df['text_cluster'].unique()):
        print(f"\n🔹 Cluster {c}:")
        samples = (
            df[df['text_cluster'] == c]['question_text_validated']
              .dropna()
              .sample(n, random_state=42)
        )
        for q in samples:
            print(f"   • {q}")


def print_cluster_stage_suggestions():
    """
    Print marketing funnel stage suggestions based on common question types.

    Output:
    -------
    Prints predefined insights linking question patterns to TOFU, MOFU, and BOFU use cases.
    """
    print("\n--- Suggested FAQ Use Cases ---")

    # Predefined insights for marketing funnel alignment
    cluster_insights = {
        'TOFU': "Primarily 'how' and 'what' → good for TOFU onboarding or guides.",
        'MOFU': "'Can', 'is', 'do' → decision or action questions (mid-funnel).",
        'BOFU': "'Why', 'should' → intent-heavy, support or decision-making."
    }

    for k, v in cluster_insights.items():
        print(f"{k}: {v}")
