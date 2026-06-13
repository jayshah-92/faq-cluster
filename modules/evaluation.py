# modules/evaluation.py

import matplotlib.pyplot as plt              # For plotting and visualization
import seaborn as sns                       # For attractive statistical plots
import os                                   # For file and directory handling

from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numeric features
from sklearn.decomposition import TruncatedSVD               # Dimensionality reduction
from sklearn.cluster import KMeans                          # Clustering algorithm

def perform_clustering(df, text_column='title_cleaned', n_clusters=3):
    """
    Perform text clustering using TF-IDF, KMeans, and SVD for 2D visualization.
    This can help with topic discovery, FAQ grouping, or content recommendation.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the text data.
    text_column : str
        Column in the DataFrame with cleaned text.
    n_clusters : int
        Number of clusters to form.

    Returns:
    --------
    df : pd.DataFrame
        Original DataFrame with an added 'text_cluster' column indicating cluster assignment.
    components : ndarray
        2D array of SVD-reduced components for visualization.
    """

    # Step 1: TF-IDF Vectorization with stopword removal and bigrams
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english", 
        max_df=0.8,               # Ignore terms in more than 80% of documents
        min_df=3,                 # Ignore terms in fewer than 3 documents
        ngram_range=(1, 2)        # Include unigrams and bigrams
    )
    X = tfidf_vectorizer.fit_transform(df[text_column])

    # Step 2: KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['text_cluster'] = kmeans.fit_predict(X)

    # Step 3: Dimensionality reduction for visualization
    svd = TruncatedSVD(n_components=2, random_state=42)
    components = svd.fit_transform(X)

    return df, components


def save_plot(fig, filename):
    """
    Save a matplotlib figure to disk under a visualisation output folder.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name of the file to save the figure as (e.g., 'plot.png').
    """
    output_dir = "data/output/visualisation"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    print(f"📊 Saved plot: {path}")
    plt.close(fig)  # Close to prevent memory leaks


def plot_cluster_scatter(df, components, save=False):
    """
    Create a 2D scatter plot of clustered text data based on SVD components.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the 'text_cluster' column.
    components : ndarray
        2D SVD-reduced coordinates for each text.
    save : bool
        If True, the plot is saved as an image. Otherwise, it is displayed.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=components[:, 0], 
        y=components[:, 1],
        hue=df['text_cluster'], 
        palette="deep", 
        ax=ax
    )
    ax.set_title("Cluster Visualization (2D SVD)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)
    ax.legend(title="Cluster")
    plt.tight_layout()

    if save:
        save_plot(fig, "cluster_scatter.png")
    else:
        plt.show()


def plot_cluster_distribution(df, save=False):
    """
    Plot a pie chart showing the distribution of data points across clusters.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the 'text_cluster' column.
    save : bool
        If True, the plot is saved as an image. Otherwise, it is displayed.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    df['text_cluster'].value_counts().plot.pie(
        autopct="%1.1f%%",           # Show percentage
        startangle=90,               # Start pie chart at the top
        colors=sns.color_palette("pastel"), 
        ax=ax
    )
    ax.set_title("Cluster Distribution")
    ax.set_ylabel("")  # Hide y-axis label for aesthetics
    plt.tight_layout()

    if save:
        save_plot(fig, "cluster_pie.png")
    else:
        plt.show()
