# # modules/visualizer.py

import matplotlib.pyplot as plt          # Core plotting library
import seaborn as sns                   # Statistical visualization library
import os                               # File and directory handling

def save_plot(fig, filename):
    """
    Save a matplotlib figure to the 'data/output/visualisation' directory.
    1. Histogram of confidence scores from the zero-shot model — helps see how confident the model was in its predictions.
    2. Pie chart of funnel stage predictions — gives an overview of content distribution across stages.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        Name of the output image file (e.g., 'plot.png').

    Output:
    -------
    Saves the figure to disk and closes it to free memory.
    """
    output_dir = "data/output/visualisation"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    print(f"📊 Saved plot: {path}")
    plt.close(fig)  # Prevent it from displaying interactively (useful in scripts)


def plot_confidence_distribution(df, save=False):
    """
    Plot a histogram showing the distribution of confidence scores.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a 'confidence' column with float values.
    save : bool
        If True, the plot will be saved to disk. Otherwise, it is shown interactively.

    Output:
    -------
    A histogram of confidence scores, optionally saved as 'confidence_distribution.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['confidence'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title("Confidence Score Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.grid(True)
    plt.tight_layout()

    if save:
        save_plot(fig, "confidence_distribution.png")
    else:
        plt.show()


def plot_funnel_stage_distribution(df, save=False):
    """
    Plot a pie chart showing the distribution of funnel stages.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a 'funnel_stage' column with categorical stage labels.
    save : bool
        If True, the plot will be saved to disk. Otherwise, it is shown interactively.

    Output:
    -------
    A pie chart of funnel stages, optionally saved as 'funnel_stage_pie.png'.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    df['funnel_stage'].value_counts().plot.pie(
        autopct="%1.1f%%", 
        startangle=90,
        colors=sns.color_palette("pastel"),
        ax=ax
    )
    ax.set_title("Funnel Stage Distribution")
    ax.set_ylabel("")  # Remove Y-axis label for cleaner look
    plt.tight_layout()

    if save:
        save_plot(fig, "funnel_stage_pie.png")
    else:
        plt.show()
