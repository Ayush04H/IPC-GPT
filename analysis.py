import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def generate_research_paper_figures(dataset_path):
    """
    Generates Figures 1, 2, and 3 with specified colors and improved professional styling.
    """

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Create 'graphs' directory if it doesn't exist
    graphs_dir = 'graphs'
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    # --- Research Paper Standard Graph Styling ---
    plt.style.use('default')
    paper_colors = {'blue': '#0056b3', 'red': '#cc3300', 'green': '#008000', 'gray': '#666666'}
    model_colors = {'Baseline': paper_colors['red'], 'Fine-Tuned': paper_colors['blue']} # Model-specific colors
    font_size = 10
    title_font_size = 12
    line_width = 1.5
    bar_width = 0.6
    marker_size = 5
    grid_alpha = 0.5
    model_types = ['Baseline', 'Fine-Tuned']

    # --- Figure 1: Hallucination Rate Comparison ---
    if 'Is Hallucination' in df.columns and 'Model Type' in df.columns:
        hallucination_rates_by_model = df.groupby('Model Type')['Is Hallucination'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

        plt.figure(figsize=(6, 4), dpi=300)
        ax = plt.gca() # Get current axes for manual bar plotting
        x_coords = range(len(model_types)) # X-coordinates for bars
        for i, model_type in enumerate(model_types):
            rate = hallucination_rates_by_model.loc[model_type, True]
            ax.bar(x_coords[i], rate, width=bar_width, color=model_colors[model_type], label=model_type) # Plot bars with specific colors

        plt.ylabel('Hallucination Rate (%)', fontsize=font_size)
        plt.xlabel('Model Type', fontsize=font_size)
        plt.title('Figure 1: Hallucination Rate Comparison', fontsize=title_font_size)
        plt.ylim(0, max(hallucination_rates_by_model[True].max() + 5, 30))
        plt.xticks(x_coords, model_types, fontsize=font_size, rotation=0) # Set x-ticks and labels
        plt.yticks(fontsize=font_size)
        plt.grid(axis='y', linestyle='-', alpha=grid_alpha)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(fontsize=font_size - 1) # Add legend

        for i, model_type in enumerate(model_types): # Annotate bars
            height = hallucination_rates_by_model.loc[model_type, True]
            plt.text(x_coords[i], height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=font_size - 1)


        plt.tight_layout()
        graph_filename = os.path.join(graphs_dir, 'figure1_hallucination_rate_comparison_professional.png')
        plt.savefig(graph_filename)
        plt.close()
        print(f"Figure 1 (professional style) saved to: {graph_filename}")

    else:
        print("Warning: 'Is Hallucination' or 'Model Type' column missing. Figure 1 skipped.")
        # Placeholder graph - no changes needed here, already professional

    # --- Figure 2: Fact-Check Accuracy Comparison ---
    if 'Is Correct Answer' in df.columns and 'Model Type' in df.columns:
        accuracy_rates_by_model = df.groupby('Model Type')['Is Correct Answer'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

        plt.figure(figsize=(6, 4), dpi=300)
        ax = plt.gca()
        x_coords = range(len(model_types))
        for i, model_type in enumerate(model_types):
            rate = accuracy_rates_by_model.loc[model_type, True]
            ax.bar(x_coords[i], rate, width=bar_width, color=model_colors[model_type], label=model_type) # Plot bars with specific colors

        plt.ylabel('Fact-Check Accuracy (%)', fontsize=font_size)
        plt.xlabel('Model Type', fontsize=font_size)
        plt.title('Figure 2: Fact-Check Accuracy Comparison', fontsize=title_font_size)
        plt.ylim(0, max(accuracy_rates_by_model[True].max() + 5, 100))
        plt.xticks(x_coords, model_types, fontsize=font_size, rotation=0)
        plt.yticks(fontsize=font_size)
        plt.grid(axis='y', linestyle='-', alpha=grid_alpha)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(fontsize=font_size - 1) # Add legend

        for i, model_type in enumerate(model_types): # Annotate bars
            height = accuracy_rates_by_model.loc[model_type, True]
            plt.text(x_coords[i], height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=font_size - 1)


        plt.tight_layout()
        graph_filename = os.path.join(graphs_dir, 'figure2_fact_check_accuracy_comparison_professional.png')
        plt.savefig(graph_filename)
        plt.close()
        print(f"Figure 2 (professional style) saved to: {graph_filename}")


    else:
        print("Warning: 'Is Correct Answer' or 'Model Type' column missing. Figure 2 skipped.")
        # Placeholder graph - no changes needed here, already professional


    # --- Figure 3: Confidence Score Correlation ---
    if 'Confidence Score' in df.columns and 'Is Correct Answer' in df.columns and 'Model Type' in df.columns:
        plt.figure(figsize=(6, 4), dpi=300)

        for model_type in model_types: # Plot scatter for each model
            model_df = df[df['Model Type'] == model_type]
            confidence_scores_model = model_df['Confidence Score']
            is_correct_numeric_model = model_df['Is Correct Answer'].astype(int)
            correlation, p_value = pearsonr(confidence_scores_model, is_correct_numeric_model)
            print(f"Figure 3 - Pearson Correlation ({model_type} Model): r = {correlation:.2f}, p = {p_value:.3f}")
            plt.scatter(confidence_scores_model, is_correct_numeric_model, alpha=0.6, color=model_colors[model_type], s=marker_size*10, label=model_type) # Color by model


        plt.xlabel('Confidence Score', fontsize=font_size)
        plt.ylabel('Is Correct Answer (0=False, 1=True)', fontsize=font_size)
        plt.title('Figure 3: Confidence Score Correlation', fontsize=title_font_size)
        plt.yticks([0, 1], ['Incorrect', 'Correct'], fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.grid(True, alpha=grid_alpha, linestyle='-')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(fontsize=font_size - 1) # Add legend

        plt.tight_layout()
        graph_filename = os.path.join(graphs_dir, 'figure3_confidence_correlation_scatter_plot_professional.png')
        plt.savefig(graph_filename)
        plt.close()
        print(f"Figure 3 (professional style) saved to: {graph_filename}")


    else:
        print("Warning: 'Confidence Score', 'Is Correct Answer', or 'Model Type' column missing. Figure 3 skipped.")
        # Placeholder graph - no changes needed here, already professional


if __name__ == "__main__":
    dataset_file = 'legal_ai_dataset.csv'  # Or your dataset file
    generate_research_paper_figures(dataset_file)