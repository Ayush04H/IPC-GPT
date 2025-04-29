import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Added for violin plot
import numpy as np   # Added for jitter
from scipy.stats import pearsonr
import os

def generate_research_paper_figures(dataset_path):
    """
    Generates Figures 1-7 with specified colors and professional styling
    for a research paper on mitigating AI hallucinations.
    """

    try:
        df = pd.read_csv(dataset_path)
        print(f"Successfully loaded dataset from '{dataset_path}'.")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")
        # Basic validation
        required_cols = ['Is Hallucination', 'Is Correct Answer', 'Confidence Score', 'Model Type']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Dataset missing one or more required columns: {required_cols}")
            return
        if df['Model Type'].nunique() < 2:
            print("Warning: Dataset contains data for less than two model types. Comparisons might be limited.")

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
        print(f"Created directory: '{graphs_dir}'")

    # --- Research Paper Standard Graph Styling ---
    plt.style.use('default') # Start with default and customize
    paper_colors = {'blue': '#0056b3', 'red': '#cc3300', 'green': '#008000', 'gray': '#666666', 'orange': '#ff7f0e', 'purple': '#9467bd'}
    model_colors = {'Baseline': paper_colors['red'], 'Fine-Tuned': paper_colors['blue']}
    font_size = 10
    title_font_size = 12
    line_width = 1.5
    bar_width = 0.35 # Adjusted for grouped bars later
    marker_size = 6
    grid_alpha = 0.6
    model_types = ['Baseline', 'Fine-Tuned'] # Ensure consistent order

    # Check if model types exist in data
    actual_model_types = df['Model Type'].unique()
    if not all(mt in actual_model_types for mt in model_types):
        print(f"Warning: Expected model types {model_types} not fully present in data ({actual_model_types}). Adjusting.")
        # Use only models present in the data for calculations where applicable
        model_types = [mt for mt in model_types if mt in actual_model_types]
        if not model_types:
            print("Error: No expected model types found in the data. Cannot generate plots.")
            return


    # --- Figure 1: Hallucination Rate Comparison (Original) ---
    print("\n--- Generating Figure 1: Hallucination Rate Comparison ---")
    if 'Is Hallucination' in df.columns and 'Model Type' in df.columns:
        try:
            # Calculate rates only for the models present
            hallucination_rates_by_model = df[df['Model Type'].isin(model_types)].groupby('Model Type')['Is Hallucination'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

            # Check if 'True' column exists after unstacking (i.e., if there were any hallucinations)
            if True not in hallucination_rates_by_model.columns:
                hallucination_rates_by_model[True] = 0.0 # Add a column of zeros if no hallucinations found

            plt.figure(figsize=(6, 4), dpi=300)
            ax = plt.gca()
            x_coords = np.arange(len(model_types)) # Use numpy arange for positioning

            bars = ax.bar(x_coords, [hallucination_rates_by_model.loc[mt, True] for mt in model_types], width=bar_width*1.5, color=[model_colors[mt] for mt in model_types])

            plt.ylabel('Hallucination Rate (%)', fontsize=font_size)
            plt.xlabel('Model Type', fontsize=font_size)
            plt.title('Figure 1: Hallucination Rate Comparison', fontsize=title_font_size, fontweight='bold')
            plt.ylim(0, max(hallucination_rates_by_model[True].max() * 1.1, 10)) # Dynamic ylim with minimum
            plt.xticks(x_coords, model_types, fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.grid(axis='y', linestyle='--', alpha=grid_alpha)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add bar labels
            ax.bar_label(bars, fmt='%.1f%%', fontsize=font_size - 1, padding=3)

            plt.tight_layout()
            graph_filename = os.path.join(graphs_dir, 'figure1_hallucination_rate.png')
            plt.savefig(graph_filename)
            plt.close()
            print(f"Figure 1 saved to: {graph_filename}")
        except Exception as e:
            print(f"Error generating Figure 1: {e}")
            plt.close() # Ensure plot is closed even if error occurs
    else:
        print("Warning: 'Is Hallucination' or 'Model Type' column missing. Figure 1 skipped.")

    # --- Figure 2: Fact-Check Accuracy Comparison (Original) ---
    print("\n--- Generating Figure 2: Fact-Check Accuracy Comparison ---")
    if 'Is Correct Answer' in df.columns and 'Model Type' in df.columns:
        try:
            accuracy_rates_by_model = df[df['Model Type'].isin(model_types)].groupby('Model Type')['Is Correct Answer'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

            if True not in accuracy_rates_by_model.columns:
                 accuracy_rates_by_model[True] = 0.0 # Add if no correct answers (unlikely but safe)
            if False not in accuracy_rates_by_model.columns:
                 accuracy_rates_by_model[False] = 0.0 # Add if all answers correct


            plt.figure(figsize=(6, 4), dpi=300)
            ax = plt.gca()
            x_coords = np.arange(len(model_types))

            bars = ax.bar(x_coords, [accuracy_rates_by_model.loc[mt, True] for mt in model_types], width=bar_width*1.5, color=[model_colors[mt] for mt in model_types])

            plt.ylabel('Fact-Check Accuracy (%)', fontsize=font_size)
            plt.xlabel('Model Type', fontsize=font_size)
            plt.title('Figure 2: Fact-Check Accuracy Comparison', fontsize=title_font_size, fontweight='bold')
            plt.ylim(0, 105) # Ylim up to 100% + buffer
            plt.xticks(x_coords, model_types, fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.grid(axis='y', linestyle='--', alpha=grid_alpha)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.bar_label(bars, fmt='%.1f%%', fontsize=font_size - 1, padding=3)

            plt.tight_layout()
            graph_filename = os.path.join(graphs_dir, 'figure2_fact_check_accuracy.png')
            plt.savefig(graph_filename)
            plt.close()
            print(f"Figure 2 saved to: {graph_filename}")
        except Exception as e:
            print(f"Error generating Figure 2: {e}")
            plt.close()
    else:
        print("Warning: 'Is Correct Answer' or 'Model Type' column missing. Figure 2 skipped.")

    # --- Figure 3: Confidence Score Correlation (Original Scatter) ---
    print("\n--- Generating Figure 3: Confidence Score Correlation ---")
    if 'Confidence Score' in df.columns and 'Is Correct Answer' in df.columns and 'Model Type' in df.columns:
        try:
            plt.figure(figsize=(7, 5), dpi=300) # Slightly larger figure
            ax = plt.gca()

            for model_type in model_types:
                model_df = df[df['Model Type'] == model_type]
                if not model_df.empty:
                    confidence_scores_model = model_df['Confidence Score']
                    is_correct_numeric_model = model_df['Is Correct Answer'].astype(int)

                    # Calculate correlation only if there's variance in both variables
                    if confidence_scores_model.nunique() > 1 and is_correct_numeric_model.nunique() > 1:
                       correlation, p_value = pearsonr(confidence_scores_model, is_correct_numeric_model)
                       print(f"  - Pearson Correlation ({model_type}): r = {correlation:.2f}, p = {p_value:.3f}")
                       label_text = f'{model_type} (r={correlation:.2f})'
                    elif confidence_scores_model.nunique() <= 1:
                         print(f"  - Correlation ({model_type}): Not calculated (Confidence scores lack variance)")
                         label_text = f'{model_type} (r=N/A)'
                    else: # is_correct_numeric_model variance is 0 or 1
                         print(f"  - Correlation ({model_type}): Not calculated (Correctness lacks variance)")
                         label_text = f'{model_type} (r=N/A)'

                    # Add jitter for better visibility
                    jitter_correct = is_correct_numeric_model + np.random.normal(0, 0.05, size=len(model_df))
                    ax.scatter(confidence_scores_model, jitter_correct, alpha=0.6,
                                color=model_colors[model_type], s=marker_size*10, label=label_text)


            plt.xlabel('Confidence Score', fontsize=font_size)
            plt.ylabel('Is Correct Answer (0=False, 1=True, with jitter)', fontsize=font_size)
            plt.title('Figure 3: Confidence Score vs. Correctness', fontsize=title_font_size, fontweight='bold')
            plt.yticks([0, 1], ['Incorrect', 'Correct'], fontsize=font_size)
            plt.ylim(-0.2, 1.2) # Accommodate jitter
            plt.xticks(fontsize=font_size)
            plt.grid(True, alpha=grid_alpha, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(fontsize=font_size - 1, loc='center left', bbox_to_anchor=(1, 0.5)) # Legend outside

            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
            graph_filename = os.path.join(graphs_dir, 'figure3_confidence_correlation_scatter.png')
            plt.savefig(graph_filename)
            plt.close()
            print(f"Figure 3 saved to: {graph_filename}")
        except Exception as e:
            print(f"Error generating Figure 3: {e}")
            plt.close()
    else:
        print("Warning: 'Confidence Score', 'Is Correct Answer', or 'Model Type' column missing. Figure 3 skipped.")


    # --- Figure 4: Confidence Distribution (Violin Plot) ---
    print("\n--- Generating Figure 4: Confidence Distribution by Hallucination Status ---")
    if 'Confidence Score' in df.columns and 'Is Hallucination' in df.columns and 'Model Type' in df.columns:
        try:
            plt.figure(figsize=(8, 5), dpi=300)

            # Create a temporary column for clearer legend labels
            df_copy = df.copy() # Work on a copy
            df_copy['Hallucination Status'] = df_copy['Is Hallucination'].map({True: 'Hallucination', False: 'Not Hallucination'})

            sns.violinplot(data=df_copy[df_copy['Model Type'].isin(model_types)], # Filter data used for plot
                           x='Model Type', y='Confidence Score', hue='Hallucination Status',
                           split=True, # Combine violins for direct comparison
                           palette={'Hallucination': paper_colors['red'], 'Not Hallucination': paper_colors['green']},
                           inner='quartile', # Show quartiles
                           linewidth=1.5,
                           order=model_types) # Ensure consistent order

            plt.title('Figure 1: Confidence Score Distribution by Model and Hallucination Status', fontsize=title_font_size, fontweight='bold')
            plt.xlabel('Model Type', fontsize=font_size)
            plt.ylabel('Confidence Score', fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.ylim(0, 1.05) # Confidence score range
            plt.grid(axis='y', linestyle='--', alpha=grid_alpha)
            plt.legend(title='Answer Status', title_fontsize=font_size-1, fontsize=font_size-1, loc='upper right')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            graph_filename = os.path.join(graphs_dir, 'figure4_confidence_distribution_violin.png')
            plt.savefig(graph_filename)
            plt.close()
            print(f"Figure 4 saved to: {graph_filename}")
            del df_copy # Clean up temporary copy
        except Exception as e:
            print(f"Error generating Figure 4: {e}")
            plt.close()
    else:
        print("Warning: 'Confidence Score', 'Is Hallucination', or 'Model Type' column missing. Figure 4 skipped.")


    # --- Figure 5: Hallucination Rate by Question Type (Requires Annotation) ---
    print("\n--- Generating Figure 5: Hallucination Rate by Question Type ---")
    if 'Question Type' in df.columns and 'Is Hallucination' in df.columns and 'Model Type' in df.columns:
        print("  'Question Type' column found. Proceeding...")
        try:
            # Calculate rates only for models present
            hallucination_by_qtype = df[df['Model Type'].isin(model_types)].groupby(['Model Type', 'Question Type'])['Is Hallucination'].mean().mul(100).unstack(level=0)

            if hallucination_by_qtype.empty:
                 print("  Warning: No data found after grouping by Question Type and Model Type. Skipping Figure 5.")
            else:
                # Reindex to ensure both models are present, filling missing with 0
                hallucination_by_qtype = hallucination_by_qtype.reindex(columns=model_types, fill_value=0)

                plt.figure(figsize=(10, 5), dpi=300)
                n_types = len(hallucination_by_qtype.index)
                index = np.arange(n_types) # X locations for the groups

                ax = plt.gca()
                rects1 = ax.bar(index - bar_width/2, hallucination_by_qtype['Baseline'], bar_width, label='Baseline', color=model_colors['Baseline'])
                rects2 = ax.bar(index + bar_width/2, hallucination_by_qtype['Fine-Tuned'], bar_width, label='Fine-Tuned', color=model_colors['Fine-Tuned'])


                plt.title('Figure 5: Hallucination Rate by Question Type and Model', fontsize=title_font_size, fontweight='bold')
                plt.xlabel('Question Type', fontsize=font_size)
                plt.ylabel('Hallucination Rate (%)', fontsize=font_size)
                plt.xticks(index, hallucination_by_qtype.index, fontsize=font_size, rotation=15, ha='right') # Rotate labels slightly if needed
                plt.yticks(fontsize=font_size)
                plt.ylim(0, max(hallucination_by_qtype.max().max() * 1.1, 10)) # Dynamic ylim
                plt.legend(title='Model Type', fontsize=font_size-1)
                plt.grid(axis='y', linestyle='--', alpha=grid_alpha)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Add bar labels
                ax.bar_label(rects1, fmt='%.1f%%', padding=3, fontsize=font_size-2)
                ax.bar_label(rects2, fmt='%.1f%%', padding=3, fontsize=font_size-2)

                plt.tight_layout()
                graph_filename = os.path.join(graphs_dir, 'figure5_hallucination_by_qtype.png')
                plt.savefig(graph_filename)
                plt.close()
                print(f"Figure 5 saved to: {graph_filename}")
        except KeyError as e:
             print(f"  KeyError during Figure 5 generation (likely missing 'Baseline' or 'Fine-Tuned' after grouping/reindexing): {e}. Skipping.")
             plt.close()
        except Exception as e:
            print(f"Error generating Figure 5: {e}")
            plt.close()
    else:
        print("Warning: 'Question Type' column not found in dataset. Figure 5 skipped. (Requires manual annotation).")


    # --- Figure 6: Typology of Hallucinations (Requires Annotation) ---
    print("\n--- Generating Figure 6: Typology of Hallucinations ---")
    if 'Hallucination Type' in df.columns and 'Is Hallucination' in df.columns and 'Model Type' in df.columns:
        print("  'Hallucination Type' column found. Proceeding...")
        try:
            # Filter for hallucinations and ensure Hallucination Type is not null
            hallucinations_df = df[(df['Is Hallucination']) & (df['Hallucination Type'].notna()) & (df['Model Type'].isin(model_types))].copy()

            if hallucinations_df.empty:
                print("  Warning: No valid hallucination entries with 'Hallucination Type' found. Skipping Figure 6.")
            else:
                hallucination_counts = hallucinations_df.groupby(['Model Type', 'Hallucination Type']).size().unstack(fill_value=0)

                # Ensure both models are represented, even if one had 0 hallucinations of annotated types
                hallucination_counts = hallucination_counts.reindex(index=model_types, fill_value=0)

                # Optional: Convert to percentage of total hallucinations per model
                # total_hallucinations_per_model = hallucination_counts.sum(axis=1)
                # hallucination_percentages = hallucination_counts.apply(lambda x: x*100 / total_hallucinations_per_model[x.name] if total_hallucinations_per_model[x.name] > 0 else 0, axis=1)
                # plot_data = hallucination_percentages
                # ylabel_text = 'Percentage of Hallucinations (%)'

                plot_data = hallucination_counts # Using raw counts
                ylabel_text = 'Count of Hallucinations'

                if plot_data.empty or plot_data.sum().sum() == 0:
                     print("  Warning: No hallucination counts to plot after processing. Skipping Figure 6.")
                else:
                    plt.figure(figsize=(8, 5), dpi=300)
                    ax = plot_data.plot(kind='bar', stacked=True,
                                        # Use a predefined colormap or manually define colors if types are fixed
                                        colormap='tab10', # Example: 'viridis', 'plasma', 'tab10', 'Set3'
                                        width=0.6)

                    plt.title('Figure 6: Typology of Hallucinations by Model', fontsize=title_font_size, fontweight='bold')
                    plt.xlabel('Model Type', fontsize=font_size)
                    plt.ylabel(ylabel_text, fontsize=font_size)
                    plt.xticks(rotation=0, fontsize=font_size)
                    plt.yticks(fontsize=font_size)
                    plt.legend(title='Hallucination Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-1)
                    plt.grid(axis='y', linestyle='--', alpha=grid_alpha)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    # Add total count labels on top of stacks (optional)
                    # for i, model_type in enumerate(model_types):
                    #     total = plot_data.loc[model_type].sum()
                    #     if total > 0:
                    #         ax.text(i, total, f'n={int(total)}', ha='center', va='bottom', fontsize=font_size-1)


                    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust layout for external legend
                    graph_filename = os.path.join(graphs_dir, 'figure6_hallucination_typology.png')
                    plt.savefig(graph_filename)
                    plt.close()
                    print(f"Figure 6 saved to: {graph_filename}")
        except Exception as e:
            print(f"Error generating Figure 6: {e}")
            plt.close()
    else:
        print("Warning: 'Hallucination Type' column not found in dataset. Figure 6 skipped. (Requires manual annotation).")

    # --- Figure 7: Confidence vs. Hallucination Scatter Plot (Highlighted) ---
    print("\n--- Generating Figure 7: Confidence vs. Hallucination (Highlighting High-Confidence Errors) ---")
    if 'Confidence Score' in df.columns and 'Is Hallucination' in df.columns and 'Model Type' in df.columns:
        try:
            plt.figure(figsize=(8, 5), dpi=300)
            ax = plt.gca()
            high_conf_threshold = 0.8 # Define high confidence threshold

            for model_type in model_types:
                model_df = df[df['Model Type'] == model_type].copy() # Use .copy()
                if not model_df.empty:
                    # Add jitter to the hallucination status for better visualization
                    jitter = np.random.normal(0, 0.05, size=len(model_df))
                    model_df['Hallucination Jittered'] = model_df['Is Hallucination'].astype(int) + jitter

                    ax.scatter(model_df['Confidence Score'], model_df['Hallucination Jittered'],
                                alpha=0.6,
                                s=30, # Marker size
                                color=model_colors[model_type],
                                label=f'{model_type}')

            # Highlight high-confidence region
            plt.axvline(high_conf_threshold, color=paper_colors['gray'], linestyle='--', linewidth=1.2, label=f'High Conf. ({high_conf_threshold})')
            # Get current xlim *after* plotting points
            current_xlim = plt.xlim()
            # Use axhspan for shading the zone
            ax.axhspan(1 - 0.15, 1 + 0.15, # Shaded area around Y=1 (Hallucination)
                       xmin=(high_conf_threshold - current_xlim[0]) / (current_xlim[1] - current_xlim[0]), # Normalize xmin to axis limits
                       facecolor=paper_colors['red'], alpha=0.10, label='High-Conf. Hallucination Zone', zorder=0) # zorder=0 sends it behind points

            plt.xlabel('Confidence Score', fontsize=font_size)
            plt.ylabel('Hallucination Status (0=No, 1=Yes, with jitter)', fontsize=font_size)
            plt.title('Figure 7: Confidence vs. Hallucination (Highlighting High-Confidence Errors)', fontsize=title_font_size, fontweight='bold')
            plt.yticks([0, 1], ['Not Hallucination', 'Hallucination'], fontsize=font_size)
            plt.ylim(-0.2, 1.2) # Adjust limits to accommodate jitter and highlight zone
            plt.xlim(current_xlim) # Maintain xlim from data
            plt.xticks(fontsize=font_size)
            plt.grid(True, alpha=grid_alpha, linestyle='--')
            plt.legend(fontsize=font_size - 1, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust layout for external legend
            graph_filename = os.path.join(graphs_dir, 'figure7_confidence_vs_hallucination_highlighted.png')
            plt.savefig(graph_filename)
            plt.close()
            print(f"Figure 7 saved to: {graph_filename}")
        except Exception as e:
            print(f"Error generating Figure 7: {e}")
            plt.close()
    else:
        print("Warning: 'Confidence Score', 'Is Hallucination', or 'Model Type' column missing. Figure 7 skipped.")


# --- Main Execution ---
if __name__ == "__main__":
    dataset_file = 'legal_ai_dataset.csv'  # Make sure this file exists
    print(f"Starting graph generation for dataset: {dataset_file}")
    generate_research_paper_figures(dataset_file)
    print("\nGraph generation process completed.")