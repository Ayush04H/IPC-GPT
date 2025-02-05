import pandas as pd

# Load the CSV file
file_path = r'Dataset\finetunned\evaluation.csv'
df = pd.read_csv(file_path)

# Define functions to categorize the scores
def categorize_cosine_similarity(score):
    if score >= 0.40:
        return "Good"
    elif 0.50 <= score < 0.15:
        return "Moderate"
    else:
        return "Poor"

def categorize_jaccard_similarity(score):
    if score >= 0.25:
        return "Good"
    elif 0.25 <= score < 0.12:
        return "Moderate"
    else:
        return "Poor"

def categorize_euclidean_distance(score):
    if score < 5:
        return "Good"
    elif 5 <= score <= 15:
        return "Moderate"
    else:
        return "Poor"

def categorize_rouge(score):
    if score >= 0.25:
        return "Good"
    elif 0.15 <= score < 0.25:
        return "Moderate"
    else:
        return "Poor"

def categorize_bleu(score):
    if score >= 0.25:
        return "Good"
    elif 0.15 <= score < 0.25:
        return "Moderate"
    else:
        return "Poor"

# Apply categorization functions to each metric
df['Cosine Category'] = df['Cosine Similarity'].apply(categorize_cosine_similarity)
df['Jaccard Category'] = df['Jaccard Similarity'].apply(categorize_jaccard_similarity)
df['Euclidean Category'] = df['Euclidean Distance'].apply(categorize_euclidean_distance)
df['ROUGE-1 Category'] = df['ROUGE-1'].apply(categorize_rouge)
df['ROUGE-L Category'] = df['ROUGE-L'].apply(categorize_rouge)
df['BLEU Category'] = df['BLEU'].apply(categorize_bleu)

# Save the updated DataFrame back to the same file
df.to_csv(file_path, index=False)
print(f"Updated CSV with categories saved to {file_path}")
