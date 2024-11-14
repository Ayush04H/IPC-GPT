import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean
from sklearn.metrics import jaccard_score

# Load the CSV file
file_path = r'C:\Users\nehal\Downloads\ipc_sections_copy_with_answers.csv'
df = pd.read_csv(file_path)

# Ensure required columns are present
required_columns = ['Punishment', 'Predicted Punishment', 'gpt_neo_answer']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' is missing in the CSV file.")

# Define a function to calculate cosine similarity
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Define a function to calculate Jaccard similarity
def calculate_jaccard_similarity(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# Define a function to calculate Euclidean distance
def calculate_euclidean_distance(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return euclidean(vectors[0], vectors[1])

# Apply metrics for each row and store in new columns
df['Cosine Eval - Llama'] = df.apply(lambda row: calculate_cosine_similarity(str(row['Punishment']), str(row['Predicted Punishment'])), axis=1)
df['Cosine Eval - Gpt'] = df.apply(lambda row: calculate_cosine_similarity(str(row['Punishment']), str(row['gpt_neo_answer'])), axis=1)

df['Jaccard Eval - Llama'] = df.apply(lambda row: calculate_jaccard_similarity(str(row['Punishment']), str(row['Predicted Punishment'])), axis=1)
df['Jaccard Eval - Gpt'] = df.apply(lambda row: calculate_jaccard_similarity(str(row['Punishment']), str(row['gpt_neo_answer'])), axis=1)

df['Euclidean Eval - Llama'] = df.apply(lambda row: calculate_euclidean_distance(str(row['Punishment']), str(row['Predicted Punishment'])), axis=1)
df['Euclidean Eval - Gpt'] = df.apply(lambda row: calculate_euclidean_distance(str(row['Punishment']), str(row['gpt_neo_answer'])), axis=1)

# Print the DataFrame with the new metric columns
print(df.head())

# Save the updated DataFrame to a new CSV
# Define the correct output path for your local machine
output_path = r'C:\Users\nehal\Downloads\ipc_sections_metrics.csv'

df.to_csv(output_path, index=False)
print(f"Updated CSV with metrics saved to {output_path}")



