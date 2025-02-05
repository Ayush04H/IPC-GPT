import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the CSV file
file_path = r'Dataset\finetunned\analysis_dataset_copy.csv'
df = pd.read_csv(file_path)

# Ensure required columns are present
required_columns = ['answer', 'fine_tunned_model_answers']
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

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Apply metrics for each row and store in new columns
df['Cosine Similarity'] = df.apply(lambda row: calculate_cosine_similarity(str(row['answer']), str(row['fine_tunned_model_answers'])), axis=1)
df['Jaccard Similarity'] = df.apply(lambda row: calculate_jaccard_similarity(str(row['answer']), str(row['fine_tunned_model_answers'])), axis=1)
df['Euclidean Distance'] = df.apply(lambda row: calculate_euclidean_distance(str(row['answer']), str(row['fine_tunned_model_answers'])), axis=1)

# ROUGE-1, ROUGE-L
def calculate_rouge_scores(text1, text2):
    scores = scorer.score(text1, text2)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

df[['ROUGE-1', 'ROUGE-L']] = df.apply(lambda row: pd.Series(calculate_rouge_scores(str(row['answer']), str(row['fine_tunned_model_answers']))), axis=1)

# BLEU Score
def calculate_bleu_score(text1, text2):
    reference = [text1.split()]
    candidate = text2.split()
    smoothing = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

df['BLEU'] = df.apply(lambda row: calculate_bleu_score(str(row['answer']), str(row['fine_tunned_model_answers'])), axis=1)

# Print the DataFrame with the new metric columns
print(df.head())

# Save the updated DataFrame to a new CSV
output_path = r'Dataset\finetunned\evaluation.csv'

df.to_csv(output_path, index=False)
print(f"Updated CSV with metrics saved to {output_path}")
