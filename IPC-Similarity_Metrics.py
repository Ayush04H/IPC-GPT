import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock, minkowski, hamming

# Load the CSV file
file_path = r'Dataset\Final_ipc_sections_metrics.csv'
df = pd.read_csv(file_path)

# Ensure required columns are present
required_columns = ['Punishment', 'Predicted Punishment', 'gpt_neo_answer','Predicted Punishment Gemini']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' is missing in the CSV file.")

# Define a function to convert text to numerical vectors based on character codes
def text_to_vector(text):
    return np.array([ord(c) for c in text])

# Define a function to calculate Manhattan distance
def calculate_manhattan_distance(text1, text2):
    vector1, vector2 = text_to_vector(text1), text_to_vector(text2)
    max_len = max(len(vector1), len(vector2))
    vector1 = np.pad(vector1, (0, max_len - len(vector1)), constant_values=0)
    vector2 = np.pad(vector2, (0, max_len - len(vector2)), constant_values=0)
    return cityblock(vector1, vector2)

# Define a function to calculate Minkowski distance with p=3
def calculate_minkowski_distance(text1, text2, p=3):
    vector1, vector2 = text_to_vector(text1), text_to_vector(text2)
    max_len = max(len(vector1), len(vector2))
    vector1 = np.pad(vector1, (0, max_len - len(vector1)), constant_values=0)
    vector2 = np.pad(vector2, (0, max_len - len(vector2)), constant_values=0)
    return minkowski(vector1, vector2, p)

# Define a function to calculate Hamming distance
def calculate_hamming_distance(text1, text2):
    max_len = max(len(text1), len(text2))
    text1 = text1.ljust(max_len)
    text2 = text2.ljust(max_len)
    return hamming(list(text1), list(text2)) * max_len

# Apply metrics for each row and store in new columns
df['Manhattan Eval - Llama'] = df.apply(lambda row: calculate_manhattan_distance(str(row['Punishment']), str(row['Predicted Punishment'])), axis=1)
df['Manhattan Eval - Gpt'] = df.apply(lambda row: calculate_manhattan_distance(str(row['Punishment']), str(row['gpt_neo_answer'])), axis=1)
df['Manhattan Eval - Gemini'] = df.apply(lambda row: calculate_manhattan_distance(str(row['Punishment']), str(row['Predicted Punishment Gemini'])), axis=1)

df['Minkowski Eval - Llama'] = df.apply(lambda row: calculate_minkowski_distance(str(row['Punishment']), str(row['Predicted Punishment'])), axis=1)
df['Minkowski Eval - Gpt'] = df.apply(lambda row: calculate_minkowski_distance(str(row['Punishment']), str(row['gpt_neo_answer'])), axis=1)
df['Minkowski Eval - Gemini'] = df.apply(lambda row: calculate_minkowski_distance(str(row['Punishment']), str(row['Predicted Punishment Gemini'])), axis=1)

df['Hamming Eval - Llama'] = df.apply(lambda row: calculate_hamming_distance(str(row['Punishment']), str(row['Predicted Punishment'])), axis=1)
df['Hamming Eval - Gpt'] = df.apply(lambda row: calculate_hamming_distance(str(row['Punishment']), str(row['gpt_neo_answer'])), axis=1)
df['Hamming Eval - Gemini'] = df.apply(lambda row: calculate_hamming_distance(str(row['Punishment']), str(row['Predicted Punishment Gemini'])), axis=1)

# Save the updated DataFrame to the existing output CSV
df.to_csv(file_path, index=False)
print(f"Updated CSV with distances saved to {file_path}")

