import pandas as pd
import requests
import time

# API key and endpoint
API_KEY = ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Function to get response from Gemini API and extract only the text part
def get_gemini_response(description, offense):
    prompt = f"Description: {description}\nOffense: {offense}\nProvide only the punishment details in a short, complete sentence definitions, or summaries related to IPC sections. Answer:"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(
            GEMINI_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": API_KEY}
        )
        response.raise_for_status()  # Ensure we catch HTTP errors
        result = response.json()
        
        # Extracting only the text part from the response JSON
        generated_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        
        return generated_text  # Return only the text part
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return "Error: Unable to fetch response"

# Load the dataset
input_file_path = 'Dataset/Gemini.csv'  # Adjust the path if necessary
output_file_path = 'Dataset/Gemini_Model.csv'

df = pd.read_csv(input_file_path)

# Ensure required columns are present
if 'Description' not in df.columns or 'Offense' not in df.columns:
    raise ValueError("CSV file must contain 'Description' and 'Offense' columns.")

# Add responses with a delay
df['Predicted Punishment Gemini'] = df.apply(
    lambda row: get_gemini_response(row['Description'], row['Offense']), axis=1
)

time.sleep(1)  # Delay between calls to avoid rate limiting

# Save updated CSV
print(df.head())  # Verify new column content
df.to_csv(output_file_path, index=False)
print(f"Updated file saved to: {output_file_path}")
