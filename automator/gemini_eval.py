import pandas as pd
import requests

# API key for Gemini (directly included as requested)
API_KEY = "Enter APi Keys"

# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Function to get response from Gemini API
def get_gemini_response(description, offense):
    # Refine the prompt using both Description and Offense
    prompt = f"Description: {description}\nOffense: {offense}\nProvide only the punishment details in a short, complete sentence definitions, or summaries related to IPC sections. Answer:"
    
    # Prepare the payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Send the request
    response = requests.post(
        GEMINI_API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        params={"key": API_KEY}
    )
    
    # Check the response status
    if response.status_code == 200:
        result = response.json()
        # Extract the generated text (assuming the response structure contains it)
        generated_text = result.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "").strip()
        return generated_text
    else:
        # Handle error case
        return f"Error: {response.status_code} - {response.text}"

# Path to the input CSV file
input_file_path = 'Dataset/ipc_sections_copy_with_answers.csv'  # Use your dataset path
output_file_path = 'Dataset/Comparative_Analysis_for_Differnt_genai_Model.csv'

# Load the CSV file
df = pd.read_csv(input_file_path)

# Ensure required columns are present
if 'Description' not in df.columns or 'Offense' not in df.columns:
    raise ValueError("CSV file must contain 'Description' and 'Offense' columns.")

# Add a new column for Gemini responses
df['Predicted Punishment Gemini'] = df.apply(lambda row: get_gemini_response(row['Description'], row['Offense']), axis=1)

# Save the updated DataFrame back to the CSV
df.to_csv(output_file_path, index=False)
print(f"Gemini answers added to '{output_file_path}'")
