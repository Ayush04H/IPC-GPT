import os
import google.generativeai as genai
from docx import Document

# Configure API key
genai.configure(api_key="")

# Create the model
generation_config = {
    "temperature": 1.05,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Automate generation for a list of IPC sections with descriptions
def generate_and_save_ipc_descriptions(sections, save_dir=None):
    """
    Generates descriptions for a list of IPC sections, including their names and statements.

    Args:
        sections (list): A list of dictionaries, where each dictionary contains
                       the section number and the section title.  Example:
                       [{'number': 1, 'title': 'Title and extent of operation of the Code.'},
                        {'number': 2, 'title': 'Punishment of offences committed within India.'}]
        save_dir (str, optional): The directory to save the generated documents.
                                   Defaults to the current working directory.
    """

    # Set save directory or use current directory
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it doesn't exist
    else:
        save_dir = os.getcwd()  # Use current directory

    chat_session = model.start_chat(history=[])
    for section in sections:
        section_number = section['number']
        section_title = section['title']

        # Generate prompt
        prompt = f"Describe IPC section {section_number}: '{section_title}' according to Indian Penal Code in 1000 words strictly"

        # Generate response
        print(f"Generating content for IPC Section {section_number}: {section_title}...")
        response = chat_session.send_message(prompt)
        content = response.text

        # Save to Word document
        file_name = os.path.join(save_dir, f"IPC Section {section_number} - {section_title[:50]}.docx") # Truncate title for filename
        document = Document()
        document.add_heading(f"IPC Section {section_number}: {section_title}", level=1)
        document.add_paragraph(content)
        document.save(file_name)
        print(f"Saved content for IPC Section {section_number}: {section_title} to {file_name}")


# Define the list of sections to process
sections_to_generate = [
    {'number': 50, 'title': '“Section”.'},
    {'number': 51, 'title': '“Oath”.'},
    {'number': 52, 'title': '“Good faith”'},
    {'number': 52.1, 'title': '“Harbour-“'}, # Corrected section number
    {'number': 53, 'title': 'Punishments.'},
    {'number': 53.1, 'title': 'Construction of reference to transportation.'},# Corrected section number
    {'number': 54, 'title': 'Commutation of sentence of death.'},
    {'number': 55, 'title': 'Commutation of sentence of imprisonment for life.'},
    {'number': 55.1, 'title': 'Definition of "appropriate Government".'}, # Corrected section number
    {'number': 56, 'title': '[Repealed.]'},
    {'number': 57, 'title': 'Fractions of terms of punishment.'},
    {'number': 58, 'title': '[Repealed.]'},
    {'number': 59, 'title': '[Repealed.]'},
    {'number': 60, 'title': 'Sentence may be (in certain cases of imprisonment) wholly or partly rigorous of simple.'},
    {'number': 61, 'title': '[Repealed.]'},
    {'number': 62, 'title': '[Repealed.]'},
    {'number': 63, 'title': 'Amount of fine.'},
    {'number': 64, 'title': 'Sentence of imprisonment for non-payment of fine.'},
    {'number': 65, 'title': 'Limit to imprisonment for non-payment of fine, when imprisonment and fine awardable.'},
    {'number': 66, 'title': 'Description of imprisonment for non-payment of fine.'},
    {'number': 67, 'title': 'Imprisonment for non-payment of fine, when offence punishable with fine only.'},
    {'number': 68, 'title': 'Imprisonment to terminate on payment of fine.'},
    {'number': 69, 'title': 'Termination of imprisonment on payment of proportional part of fine.'},
    {'number': 70, 'title': 'Fine leviable within six years, of during imprisonment. Death not to discharge property from liability.'},
    {'number': 71, 'title': 'Limit of punishment of offence made up of several offences.'},
    {'number': 72, 'title': 'Punishment of person guilty of one of several offences, the judgment stating that is doubtful of which.'},
    {'number': 73, 'title': 'Solitary confinement.'},
    {'number': 74, 'title': 'Limit of solitary confinement.'},
    {'number': 75, 'title': 'Enhanced punishment for certain offences under Chapter XII or Chapter XVII after previous conviction.'}
]

# You can now print the list to verify
# for section in sections_to_generate:
#     print(f"Section {section['number']}: {section['title']}")
# Call the function with the list of sections and a save directory
generate_and_save_ipc_descriptions(sections_to_generate, save_dir="Sections_2")