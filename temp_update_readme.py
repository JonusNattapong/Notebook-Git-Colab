import json
import os
# Placeholder for Mistral API import and setup
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

def generate_markdown_table(data, headers):
    """Generates a Markdown table from a list of dictionaries."""
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for item in data:
        row = "| " + " | ".join(str(item.get(header, "")) for header in headers) + " |"
        rows.append(row)
    return "\n".join([header_row, separator_row, *rows]) + "\n"

def fetch_data(data_dir="data"):
    """Fetches data from JSON files in the specified directory."""
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:  # Specify UTF-8 encoding
                    content = json.load(f)
                    # Assuming each JSON file represents a single data entry for now
                    data.append(content)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error reading or parsing {filename}: {e}")
    return data

def process_data(data):
    """Processes the fetched data, extracting relevant information and using Mistral API."""
    processed_data = []
    for item in data:
        # Example: Extract URL and title from the JSON data
        url = item.get('url', '')
        # Extract a relevant piece of text for summarization.  This is highly
        # dependent on the structure of your JSON data.  Adapt as needed.
        text_to_summarize = item.get('content', {}).get('json_ld', [{}])[0].get('headline', '')

        # Placeholder for Mistral API call
        # summary = get_summary_from_mistral(text_to_summarize)
        summary = "Placeholder summary. Mistral API integration needed." # Placeholder

        processed_data.append({
            "URL": url,
            "Title": text_to_summarize,
            "Summary": summary
        })
    return processed_data

# Placeholder function for Mistral API interaction
# def get_summary_from_mistral(text):
#     """Gets a summary from the Mistral API."""
#     api_key = os.environ["MISTRAL_API_KEY"]  # Get API key from environment variable
#     client = MistralClient(api_key=api_key)
#
#     messages = [
#         ChatMessage(role="user", content=f"Summarize the following text: {text}")
#     ]
#
#     chat_response = client.chat(
#         model="mistral-small",  # Or any other suitable model
#         messages=messages,
#     )
#
#     return chat_response.choices[0].message.content

def generate_readme(processed_data):
    """Generates the README.md content."""
    readme_content = "# Dynamic README\n\n"
    headers = ["URL", "Title", "Summary"]
    readme_content += generate_markdown_table(processed_data, headers)
    return readme_content

def main():
    """Main function to orchestrate the process."""
    data = fetch_data()
    processed_data = process_data(data)
    readme_content = generate_readme(processed_data)

    with open("README.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()