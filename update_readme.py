import json
import os

def generate_readme(data_dir="data"):
    """
    Generates a README.md file from JSON data in the specified directory.

    Args:
        data_dir: The directory containing the JSON data files.

    Returns:
        A string containing the Markdown content for the README.md file.
    """

    markdown_content = ["# Dynamically Generated README\n"]

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:  # Added encoding="utf-8"
                    data = json.load(f)

                markdown_content.append(f"## Data from {filename}\n")
                markdown_content.append("| Category | Value |\n")
                markdown_content.append("|---|---|\n")

                # Basic processing - adapt based on actual data structure
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            markdown_content.append(f"| {key}.{sub_key} | {sub_value} |\n")
                    elif isinstance(value, list):
                        # Handle lists (e.g., links)
                        if all(isinstance(item, dict) and 'url' in item and 'text' in item for item in value):
                            #  Handle list of links
                            links_str = ", ".join(f"[{item['text']}]({item['url']})" for item in value)
                            markdown_content.append(f"| {key} | {links_str} |\n")

                        else:
                            markdown_content.append(f"| {key} | {', '.join(str(v) for v in value)} |\n")

                    else:
                        markdown_content.append(f"| {key} | {value} |\n")



            except json.JSONDecodeError:
                markdown_content.append(f"**Error: Could not decode JSON in {filename}**\n")
            except Exception as e:
                markdown_content.append(f"**Error processing {filename}: {e}**\n")


    return "".join(markdown_content)


if __name__ == "__main__":
    readme_content = generate_readme()
    with open("README.md", "w", encoding="utf-8") as f: # Added encoding to output file too
        f.write(readme_content)

    print("README.md generated successfully.")
