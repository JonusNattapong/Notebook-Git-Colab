import os
import datetime
import json

def get_project_files():
    """Get list of files in project directory excluding certain files/folders"""
    exclude = {'.git', '__pycache__', 'venv', 'env', 'data'}
    files = []

    for root, dirs, filenames in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude]
        for filename in filenames:
            if not any(x in root for x in exclude):
                files.append(os.path.join(root, filename)[2:])  # Remove './'

    return sorted(files)

def get_file_structure():
    """Generate a formatted file structure tree"""
    files = get_project_files()
    tree = []
    for file in files:
        tree.append(f"- {file}")
    return "\n".join(tree)

def read_data_files():
    """Reads and processes markdown files from the data directory."""
    data_dir = "data"
    data_content = ""

    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".md"):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data_content += f"#### Content from {filename}:\n\n"
                        data_content += f"{f.read()}\n\n"
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            data_content += f"#### Content from {filename} (Read with latin-1 encoding):\n\n"
                            data_content += f"{f.read()}\n\n"
                    except Exception as e:
                        data_content += f"Error reading {filename}: {e}\n\n"
                except Exception as e:
                    data_content += f"Error reading {filename}: {e}\n\n"
    return data_content

def get_license_type():
    """Try to detect license type from LICENSE file"""
    try:
        with open('LICENSE', 'r') as f:
            content = f.read().lower()
            if 'mit' in content:
                return 'MIT'
            elif 'apache' in content:
                return 'Apache'
            elif 'gnu' in content or 'gpl' in content:
                return 'GPL'
            else:
                return 'See LICENSE file'
    except:
        return 'Not specified'

def generate_readme():
    """Generate README.md content"""
    project_name = "Notebook-Git-Colab"
    
    # Get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    content = f"""# {project_name}

## Description
A collaborative notebook system integrating Git version control for efficient note-taking and team collaboration.

## Features
- Git-based version control for notebooks
- Collaborative editing capabilities
- File structure organization
- Change tracking and history

## Project Structure
```
{get_file_structure()}
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/{project_name}.git
cd {project_name}
```

2. Install dependencies (if any):
```bash
# Add any specific installation steps here
```

## Usage
Detailed usage instructions will be added as the project develops.

## Development
The project is under active development. More features and documentation will be added soon.

## License
{get_license_type()}

## Last Updated
{current_date}

## Data
{read_data_files()}

---
Note: This README is automatically generated using update_readme.py
"""

    return content

def update_readme():
    """Update the README.md file"""
    content = generate_readme()
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("README.md has been updated successfully!")

if __name__ == "__main__":
    update_readme()
