from data_manager import save_all_data, save_all_data_as_json
import os
from update_readme import update_readme

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save all CSV files
    print("Generating CSV files...")
    save_all_data()
    print("CSV files generated successfully.")

    # Generate and save all data as JSON
    print("Generating JSON file...")
    save_all_data_as_json()
    print("JSON file generated successfully.")
    
    # Update README with new format
    print("Updating README.md...")
    update_readme()
    print("Documentation updated successfully.")

if __name__ == "__main__":
    main()
