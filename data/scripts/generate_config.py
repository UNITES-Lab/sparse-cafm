import os
import json
import argparse

def generate_json(dataset_root):
    # Define paths to source and target folders
    source_dir = os.path.join(dataset_root, 'source')
    target_dir = os.path.join(dataset_root, 'target')
    
    # List all files in source and target directories
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.png')])

    # Check that both directories have the same number of files
    if len(source_files) != len(target_files):
        raise ValueError("The number of files in 'source' and 'target' directories does not match.")
    
    # Open the output file in write mode
    output_file = os.path.join(dataset_root, 'prompt.json')
    with open(output_file, 'w') as f:
        # Generate each JSON entry and write it as a new line
        for i, (source_file, target_file) in enumerate(zip(source_files, target_files)):
            if source_file != target_file:
                raise ValueError(f"File names do not match: {source_file} and {target_file}")
            
            # Create entry in specified format
            entry = {
                "source": f"source/{source_file}",
                "target": f"target/{target_file}",
                "prompt": ""
            }
            
            # Write the JSON entry as a new line
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"JSON file created at {output_file}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON file for dataset.")
    parser.add_argument("dataset_root", type=str, help="Path to the dataset root directory")
    args = parser.parse_args()
    generate_json(args.dataset_root)