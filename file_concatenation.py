import os
import json
import argparse
from tqdm import tqdm

def concatenate_json_files(folder_path):
    combined_data = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f != 'train.json']

    for filename in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    combined_data.append(data)
                else:
                    print(f"⚠️  Skipping {filename}: not a JSON object")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    output_path = os.path.join(folder_path, 'val.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Concatenated {len(combined_data)} JSON files into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate JSON files into train.json')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing JSON files')
    args = parser.parse_args()

    concatenate_json_files(args.folder_path)
