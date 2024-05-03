import jsonlines
import random
import argparse
import sys
import os
from tqdm import tqdm

def filter_jsonlines(input_file, output_file):
    try:
        with jsonlines.open(input_file, 'r') as reader:
            lines = list(reader)
            input_line_count = len(lines)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except jsonlines.jsonlines.InvalidLineError:
        print("Error: Invalid JSONLines format.")
        sys.exit(1)

    output_lines = []
    # Loop until the number of written lines matches the number of read lines
    while len(output_lines) < input_line_count:
        for line in lines:
            if len(output_lines) >= input_line_count:
                break
            density_factor = line.get('density_factor')
            if density_factor is None:
                print(f"Warning: 'density_factor' not found in {input_file}. Skipping this entry.")
                continue
            if random.random() <= density_factor:
                output_lines.append(line)

    try:
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(output_lines)
    except jsonlines.jsonlines.InvalidLineError:
        print(f"Error: Invalid JSONLines format at writing in {output_file}.")
        sys.exit(1)

    print(f"Processed '{input_file}': {input_line_count} lines read, {len(output_lines)} lines written.")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    for filename in tqdm(files, desc="Processing files", unit="file"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        filter_jsonlines(input_file, output_file)

def main():
    parser = argparse.ArgumentParser(description="Filter JSONLines files in a directory based on density score.")
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory containing JSONLines files")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for filtered JSONLines files")
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: The directory {args.input_dir} does not exist.")
        sys.exit(1)

    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

