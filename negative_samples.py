import json
import argparse

def extract_ids(reference_file):
    """
    Extracts IDs from the reference JSONLines file and stores them in a set.
    
    Args:
        reference_file (str): Path to the reference JSONLines file.
        
    Returns:
        set: A set of IDs present in the reference file.
    """
    ids = set()
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            ids.add(item['id'])
    return ids

def filter_file(target_file, exclude_ids, output_file):
    """
    Filters items in the target JSONLines file that have IDs not present in the exclude_ids set.
    
    Args:
        target_file (str): Path to the target JSONLines file.
        exclude_ids (set): A set of IDs to exclude.
        output_file (str): Path to the output JSONLines file.
        
    Returns:
        tuple: A tuple containing the number of IDs found in the target file, and the number of IDs not found (written to output).
    """
    ids_found = 0
    ids_not_found = 0
    
    with open(target_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line)
            if item['id'] in exclude_ids:
                ids_found += 1
            else:
                outfile.write(line)
                ids_not_found += 1
    
    return ids_found, ids_not_found

def main(reference_file, target_file, output_file):
    """
    Main function to extract IDs from the reference file and filter the target file.
    
    Args:
        reference_file (str): Path to the reference JSONLines file.
        target_file (str): Path to the target JSONLines file.
        output_file (str): Path to the output JSONLines file.
    """
    # Extract IDs from the reference file
    ids_in_reference_file = extract_ids(reference_file)
    
    # Filter the target file based on the extracted IDs
    ids_found, ids_not_found = filter_file(target_file, ids_in_reference_file, output_file)
    
    # Print statistics
    print(f"Number of IDs in reference file: {len(ids_in_reference_file)}")
    print(f"Number of IDs found in target file: {ids_found}")
    print(f"Number of IDs not found (written to output): {ids_not_found}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Filter items in the target file that have IDs not present in the reference file.')
    parser.add_argument('--reference_file', required=True, help='Path to the reference JSONLines file')
    parser.add_argument('--target_file', required=True, help='Path to the target JSONLines file')
    parser.add_argument('--output_file', required=True, help='Path to the output JSONLines file')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    main(args.reference_file, args.target_file, args.output_file)
