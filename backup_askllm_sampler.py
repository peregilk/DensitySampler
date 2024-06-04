import os
import argparse
import json
import google.generativeai as genai
import time
import jsonlines
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Google AI API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load the templates from the template.json file
with open('template.json', 'r') as f:
    templates = json.load(f)

# Define the model configuration
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

def process_json_lines(json_lines_file, output_file, num_examples, max_requests_per_minute, language, text_field, verbose):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    chat_prompt = templates[language]

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    chat_prompt
                ],
            },
        ]
    )

    try:
        # Count the number of lines already processed in the output file
        if os.path.exists(output_file):
            with jsonlines.open(output_file, mode='r') as reader:
                processed_lines_count = sum(1 for _ in reader)
        else:
            processed_lines_count = 0

        with jsonlines.open(json_lines_file, mode='r') as reader:
            lines = list(reader)

        retries = 0

        with jsonlines.open(output_file, mode='a') as writer:
            for idx, line in enumerate(tqdm(lines, desc="Processing lines", disable=verbose)):
                if idx < processed_lines_count:
                    continue  # Skip already processed lines
                
                if idx >= num_examples:
                    break

                if "educational score" in line:
                    writer.write(line)
                    continue
                if text_field not in line:
                    logging.error(f"Field '{text_field}' not found in line {idx+1}. Make sure the input JSONL file contains this field.")
                    print(f"Error: Field '{text_field}' not found in line {idx+1}. Make sure the input JSONL file contains this field.")
                    writer.write(line)
                    continue

                input_text = line[text_field]
                prompt = chat_prompt.format(content=input_text)
                
                logging.debug(f"Processing line {idx+1}/{num_examples}: {prompt}")
                
                while retries < 5:
                    try:
                        response = chat_session.send_message(prompt)
                        response_json = json.loads(response.text)

                        logging.debug(f"Response for line {idx+1}: {response_json}")

                        line["justification"] = response_json.get("reason", "No justification found")
                        line["educational score"] = response_json.get("educational score", 0)
                        
                        writer.write(line)
                        retries = 0  # Reset retries after a successful operation

                        if (idx + 1) % max_requests_per_minute == 0:
                            logging.debug(f"Processed {idx + 1} entries. Waiting for a minute to respect rate limit.")
                            print("Sleeping for 60 seconds")
                            time.sleep(60)  # Wait for a minute to respect rate limit
                        break
                    except Exception as e:
                        retries += 1
                        logging.error(f"An error occurred while processing line {idx+1} (attempt {retries}): {e}")
                        if retries >= 5:
                            logging.error("Maximum retry limit reached. Exiting the script.")
                            exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Google Generative AI model.")
    parser.add_argument('--json_lines_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONLines file.')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of requests to process (default: 100).')
    parser.add_argument('--max_requests_per_minute', type=int, default=1000, help='Maximum number of requests per minute (default: 1000).')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')
    parser.add_argument('--text_field', type=str, default='text', help='Field in JSON lines containing the text (default: text).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')

    args = parser.parse_args()

    process_json_lines(args.json_lines_file, args.output_file, args.num_examples, args.max_requests_per_minute, args.language, args.text_field, args.verbose)

if __name__ == "__main__":
    main()
