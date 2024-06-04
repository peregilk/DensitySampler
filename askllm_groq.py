import os
import argparse
import json
import time
import jsonlines
import logging
from tqdm import tqdm
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Reduce logging level for the HTTP requests
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure the Groq API
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load the templates from the template.json file
with open('template.json', 'r') as f:
    templates = json.load(f)

model_name = "llama3-70b-8192"
max_words_per_minute = 4000

def process_json_lines(json_lines_file, output_file, num_examples, language, text_field, verbose):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    chat_prompt = templates[language]

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
        words_in_minute = 0
        start_time = time.time()

        with jsonlines.open(output_file, mode='a') as writer:
            for idx in tqdm(range(processed_lines_count, min(processed_lines_count + num_examples, len(lines))), desc="Processing lines", disable=verbose):
                line = lines[idx]

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
                    request_start_time = time.time()

                    try:
                        chat_completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            temperature=1,
                            max_tokens=1024,
                            top_p=1,
                            stream=False,
                            response_format={"type": "json_object"},
                            stop=None,
                        )

                        response_content = chat_completion.choices[0].message.content
                        logging.debug(f"Response content for line {idx+1}: {response_content}")

                        try:
                            response_json = json.loads(response_content) if isinstance(response_content, str) else response_content
                        except Exception as e:
                            logging.error(f"Failed to parse JSON from response for line {idx+1}: {e}")
                            raise e

                        response_words = len(response_json["reason"].split())

                        # Check if the word limit per minute is reached
                        if words_in_minute + response_words > max_words_per_minute:
                            elapsed_time = time.time() - start_time
                            sleep_time = max(0, 60 - elapsed_time)
                            logging.debug(f"Word limit reached. Sleeping for {sleep_time} seconds.")
                            time.sleep(sleep_time)
                            start_time = time.time()
                            words_in_minute = 0

                        words_in_minute += response_words

                        logging.debug(f"Response JSON for line {idx+1}: {response_json}")

                        line["justification"] = response_json.get("reason", "No justification found")
                        line["educational score"] = response_json.get("educational score", 0)

                        writer.write(line)
                        retries = 0  # Reset retries after a successful operation

                        request_end_time = time.time()
                        request_duration = request_end_time - request_start_time
                        sleep_time = max(0, 2.1 - request_duration)
                        logging.debug(f"Sleeping for {sleep_time} seconds to ensure 2.1-second interval between requests.")
                        time.sleep(sleep_time)
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
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Groq API.")
    parser.add_argument('--json_lines_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONLines file.')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of requests to process (default: 100).')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')
    parser.add_argument('--text_field', type=str, default='text', help='Field in JSON lines containing the text (default: text).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')

    args = parser.parse_args()

    process_json_lines(args.json_lines_file, args.output_file, args.num_examples, args.language, args.text_field, args.verbose)

if __name__ == "__main__":
    main()
