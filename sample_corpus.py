import numpy as np
import argparse
import os
import multiprocessing as mp
import jsonlines
from tqdm import tqdm


def sample_without_replacement(scores, proportion):
    """
    Sample indices from 0 to len(scores) without replacement.
    Return a boolean mask with True values for sampled indices,
    and False values for the rest.

    Args:
    scores: np.array
        Array of scores to use as weights for sampling.
    proportion: float
        Proportion of observations to sample from corpus.
    """
    sampling_probabilities = scores / np.sum(scores)
    sampled_indices = np.random.choice(
        a=np.arange(len(scores)),
        size=int(proportion * len(scores)),
        replace=False,
        p=sampling_probabilities,
    )
    boolean_mask = np.zeros(len(scores), dtype=bool)  # Initialize with False
    boolean_mask[sampled_indices] = True
    return boolean_mask


def add_is_sampled_field(meta):
    """
    Add a boolean field "is_sampled_density" to the JSONLines file(s)
    in the output directory.
    """
    with jsonlines.open(meta["json_path"], "r") as reader, jsonlines.open(
        os.path.join(args.json_output_dir, meta["json_file"]), "w"
    ) as writer:
        # Add nested tqdm for the reader to show progress

        for idx, record in enumerate(tqdm(reader, desc=meta["json_file"], leave=False)):
            record["is_sampled_density"] = bool(meta["boolean_chunk"][idx])

            if args.filter and not record["is_sampled_density"]:
                continue

            writer.write(record)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter JSONLines files in a directory based on density score."
    )
    parser.add_argument(
        "--scores_input_dir",
        type=str,
        help="Directory containsing memmap density scores files",
        default="scores",
    )
    parser.add_argument(
        "--json_input_dir",
        required=True,
        type=str,
        help="Input directory containing JSONLines files",
    )
    parser.add_argument(
        "--json_output_dir",
        required=True,
        type=str,
        help="Output directory for JSONLines files with density_sampled field added",
    )
    parser.add_argument(
        "--proportion",
        default=0.5,
        type=float,
        help="Proportion of records to sample",
    )
    parser.add_argument(
        "--filter",
        default=False,
        action="store_true",
        help="Save only sampled records (is_sampled_density==True) to json output.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    score_filenames = [
        file for file in os.listdir(args.scores_input_dir) if file.endswith(".memmap")
    ]
    json_filenames = [
        file.replace("_weights", "").replace("memmap", "jsonl") for file in score_filenames
    ]

    scores = []
    metadata = []
    start_idx = 0
    for score_file, json_file in zip(score_filenames, json_filenames):
        score = np.memmap(
            os.path.join(args.scores_input_dir, score_file), dtype="float32", mode="r"
        )
        scores.append(score)
        nr_records = score.shape[0]
        # Keep track of the start and end indices of each record in the jsonlines files
        # within the concatenated corpus
        metadata.append(
            {
                "json_file": json_file,
                "score_file": score_file,
                "start_idx": start_idx,
                "end_idx": start_idx + nr_records,
            }
        )
        start_idx += nr_records

    scores = np.concatenate(scores, axis=0)
    boolean_index = sample_without_replacement(scores, args.proportion)
    print(f"Sampled {np.sum(boolean_index)} records out of {len(scores)}")

    # Divide the boolean index into chunks corresponding to the original files
    for meta in metadata:
        meta["boolean_chunk"] = boolean_index[meta["start_idx"] : meta["end_idx"]]
        meta["json_path"] = os.path.join(args.json_input_dir, meta["json_file"])

    os.makedirs(args.json_output_dir, exist_ok=True)

    # Write new JSONLines files with the "is_sampled_density" field included
    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(add_is_sampled_field, metadata), total=len(metadata)))

