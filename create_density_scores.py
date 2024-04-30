import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RACE:
    def __init__(self, repetitions: int, hash_range: int, dtype=np.int32):
        self.dtype = dtype
        self.R = repetitions
        self.W = hash_range
        self.counts = np.zeros((self.R, self.W), dtype=self.dtype)

    def add_batch(self, allhashes):
        allhashes = np.array(allhashes, dtype=int) % self.W
        for i in range(self.R):
            self.counts[i, :] += np.bincount(allhashes[i, :], minlength=self.W)

    def query_batch(self, allhashes):
        allhashes = np.array(allhashes, dtype=int) % self.W
        allhashes = allhashes.T
        values = np.zeros(allhashes.shape[0], dtype=float)
        N = np.sum(self.counts) / self.R
        for i, hashvalues in enumerate(allhashes):
            mean = 0
            for idx, hashvalue in enumerate(hashvalues):
                mean += self.counts[idx, hashvalue]
            values[i] = mean / (self.R * N)
        return values

class L2Hash:
    def __init__(self, N: int, d: int, r: float, seed: int = 0):
        self.d = d
        self.N = N
        self.seed = seed
        self.r = r
        self._init_projections()

    def _init_projections(self):
        np.random.seed(self.seed)
        self.W = np.random.normal(size=(self.N, self.d))
        self.b = np.random.uniform(low=0, high=self.r, size=self.N)

    def hash_batch(self, X):
        h = np.dot(self.W, X.T) + self.b[:, np.newaxis]
        h /= self.r
        return np.floor(h)

def print_stats(filepath, num_bins=10, weight_scale_factor=0.01):
    scores = np.memmap(
        filepath,
        dtype="float32",
        mode="r",
    )
    scores = scores * weight_scale_factor
    counts, bins = np.histogram(scores, bins=num_bins, range=(0, 1))
    total = counts.sum()
    print("\nProbability Distribution Bar Chart:")
    for count, bin_edge in zip(counts, bins):
        bin_width = 1 / num_bins
        print(
            f"{bin_edge:.2f} - {bin_edge + bin_width:.2f} | {'#' * int(50 * count / total)} ({count})"
        )
    print("\nStatistics Table:")
    print("Range\t\t\tCount\tPercentage")
    for count, bin_edge in zip(counts, bins):
        bin_width = 1 / num_bins
        print(f"{bin_edge:.2f} - {bin_edge + bin_width:.2f}\t{count}\t{count / total * 100:.2f}%")

def load_or_construct_sketch(args, hash_fn, sketch):
    if args.sketch_file and os.path.exists(args.sketch_file):
        with open(args.sketch_file, "rb") as f:
            return pickle.load(f)
    else:
        logger.info("Constructing sketch from dataset")
        memmap_files = [f for f in os.listdir(args.embedding_input_folder) if f.endswith(".memmap")]
        if not memmap_files:
            logger.warning("No .memmap files found in the input directory.")
            return sketch
        for filename in tqdm(memmap_files):
            file_path = os.path.join(args.embedding_input_folder, filename)
            dataset = np.memmap(file_path, dtype="float32", mode="r")
            num_embeddings = dataset.shape[0] // args.embedding_size
            dataset = dataset.reshape((num_embeddings, args.embedding_size))
            batch_nr = 0
            while True:
                offset_batch = args.batch_size * batch_nr
                dataset_batch = dataset[offset_batch:offset_batch + args.batch_size]
                if dataset_batch.shape[0] == 0:
                    break
                sketch.add_batch(hash_fn.hash_batch(dataset_batch))
                batch_nr += 1
        sketch_file_path = os.path.join(args.embedding_output_folder, "sketch.pkl")
        with open(sketch_file_path, "wb") as f:
            pickle.dump(sketch, f)
        return sketch

def query_sketch_and_save_results(args, sketch, hash_fn):
    logger.info("Querying sketch for each embedding")
    for filename in tqdm(os.listdir(args.embedding_input_folder)):
        if filename.endswith(".memmap"):
            file_path = os.path.join(args.embedding_input_folder, filename)
            dataset = np.memmap(file_path, dtype="float32", mode="r")
            num_embeddings = dataset.shape[0] // args.embedding_size
            results = np.memmap(os.path.join(args.embedding_output_folder, filename.replace('.memmap', '_weights.memmap')),
                                dtype="float32", mode="w+", shape=(num_embeddings,))
            dataset = dataset.reshape((num_embeddings, args.embedding_size))
            batch_nr = 0
            while True:
                offset_batch = args.batch_size * batch_nr
                dataset_batch = dataset[offset_batch:offset_batch + args.batch_size]
                if dataset_batch.shape[0] == 0:
                    break
                scores = sketch.query_batch(hash_fn.hash_batch(dataset_batch))
                weights = 1 / (scores + 1e-8)  # Numerical stability
                if not args.nonormalise:
                    max_weight = np.max(weights)
                    min_weight = np.min(weights)
                    normalized_weights = (weights - min_weight) / (max_weight - min_weight)
                    results[offset_batch:offset_batch + args.batch_size] = normalized_weights
                else:
                    results[offset_batch:offset_batch + args.batch_size] = weights
                batch_nr += 1
            if not args.nostats:
                weight_scale_factor = 1 if not args.nonormalise else 0.01
                print_stats(
                    filepath=os.path.join(
                        args.embedding_output_folder, filename.replace(".memmap", "_weights.memmap")
                    ),
                    num_bins=10,
                    weight_scale_factor=weight_scale_factor,
                )
            if args.json_input_folder and args.json_output_folder:
                json_input_file = os.path.join(args.json_input_folder, filename.replace('.memmap', '.jsonl'))
                json_output_file = os.path.join(args.json_output_folder, filename.replace('.memmap', '.jsonl'))
                with open(json_input_file, 'r') as f_in, open(json_output_file, 'w') as f_out:
                    for i, line in enumerate(f_in):
                        data = json.loads(line)
                        data['density_factor'] = float(results[i])
                        f_out.write(json.dumps(data) + '\n')

def main():
    # Parse command-line arguments
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create the output folder if it doesn't exist
    os.makedirs(args.embedding_output_folder, exist_ok=True)
    if args.json_output_folder:
        os.makedirs(args.json_output_folder, exist_ok=True)

    # Initialize the L2Hash object with the specified parameters
    # N: number of hash functions (sketch_reps)
    # d: embedding size
    # r: kernel bandwidth
    hash_fn = L2Hash(N=args.sketch_reps, d=args.embedding_size, r=args.kernel_bandwidth, seed=0)

    # Initialize the RACE (Repeated Array-based Counting Estimator) object
    # repetitions: number of hash functions (sketch_reps)
    # hash_range: width of the sketch matrix (sketch_range)
    sketch = RACE(repetitions=args.sketch_reps, hash_range=args.sketch_range)

    # Load the sketch from a file if it exists, or construct it from the dataset
    sketch = load_or_construct_sketch(args, hash_fn, sketch)

    # Query the sketch for each embedding and save the results
    query_sketch_and_save_results(args, sketch, hash_fn)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate inverse propensity scores via KDE using locality sensitive hashing.")
    parser.add_argument("--embedding_input_folder", type=str, required=True, help="Folder containing input memmap files")
    parser.add_argument("--embedding_output_folder", type=str, default="scores", help="Folder where memmap scores will be saved")
    parser.add_argument("--json_input_folder", type=str, default=None, help="Folder containing input jsonlines files")
    parser.add_argument("--json_output_folder", type=str, default=None, help="Folder where jsonlines files with density_factor will be saved")
    parser.add_argument("--sketch_reps", type=int, default=1000, help="Number of hash functions (R): rows in sketch matrix")
    parser.add_argument("--sketch_range", type=int, default=20000, help="Width of sketch matrix (hash range B)")
    parser.add_argument("--kernel_bandwidth", type=float, default=0.05, help="Bandwidth of L2 hash kernel")
    parser.add_argument("--embedding_size", type=int, default=384, help="Size of the embeddings. 384 for MiniLM, 768 for BERT")
    parser.add_argument("--batch_size", type=int, default=16384, help="Number of embeddings to load at a time")
    parser.add_argument("--sketch_file", type=str, default=None, help="Path to load the sketch file, if it exists. Otherwise, we construct the sketch.")
    parser.add_argument("--nostats", action="store_true", help="Disable the printing of statistics")
    parser.add_argument("--nonormalise", action="store_true", help="Disable normalization of weights")
    return parser.parse_args()


if __name__ == "__main__":
    main()