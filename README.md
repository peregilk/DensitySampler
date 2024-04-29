# DensitySampler
This is an experimental implementation of density sampling for performing semantic deduplication of large corpora. The main idea is to remove semi-duplicates from large dataset to allow for faster and more accurate training. 

The script tries to implement a very efficiant way of calculating this based on the ideas in the following papers:
```
1. Coleman, Benjamin, and Anshumali Shrivastava. "Sub-linear race sketches for approximate kernel density estimation on streaming data." Proceedings of The Web Conference 2020. 2020.
2. Coleman, Benjamin, Richard Baraniuk, and Anshumali Shrivastava. "Sub-linear memory sketches for near neighbor search on streaming data." International Conference on Machine Learning. PMLR, 2020.
3. Coleman, Benjamin, and Anshumali Shrivastava. "A one-pass distributed and private sketch for kernel sums with applications to machine learning at scale." Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security. 2021.
4. Coleman, Benjamin, et al. "One-pass diversified sampling with application to terabyte-scale genomic sequence streams." International Conference on Machine Learning. PMLR, 2022.
5. Liu, Zichang, et al. "One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning." Advances in Neural Information Processing Systems 36 (2024).
```

The script assumes the following directory structure:
```
main/
|-- original_corpus/
|-- paths/
|-- embeddings/
|-- normalised_embeddings/
|-- scratch/
|-- final/
```

The first part creates embeddings. Currently it uses `sentence-transformers/all-MiniLM-L6-v2` that creates 384-dimentional multilingual embeddings. This can be replaced with any other encoder-model from HuggingFace. The default model is using L2 normalising on the embeddings already, so we can save these directly to `normalised_embeddings/`. The script reads the text-field of the jsonlines-file. If your corpus is in parque, please use the `convert_parquet_to_jsonlines.py` first. 

Not that this script takes quite a long time to run even on fast computers. It works on single files, so that it can be easily paralellised. 

```
python create_embeddings.py --input_file myfile.jsonl --paths_dir paths --embeddings_dir normalised_embeddings --emb_size 384
```

Optionally, you would run the following script to apply L2 normalisation. With the default model, this is already done, so you can skip this step.

```
python create_normalised_embeddings.py --input_folder embeddings --output_folder normalised_embeddings --emb_size 384
```

Then run the main script. This will produce a new jsonlines file identical to the original we started with except that it has an extra field called `density_probabiliy`. You will have to at least adjust the `kernel_bandwidth` here.  

```
python create_density_scores.py --input_folder ../NCC_plus_scandi/embeddings --output_folder corpus/density_scores --kernel_bandwidth 0.035 --sketch_reps 1000 --sketch_range 20000
```

