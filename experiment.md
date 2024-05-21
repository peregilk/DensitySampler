# Description of Experiment to Verify DensitySampler Method
This description contains direct references to files and folders in my local environment.

I am creating a corpus from: `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/`. The actual data is in `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/data/`, and the memmap of the embeddings are already in `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/embeddings/`.

## Create Density Scores
First create non-normalised density scores in `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/nonormalised_density_scores` (This takes around 1 + 8 hours = 9 hours on Ficino):

```
python create_density_scores.py --embedding_input_folder /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/embeddings --nonormalise
```
Note:
We can also store the results in a json by using `--json_output_folder /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/nonormalised_density_scores`. This is however not needed just for sampling a new corpus. 


## Sample a Corpus
Then sample a 50% and a 80% corpus based on these. Here we store the results in `/mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus/`:

```
mkdir /mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus
python sample_corpus.py --scores_input_dir scores --json_input_dir /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/data --json_output_dir /mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus/boolean80/ --proportion 0.8 --filter
python sample_corpus.py --scores_input_dir scores --json_input_dir /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/data --json_output_dir /mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus/boolean50/ --proportion 0.5 --filter

```

