# Description of Experiment to Verify DensitySampler Method
This description contains direct references to files and folders in my local environment.

I am creating a corpus from: `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/`. The actual data is in `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/data/`, and the memmap of the embeddings are already in `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/embeddings/`.

## Create Density Scores
First create non-normalised density scores in `/mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/nonormalised_density_scores` (takes around an hour):

```
python create_density_scores.py --embedding_input_folder /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/embeddings --json_output_folder /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/nonormalised_density_scores --nonormalise
```

## Sample a Corpus
Then sample a 50% and a 80% corpus based on these. Here we store the result in `/mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus/`:

```
mkdir /mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus
python sample_corpus.py --scores_input_dir /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/nonormalised_density_scores --json_input_dir /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/data --json_output_dir /mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus/boolean80/ --proportion 0.8 --filter
python sample_corpus.py --scores_input_dir /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/nonormalised_density_scores --json_input_dir /mnt/lv_ai_2_ficino/perk/NCC_plus_scandi/data --json_output_dir /mnt/lv_ai_2_ficino/perk/DensityExperimentCorpus/boolean50/ --proportion 0.5 --filter

```
