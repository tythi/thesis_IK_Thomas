# thesis_IK_Thomas

Individual files of Thomas Thiescheffer
**Studentnumber**: S5237009

## Repository structure and file descriptions

thesis_IK-Thomas
- code

  All of the code used to achieve the results from the thesis
- data

  Most of the data generated through the code (some excluded due to their size but still achievable through the code)
- models

  The doc2vec models used to vectorise the text

## Google drive

All other shared files can be found on [Google Drive](https://drive.google.com/drive/folders/1PclYOGt4jK8dUiOy74PvkWd5HfnA3uX6?usp=drive_link). The file called threads1000_format_preprocessed.csv was used for labeling.

## Steps to conduct the research

`pip install -r requirements.txt` to get the needed packages.

1. Download the required files for training `data/formatted_files`.
2. Use `code/prepare_data.py` to get the median and binary conversions (`data/label_medians.csv` and `data/label_binary.csv`), and the required docvecs (`models/doc2vec_[vector_size].model` and `data/docvecs_[vector_size].npy`). Change the vector size in the code to get different vector size files.
3. Use `code/models.py` to get the model results for all of the model and parameter combinations which results in `data/results_[vector_size]_[coversion].json`. Change the vector size in the code to get different result files.
4. Use `code/performances.py` to get the model performances in `data/perfs_[vector_size]_[conversion].csv` and `data/perfs_[vector_size]_[conversion].png`. This will run through each of the vector size and conversion combinations.
5. Use `code/ensemble.py` to get the ensemble model performances in `data/results_ensemble.json`.
6. Use `code/ensemble_performances.py` to print the ensemble performances.
7. Use `code/preprocess_csv.py` to preprocess the out held dataset `data/threads1000_format.csv`.
8. Use `code/label.py` to label the out held dataset `data/threads1000_format_preprocessed.csv`.
9. Use either `code/chi2.py`, `code/correlations.py` or `code/logit.py` to get analysis from the labelled out held dataset for individual label correlations, discussion length correlations or the logistic regression for the time since original post respectively.
10. Use `code/baseline.py` to get a baseline performance on the train/test set based on the most frequently appearing classes per label.

Then in no particular order but after the previous steps:
- Use `code/distributions.py` to get the train/test set label distributions for both conversions.
- Use `code/json_format.py` to remove any indentation from any `data/results_[vector_size]_[conversion].json` to reduce the amount of data present. Do not remember to delete the original file if using this, or alternatively disable indentation writing in `code/models.py`.
- Use `code/jsonl.py` on the original out held dataset `data/threads.json` found in the references of the thesis.
- Use `code/kappas.py` to get Krippendorff's alphas for each label for each combination of annotator.

Then finally, files that are not necissarily used:
- `code/annotation_old.py`, used to get csv files to annotate in Google Sheets
- `code/label_testing.py`, used to find out why some models only classified certain class combinations
