# GlossReader at SemEval-2021 Task 2: Reading Definitions Improves Contextualized Word Embeddings

#### This is code for the system comparable with the 2nd best system for French and the 6th best system for Arabic in [MCL-WiC](https://competitions.codalab.org/competitions/27054) competition.

## How to run

### Step 0: Prepare environment
1. Install [python](https://python.org/) 3.7 or later.
1. Clone repo and move to the root directory (i.e., `MCL-WiC`).
1. Run `pip install -r requirements.txt` to install the required packages.
1. To run the code with the default precomputed parameters, download `data` directory from [drive](https://yadi.sk/d/Lb_UOgEmZSCAeg).
1. Place `data` directory in the root of the project (i.e., `MCL-WiC/data`).

### Step 1: Generate predictions file
Run `python multilingual.py` to generate our best multilingual predictions (*GLM XLM-R large - Manhattan+norm (en dev)* from the results table).
Additionally, you can change the default parameters:
- `--similarity` - similarity measure for the contextualized embeddings.
- `--threshold` - threshold selection strategy. 
***
- `--en-dev-data` - MCL-WiC en-en dev data.
- `--en-dev-gold` - MCL-WiC en-en dev gold labels.
- `--ru-dev-data` - MCL-WiC ru-ru dev data.
- `--ru-dev-gold` - MCL-WiC ru-ru dev gold labels.
- `--fr-dev-data` - MCL-WiC fr-fr dev data.
- `--fr-dev-gold` - MCL-WiC fr-fr dev gold labels.
- `--ar-dev-data` - MCL-WiC ar-ar dev data.
- `--ar-dev-gold` - MCL-WiC ar-ar dev gold labels.
- `--zh-dev-data` - MCL-WiC zh-zh dev data.
- `--zh-dev-gold` - MCL-WiC zh-zh dev gold labels.
***
- `--en-test-data` - MCL-WiC en-en test data.
- `--ru-test-data` - MCL-WiC ru-ru test data.
- `--fr-test-data` - MCL-WiC fr-fr test data.
- `--ar-test-data` - MCL-WiC ar-ar test data.
- `--zh-test-data` - MCL-WiC zh-zh test data.
***
- `--en-dev-vectors` - precomputed vectors for the en-en dev data.
- `--ru-dev-vectors` - precomputed vectors for the ru-ru dev data.
- `--fr-dev-vectors` - precomputed vectors for the fr-fr dev data.
- `--ar-dev-vectors` - precomputed vectors for the ar-ar dev data.
- `--zh-dev-vectors` - precomputed vectors for the zh-zh dev data.
***
- `--en-test-vectors` - precomputed vectors for the en-en test data.
- `--ru-test-vectors` - precomputed vectors for the ru-ru test data.
- `--fr-test-vectors` - precomputed vectors for the fr-fr test data.
- `--ar-test-vectors` - precomputed vectors for the ar-ar test data.
- `--zh-test-vectors` - precomputed vectors for the zh-zh test data.
***
- `--semcor-data` - SemCor transformed WiC data.
- `--semcor-gold` - SemCor transformed WiC gold labels.
- `--semcor-vectors` -  precomputed vectors for the SemCor transformed WiC data.
***
- `--threshold-grid-start` - start point for the threshold grid search.
- `--threshold-grid-stop` - stop point for the threshold grid search.
- `--threshold-grid-num` - number of points in the threshold grid search.
***
- `--output-dir` - path to the directory where to store predictions.

### Step 2: Evaluate predictions
Run `python test_score.py` to calculate test scores for a prediction file.
Additionally, you can change the default parameters: 
- `--en-test-pred` - MCL-WiC en-en test predictions.
- `--ru-test-pred` - MCL-WiC ru-ru test predictions.
- `--fr-test-pred` - MCL-WiC fr-fr test predictions.
- `--ar-test-pred` - MCL-WiC ar-ar test predictions.
- `--zh-test-pred` - MCL-WiC zh-zh test predictions.
***
- `--en-test-gold` - MCL-WiC en-en test gold labels.
- `--ru-test-gold` - MCL-WiC ru-ru test gold labels.
- `--fr-test-gold` - MCL-WiC fr-fr test gold labels.
- `--ar-test-gold` - MCL-WiC ar-ar test gold labels.
- `--zh-test-gold` - MCL-WiC zh-zh test gold labels.

## Authors
- Maxim Rachinskiy
- Nikolay Arefyev