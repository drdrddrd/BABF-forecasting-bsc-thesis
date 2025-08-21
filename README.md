# BABF Forecasting (BSc Thesis)

This repository contains the code for a BSc thesis that forecasts changes in Swiss forests using a machine learning model. An XGBoost algorithm was trained on data from the Swiss National Forest Inventory (NFI) and CH2018 climate scenarios to predict future basal area and broadleaf fraction until 2099.



Follow these steps to set up the project and reproduce the results.

## Data

This project requires the `data.zip` file. Unzip it in the repository root. This will create the `data/` directory containing:
- Raw NFI datasets.
- Outputs from computationally intensive steps (e.g., raw CH2018 data preprocessing, that were run on the euler cluster.

**Original Data Sources:**
- **NFI:** [Swiss National Forest Inventory](https://www.lfi.ch/en/services/data-supply)
Citation: WSL, 2025: Schweizerisches Landesforstinventar LFI. Daten der Erhebung(en) 1993-95 / 2004-06 / 2009-2017 / 2018-2024. Christian Temperli. Eidg. Forschungsanstalt, Bimensdorf
- **CH2018:** [Swiss Climate Change Scenarios](https://www.nccs.admin.ch/nccs/en/home/climate-change-and-impacts/swiss-climate-change-scenarios/ch2018---climate-scenarios-for-switzerland.html)
Citation:  CH2018 Project Team (2018): CH2018 - Climate Scenarios for Switzerland. National Centre for Climate Services. doi: 10.18751/Climate/Scenarios/CH2018/1.0

## Environment

Set up the Conda environment using the provided file.

```bash```
conda env create -f BABF_env.yaml
conda activate tf

## Workflow to Reproduce Results

This project is an ordered pipeline. Execute the following steps from the repository root to reproduce the thesis results.
| #  | Step (file)                                    | Purpose                                                                                                                                                             |
|----|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 01 | `notebooks/NFI_preprocessing_until_2099.ipynb` | preprocess nfi data with a template until 2099 for combining with the climate data  data.                                                                                           |
| 02 | `scripts/euler_CH2018_processing.py`           | takes raw ch2018 files and calculates yearly metrics for selected coordinates (was done on the euler cluster, output files are in data.zip). not runnable because data is too large`.                                      |
| 03 | `notebooks/CH2018_preprocessing.ipynb`         | further preprocesses climate data and combine with nfi data data.                                                                                                                      |
| 04 | `notebooks/NFI_CH2018_plots.ipynb`             | create plots with nfi and ch2018 data dataset.                                                                                                                  |
| 05 | `scripts/hyperparameter_tuning.py`             | tune hyperparameters (was done on the euler cluster, the tuned hyperparameters are implemented in`model_evaluation.ipynb` and in `prediction_until_2099`). run once only with INVNR 150,250,350 (for model evaluation) and once with INVNR 150,250,350,450 (for iterative forecasting)                                |
| 06 | `notebooks/model_evaluation.ipynb`             | train and evaluate models.                                                                                              |
| 07 | `scripts/prediction_until_2099.py`             | make predictions until 2099 (was done on the euler cluster, output files are in data.zip). has to be run for each RCP scenario seperately                                   |
| 08 | `notebooks/prediction_plots.ipynb`             | make plots of predicted values until 2099 thesis.                                                                               |


## How to run

### Notebooks
jupyter lab   # then "Run All" in each numbered notebook

### Scripts (example)
python scripts/hyperparameter_tuning.py

## Outputs
Final plots are saved to results/plots/ and prediction data is saved as data/final_predictions.csv.

## Author
Dea Rieder
Bsc. Thesis, ETHZ, 2025