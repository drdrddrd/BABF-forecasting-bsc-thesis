# BABF-forecasting-bsc-thesis
**Modeling Changes in Basal Area and Broadleaf Fraction in Swiss Forests Using Machine Learning**

## Data & sources
- NFI: https://www.envidat.ch/organization/nfi  
- CH2018: https://www.nccs.admin.ch/nccs/en/home/climate-change-and-impacts/swiss-climate-change-scenarios/ch2018---climate-scenarios-for-switzerland.html

> I will send `data.zip` containing the **`data/` folder structure and the necessary datasets**. The raw climate data is too big so you cant run step 2.

## Environment
```bash
# from repo root
conda env create -f BABF_env.yaml
# activate the environment name defined inside BABF_env.yaml
conda activate tf


## Workflow (notebooks + scripts)

This project runs as an **ordered pipeline**. Some steps are Jupyter notebooks, others are Python scripts. Execute them **from the repo root** in the order below.

### Order & purpose

| #  | Step (file)                          | What it does                        | 
|----|--------------------------------------|-------------------------------------|
| 01 | `notebooks/NFI_preprocessing_until_2099.ipynb`   | preprocess nfi data with a template until 2099 for combining with the climate data                   | 
| 02 | `scripts/euler_CH2018_processing.py`              | takes raw ch2018 files and calculates yearly metrics for selected coordinates (was done on the euler cluster). not runnable because data is too large              |
| 03 | `CH2018_preprocessing.ipynb` | further preprocesses climate data and combine with nfi data                   |
| 04 | `NFI_CH2018_plots.ipynb`                   | create plots with nfi and ch2018 data                        |
| 05 | `hyperparameter_tuning.py`                | tune hyperparameters (was done on the euler cluster). run once only with INVNR 150,250,350 (for model evaluation) and once with INVNR 150,250,350,450 (for iterative forecasting)                  |
| 06 | `model_evaluation.ipynb`    | evaluate models              |
| 07 | `prediction_until_2099`    | make predictions until 2099 (was done on the euler cluster)             |
| 08 | `prediction_until_2099`    | make predictions until 2099 (was done on the euler cluster)             |

> Running steps **01–08** will reproduce the results form the thesis.

### How to run

# Notebooks
jupyter lab   # then "Run All" in each numbered notebook

# Scripts (examplee)
python scripts/hyperparameter_tuning.py