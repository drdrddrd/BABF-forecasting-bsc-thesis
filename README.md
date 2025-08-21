# BABF Forecasting (BSc Thesis)

This repository contains the code for a BSc thesis that forecasts changes in Swiss forests using a machine learning model. An XGBoost algorithm was trained on data from the Swiss National Forest Inventory (NFI) and CH2018 climate scenarios to predict future basal area and broadleaf fraction until 2099.


Follow these steps to set up the project and reproduce the results.

## Data

To fully reproduce the results, you must first obtain the original raw data.

**Obtaining the Data:**
1. Swiss National Forest Inventory (NFI):
The NFI data is confidential and subject to a data usage agreement. It is not publicly downloadable.
- **How to get it:** You must request the data directly from the Swiss Federal Institute for Forest, Snow and Landscape Research (WSL). This typically involves signing a formal agreement ("Vereinbarung") for a specific, non-commercial research purpose.
- **Contact:** To initiate a request, contact the LFI team at WSL. More information can be found on the NFI website: (https://www.lfi.ch/en/services/data-supply)
- **Restrictions:** Redistribution of the raw data is strictly prohibited

2. Swiss Climate Change Scenarios (CH2018):
The CH2018 climate data is publicly available.
- **How to get it:** The dataset is available on request via a contact form: (https://www.nccs.admin.ch/nccs/en/home/climate-change-and-impacts/swiss-climate-change-scenarios/contact.html)

3. Swisstopo data:
The Swisstopo data is publicly available.
- **How to get it:** The data can be downloaded at (https://www.swisstopo.admin.ch/de/ubersichtskarten-der-schweiz) and (https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d)


**Citations:**
- **NFI:** WSL, 2025: Schweizerisches Landesforstinventar LFI. Daten der Erhebung(en) 1993-95 / 2004-06 / 2009-2017 / 2018-2024. Christian Temperli. Eidg. Forschungsanstalt, Bimensdorf
- **CH2018:** CH2018 Project Team (2018): CH2018 - Climate Scenarios for Switzerland. National Centre for Climate Services. doi: 10.18751/Climate/Scenarios/CH2018/1.0
- **Swisstopo:** 
    - **Overview Maps:** Federal Office of Topography (swisstopo). Overview maps of Switzerland (Übersichtskarten der Schweiz), 2024. Place: Wabern, Switzerland, Published: Web page, URL: https://www.swisstopo.admin.ch/de/ubersichtskarten-der-schweiz, Accessed: 2025-07-16
    - **Swissboundaries3d:** Federal Office of Topography (swisstopo). swissBOUNDARIES3D: Administrative boundaries of Switzerland and Liechtenstein, 2024. Place: Wabern, Switzerland Published: Webpage, URL: https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d, Accessed: 2025-07-16

## Environment

Set up the Conda environment using the provided file.

```bash
conda env create -f BABF_env.yml
conda activate tf
```

## Workflow to Reproduce Results

This project is an ordered pipeline. Execute the following steps from the repository root to reproduce the thesis results.
| #  | Step (file)                                    | Purpose                                                                                                                                                             |
|----|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 01 | `notebooks/NFI_preprocessing_until_2099.ipynb` | Preprocess NFI data using the template through 2099 to combine with the climate data.  data.                                                                                           |
| 02 | `scripts/euler_CH2018_processing.py`           | Takes raw CH2018 files and computes yearly metrics for selected coordinates (was run on the Euler cluster, output files are in `data.zip`). Not runnable here because the data is too large.                                      |
| 03 | `notebooks/CH2018_preprocessing.ipynb`         | Further preprocesses climate data and combines it with the NFI data.                                                                                                                      |
| 04 | `notebooks/NFI_CH2018_plots.ipynb`             | Create plots with the NFI and CH2018 dataset.                                                                                                                  |
| 05 | `scripts/hyperparameter_tuning.py`             | Tune hyperparameters (was run on the Euler cluster; the tuned hyperparameters are implemented in `model_evaluation.ipynb` and in `prediction_until_2099`). Run once with INVNR 150,250,350 (for model evaluation) and once with INVNR 150,250,350,450 (for iterative forecasting).                                |
| 06 | `notebooks/model_evaluation.ipynb`             | Train and evaluate models.                                                                                              |
| 07 | `scripts/prediction_until_2099.py`             | Make predictions until 2099 (was run on the Euler cluster; output files are in `data.zip`). Must be run for each RCP scenario separately.                                   |
| 08 | `notebooks/prediction_plots.ipynb`             | Make plots of predicted values until 2099 for the thesis.                                                                               |

## Outputs
Final plots are saved to results/plots/ and prediction data is saved as `data/final_predictions.csv`.

## Author
Dea Rieder
Bsc. Thesis, ETHZ, 2025