# 3D-plant-estimation

## Running the project:
1) 

start from this folder
rename
run colmap
segmentation
rerunn pcl


git submodule init

git submodule update

cd ..

2) using anaconda create an environment using the finalEnv.yml file

conda env create -f finalEnv.yml

3) cd src

4) take the data folder from the link below and replace the current data folder with it:

https://polybox.ethz.ch/index.php/s/wEiLS1izwR2D8DG


5) You can choose whether you wish to see the metrics of the avocado dataset or the luca2 dataset

python3 ./3d_plant_estimation.py avo_6

python3 ./3d_plant_estimation.py luca2

## Git Conventions:
- Stick to the git workflow as much as possible: https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
- In order to merge a branch create a pull request for the others to review
- branch naming convention: /nethz/feature_name

## Repository Structure
```nohighlight
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling. (for example processed point clouds)
│   └── raw            <- The original, immutable data dump. (images, raw datasets)
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
