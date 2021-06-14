# 3D-plant-estimation

Smart farming technologies can revolutionize food production and help the farming industry answer the world’s growing demand for food in a sustainable way. 3D-vision methods are a promising tool which can be used for smartfarming applications. In this project, an end-to-end pipeline was developed, which uses 3D reconstruction and machine learning methods to extract three crop traits from a plants:the plant height, leaf area and leaf angle distribution. The resulting method is able to estimate height within an error of 1 cm for the datasets used.

## Prerequisites

    This project has been tested on Ubuntu 18.04 and Python 3.8.

##### Git
For the following installation instructions you will need git which you can set up with the following commands:

    sudo apt-get install git
    git config --global user.name "Your Name Here"
    git config --global user.email "Same Email as used for github"
    git config --global color.ui true
    
##### Anaconda
For the following project you will need Anaconda which can be installed following [this tutorial](https://docs.anaconda.com/anaconda/install/linux/).

##### COLMAP
For the following project you will need [COLMAP](https://colmap.github.io/index.html) which can be installed following [this tutorial](https://colmap.github.io/install.html).

## Running the project:

1. Clone the repository:
    ```bash
    git clone git@github.com:deltahue/3D-plant-estimation.git
    cd 3D-plant-estimation
    ```

2. Clone the Colmap source code
    ```bash
    cd src/colmap
    git submodule init
    git submodule update
    cd ../..
    ```

3. Create an anaconda environment
    ```bash
    conda env create -f environment.yml -n ENVNAME
    conda activate ENVNAME
    ```

4. Import data into data folder. Use the structure:
    ```nohighlight
    ├── README.md        <- The top-level README for developers using this project.
    ├── data             <- Data directory
    │   └── images       <- Directory with all image (no naming convention is required)
    │       ├── img_1 
    :       ├── img_2 
    :       :   
    :       └── img_2 
    ```


5. Run colmap:
    ```bash
    cd src
    ./3d_reconstruction.sh DATA_NAME
    ```
##### Example:
    cd src
    ./3d_reconstruction.sh test
    
You can find precomputed data 
[here](https://polybox.ethz.ch/index.php/s/wEiLS1izwR2D8DG).
Put the data into the `data` folder.

The show option determines whether the results are written to the terminal or written to the results folder.
To run the main script `3d_plant_estimation.py` go to the `src` folder and run the Python script.

`python3 ./3d_plant_estimation.py -plantName luca2 -show True`

`python3 ./3d_plant_estimation.py -plantName luca2 -show False`

`python3 ./3d_plant_estimation.py -plantName palm`

`python3 ./3d_plant_estimation.py -plantName field`

`python3 ./3d_plant_estimation.py -plantName avo_6`



    

## Config Files
The config files have multiple parameters which can be set:
```
pathRoot --> path of the data folder
pathOrgans --> path for the results of the organ segmentation data folder
pathOrganClass --> path for the training data for the organ classifier
croppedPcdFilteredName --> path for the cropped pcd
hdbscanPath --> path for the hdbscan config
colmapFolderName --> path for the colmap python scripts
apriltagSide --> size of the april tag (in cm)
nb_points --> number of points for filtering
radius --> radius parameter for filtering
```
