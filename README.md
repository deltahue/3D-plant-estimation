# 3D-plant-estimation

Smart farming technologies can revolutionize food pro-duction and help the farming industry answer the world’sgrowing demand for food in a sustainable way. 3D visionmethods are a promising tool which can be used for smartfarming applications. In this project, an end-to-end pipelinewas developed, which uses 3D reconstruction and machinelearning methods to extract three crop traits from a plants:the plant height, leaf area and leaf angle distribution. Theresulting method is able to estimate height within an errorof1cmfor the datasets used.

## Prerequisites

    This project has been tested on Ubuntu 18.04 and Python 3.x.

##### Git
For the following installation instructions you will need git which you can set up with the following commands:

    ```
    sudo apt-get install git
    git config --global user.name "Your Name Here"
    git config --global user.email "Same Email as used for github"
    git config --global color.ui true
    ```
    
##### Anaconda
For the following project you will need Anaconda which can be installed following [this tutorial](https://docs.anaconda.com/anaconda/install/linux/).

##### COLMAP
For the following project you will need [COLMAP](https://colmap.github.io/index.html) which can be installed following [this tutorial](https://colmap.github.io/install.html).

## Running the project:

1. Clone the repository:
    ```bash
    git clone git@github.com:deltahue/3D-plant-estimation.git
    cd 3D-plant-estimation
    git checkout development
    ```

2. Install Colmap
    ```bash
    cd src/colmap
    git submodule init
    git submodule update
    cd ..
    ```

3. Create a anaconda environemnt
    ```bash
    conda env create -f environment.yml -n ENVNAME
    conda activate ENVNAME
    ```

4. Import data into data folder. Use the structure:
```nohighlight
├── README.md        <- The top-level README for developers using this project.
├── data             <- Data directory
│   └── images       <- Directory with all image (na naming convention is required)
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
example:
    ```bash
        cd src
        ./3d_reconstruction.sh test
    ```
You can find precomputed data here:
https://polybox.ethz.ch/index.php/s/wEiLS1izwR2D8DG

the show option determines whether the results are written to the terminal or written to the results folder

python3 ./3d_plant_estimation.py -plantName luca2 -show True

python3 ./3d_plant_estimation.py -plantName palm

python3 ./3d_plant_estimation.py -plantName field

python3 ./3d_plant_estimation.py -plantName avo_6



    python3 ./3d_plant_estimation.py -plantName luca2 -show False

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
