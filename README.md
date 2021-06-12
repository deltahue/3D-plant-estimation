# 3D-plant-estimation

Smart farming technologies can revolutionize food pro-duction and help the farming industry answer the world’sgrowing demand for food in a sustainable way. 3D visionmethods are a promising tool which can be used for smartfarming applications. In this project, an end-to-end pipelinewas developed, which uses 3D reconstruction and machinelearning methods to extract three crop traits from a plants:the plant height, leaf area and leaf angle distribution. Theresulting method is able to estimate height within an errorof1cmfor the datasets used.

## Prerequisites

    This project has been tested on Ubuntu 18.04 and Python 3.x.

##### Git
For the following installation instructions you will need git which you can set up with the following commands:

    ```bash
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
        git checkout development
    ```

2. Create a anaconda environemnt
    ```bash
        conda env create -f finalEnv.yml -n ENVNAME
        conda activate ENVNAME
    ```

3. Import data into data folder. Use the structure:
    
    ├── README.md        <- The top-level README for developers using this project.
    ├── data             <- Data directory
    │   └── images       <- Directory with all image (na naming convention is required)
    │       ├── img_1 
    :       ├── img_2 
    :       :   im 
    :       └── img_2 


4. Run colmap:
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

5. You can choose whether you wish to see the metrics of the avocado dataset or the luca2 dataset

    python3 ./3d_plant_estimation.py avo_6

    python3 ./3d_plant_estimation.py luca2

## TODO:
Licenses?
Contribute?
