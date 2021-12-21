# The context : ML4Science
### Interdisciplinary Machine Learning Projects Across Campus

As part of the Machine Learning Course CS-433, students can bring their ML skills to practice by joining forces with any lab on the EPFL campus, or other academic institutions, across any disciplines and topics of their choice. So far, 214 projects have been successfully completed since 2018.
The project is done in a group of 3, and counts 30% to the grade of the course.

# Our project : Solar panel delineation

### 1. Objectives
This project employs deep learning models to delineate solar panels in aerial imagery. 
We were given a dataset from the belgian government, in which some solar panels were delineated.
We want to use the dataset to train different architetures of neural networks and see which one performs best in this task.

### 2. Tools
For this project we will be inspiring ourselves from the following tutorial from [Segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb) <br />
Thanks to the different libraries used, combined with Pytorch, we will be able to easily implement very different architectures with different parameters and compare them.

Given the dataset is pretty small, we will test data augmentation using a library called 'Albumentations'.



### 3. Training and running

#### 3.1 Installation
The instructions below have been tested on Windows 10 (x86). They make use of the environment manager [Conda](http://conda.io/), although alternatives can also be employed with respective adaptations.

#### 3.2 Environment setup

1. Install dependencies
```bash
sudo apt update && sudo apt install -y build-essential gdal-bin libgdal-dev
```

2. Download and install [Conda](http://conda.io/) (Miniconda is enough) and create a new environment:

```bash
conda create -n solar_panels python=3.8 -y
conda activate solar_panels
```

3. Install basic requirements:
```bash
pip install -U -r requirements.txt
conda install -c conda-forge --name solar_panels ipykernel -y # to run the jupyter notebooks
```

#### 3.3 Training

To train the model, run the entire `Training.ipynb` notebook.


#### 3.4 Predictions

To make predictions, run the entire `inference.ipynb` notebook.

