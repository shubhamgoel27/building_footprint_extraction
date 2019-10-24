# bfss
Building Footprint Segmentation from Satellite images

- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Project Setup](#project-setup)
- [Project description](#proj-des)
- [Salient features](#salient-fea)
- [To-Do](#to-do)

<a name="problem-statement"></a>
## 1. Problem Statement
Building extraction from satellite imagery has been a labor-intensive task for many organisations. This is specially true in developing nations (like India) where high resolution satellite images are still far from reach.

This project was conducted for extracting building footprints for tire-2 cities in India where the resolution for satellite imagery varies from _50cm to 65cm_. As the data is private, the same project flow has been implemented for a public [database](https://project.inria.fr/aerialimagelabeling/)

<a name="project-structure"></a>
## 2. Project Structure

```
bfss
  ├── train.py
  ├── config.py
  ├── evaluate.py
  ├── src  
  |   ├── training/
  |   ├── evaluation/
  |   ├── networks/   
  |   └── utils/
  ├── data
  |   ├── datasets/AerialImageDataset/
  |   └── test/
```

_Data_: <br>
the `data` folder is not a part of this git project as it was heavy. The same can be downloaded from below link:

```sh
https://project.inria.fr/aerialimagelabeling/
```

<a name="project-setup"></a>
## 3. Project Setup
To setup the virtual environment for this project do the following steps:

**Step 1:** ```cd bfss``` #Enter the project folder! <br />
**Step 2:** ```conda env create -f envs/bfss.yml``` #create the virutal environment from the yml file provided. <br />
**Step 3:** ```conda activate bfss``` #activate the virtual env.

<a name="proj-des"></a>
## 4. Project description
The training script is `train.py` <br>

The entire training configuration including the dataset path, hyper-parameters and other arguments are specified in `config.py`, which you can modify and experiment with. It gives a one-shot view of the entire training process. <br>

The training can be conducted in 3 modes:
 - Training from scratch
 - Training from checkpoint
 - Fine tuning
 - Transfer Learning

Explore `config.py` to learn more about the parameters you can tweak.

A Notebook has been created for explaining the different training steps: <br>
Jupyter-notebook [LINK](./notebook/training.ipynb)

<a name="salient-fea"></a>
## 5. Salient Features

  _**Infinity Generator**_
  Random tile generation at run time. This enables us to explode our dataset as random tiling can (at least theoretically) generate infinite unique tiles.

  _**Weighted Loss map**_
  My take on weighted loss map using the concept of signed distance function. Refer to [code](./src/training/metrics.py) for implementation.


<a name="to-do"></a>
## 6. To-Do

- [x] Data download and EDA.
- [x] Basic framework for evaluating existing models
- [x] Complete framework with mulitple models tested
- [ ] Pre and Post Processing techniques (on-going)
- [x] Transfer Learning framework
- [ ] Model training (On-Going)
- [ ] Conditional Random Field (CRF) for post-processing
