<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h2 align="center">Reaction Outcome Prediction (Thesis)</h2>

  <p align="center">
    Using Graph Neural Network Encoder and Transformer Decoder
    <br />
    <a href="#about-the-project"><strong>Explore the docs »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setting-up-the-conda-environment">Setting up the conda environment</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#preparing-the-dataset-for-training">Preparing the dataset</a></li>
        <li><a href="#training-the-model">Training the classification model</a></li>
      </ul>
    </li>
    <li><a href="#project-organization">Project Organization</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This is a DL based approach devised for reaction outcome prediction as part of my IITM MTech thesis. It is an encoder-decoder approach where the encoder is a graph neural network while the decoder is a Transformer Decoder. Accordingly, the model takes in the LHS as a molecular graph and gives out text/SMILES corresponding to the RHS. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

- [Pytorch](https://pytorch.org//)
- [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/)
- [RDKit-Python](https://www.rdkit.org/docs/GettingStartedInPython.html)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started
Note that the code has been run and verified on Ubuntu as well as Windows systems. The instructions that follow are confirmed to be working on an Ubuntu system. 

### Setting up the conda environment

1. Create a new Anaconda environment
   ```sh
   conda create --name envname python=3.7
   ```

2. Install Pytorch 1.11 (1.11 is the latest version as of writing)
   ```sh
   conda install pytorch cudatoolkit=11.3 -c pytorch
   ```

3. Install Pytorch-Geometric suitable for Pytorch 1.11
   ```sh
   pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
   pip install torch-geometric
   ```

4. Install latest RDKit from conda-forge repository (2022.03.2 at the time of writing)
   ```sh
   conda install -c conda-forge rdkit
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

### Preparing the dataset for training

1. Place the training dataset (text file with reaction smiles) in data/raw as train.txt.
2. Activate the conda environment. 
   ```sh
    conda activate envname
    ```
4. From the root directory, run
   ```sh
   python src/prepare_dataset.py
   ```

### Training the model:

1.  Prerequisite: <a href="#preparing-the-dataset-for-training">Preparing the dataset for training</a>:
2.  Activate the conda environment.
    ```sh
    conda activate envname
    ```
3.  Edit src/configs/train_cfg.py file with the required training settings.
4.  From the root directory, run
    ```sh
    python src/train_pairwise.py
    ```
    
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- FOLDER STRUCTURE -->

## Project Organization
------------

    ├── LICENSE
    ├── README.md               <- The top-level README for developers using this project.
    ├── .gitignore              <- Git-ignore file. 
    |
    ├── kube_files              <- Contains files for running Kubernetes jobs
    │   ├── job.yaml            <- YAML file for creating and running a job
    │   ├── logs.txt            <- Text file for logging what gets printed
    │   ├── run.sh              <- The shell file that gets run in the job
    │   └── torch_debug.py      <- File for testing the environment within the job
    |
    ├── data
    │   ├── external            <- Data from third party sources.
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks for running explorations and tests.
    │
    ├── requirements.txt        <- The requirements file [UNUSED. FOLLOW INSTRUCTIONS ABOVE FOR ENVIRONMENT SETUP]
    │
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Scripts to help with dataset loading
    │   │   └── dataset.py                <- Python file containing class definitions for dataset loading
    │   │
    │   ├── configs             <- Folder containing config files for training, evaluation, etc
    │   │   └── train_cfg.py              <- Python file containing training settings
    |   |
    │   ├── models              <- Scripts containing model definitions
    │   │   ├── embedding_models.py       <- Python file containing Embedding class definitions
    │   │   └── mpnn_models.py            <- Python file containing graph neural network (GNN) model class definitions
    │   │   └── transformer_models.py     <- Python file containing Transformer model definitions.
    |   |
    │   ├── utils               <- Scripts for basic operations
    │   │   └── model_utils.py            <- Python file containing helpers for saving and loading models
    │   │   └── rdkit_utils.py            <- Python file containing helpers for RDKit operations
    │   │   └── torch_utils.py            <- Python file containing helpers for Pytorch operations
    │   │   └── vocab_utils.py            <- Python file containing helpers for building SMILES (RHS) vocabulary
    │   |
    │   ├── prepare_dataset.py  <- Python file to prepare dataset
    │   ├── train.py   <- Python file to train the classification model

--------

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
