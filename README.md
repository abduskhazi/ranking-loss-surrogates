# Ranking Loss Surrogates
Repository for MSc-Thesis work @ Representation Lab Uni-Freiburg.

# Setup
* Install anaconda/miniconda according to the [Installation Instructions]( https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html "Named link title")
* Clone this repository into a folder. `git clone <url> ./repo`.
* Go inside the `repo` folder. `cd repo`.
* Using the exported yml file create the required conda environment.
`conda env create -f conda_environment/environment.yml`
* If the environment is already created, update it using the command
`conda env update --file conda_environment/environment.yml --prune`
* This creates a conda environment called `thesis`. Activate the conda environment. `conda activate thesis`
* After activating `thesis` environment, you can run all the scripts in the repository.

# Notes
* To run `study_hpo.py` we need to download and extract hpob\_data in the HPO\_B folder. The location is given in the documentation of the submodule HPO\_B. 

# Thesis Duration
8.12.2021 - 8.06.2022
