# Deep Ranking Ensembles
Repository for Deep Ranking Ensembles for Hyperparameter Optimization (ICLR 2023 Conference Paper942).

# Setup
## Environment Setup
* Install anaconda/miniconda according to the [Installation Instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html "Named link title").
* Clone this repository into a folder. `git clone --recurse-submodules <url> ./repo`
* Go inside the `repo` folder. `cd repo`
* Create the required conda environment. `conda env create --file linux_environment.yml`
* If the environment is already created, update it using the command `conda env update --file linux_environment.yml --prune`.
* This creates a conda environment called `DRE`. Activate the conda environment using command `conda activate DRE`.
* After activating `DRE` environment, display the script usage help message using command `python DRE.py -h`.
* Please download the data as described in the next section before running the `DRE.py` script.

## Data Download
* Download the HPO\_B data from [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip). A file named `hpob-data.zip` will be downloaded.
* Extract `hpob-data.zip` to the location `./repo/HPO_B/hpob-data/`. Here the `./repo/HPO_B` folder is the location where the HPO\_B submodule is cloned.
* After successful extraction, all the required files (in the json format) will be present in the `./repo/HPO_B/hpob-data/` folder.
