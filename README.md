# Deep learning from videography as a tool for measuring infection in poultry (under review)

Code and data for "Deep learning from videography as a tool for measuring infection in poultry". Release of video and physiological data is pending approval from the Department of Veterinary and Animal Sciences at the University of Copenhagen

## Setup

### Python

Version: 3.10.10

#### Python dependencies

The dependencies for downstream analyses are listed in ```env.yml```

You can install a virtual environment using ```conda``` by running:

```bash
conda env create -f env.yml
```

#### DeepLabCut data and training

Available soon

#### Feature extraction from DeepLabCut predictions

```bash
python -m dlc4ecoli.dlc.extract --path /path/to/data
```

#### Reproducing the figures

You can reproduce most figures by running the ```plots.ipynb``` notebook.

The other brms figures are created from the R script (see below).

### R

Version: 4.4.0

#### R dependencies

```R
install.packages(brms, envalysis, ggdist, ggplot2)
```

#### Mixed-effects modelling

Simply run the analysis.R script after setting the work directory to this repository

```bash
setwd("path/to/dlc2ecoli")
```
