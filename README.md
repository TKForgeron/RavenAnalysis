# RavenAnalysis
This repository hosts all code used for the research of raven behaviour. It includes Python scripts that prepare the data and R scripts that are used for statistical analysis.

## Requirements
- Python 3.x
- pip packages: `numpy`, `openpyxl`, and `pandas`.
- The required packages can be installed into a virtual environment using the `requirements.txt` file:
    1. Execute `py -m venv .env` to install a virtual environment in the current directory
    2. Execute `source .env/Scripts/activate` (Windows) to activate the virtual environment (Linux: `source .env/bin/activate`)
    3. Execute `pip install -r requirements.txt` to install all needed packages

## Usage Guide
- Place your data (all Excel files from BORIS) into the folder `./data/raw`
- Run `py main.py` to process all the files so that it can be used for analysis in R
- The processed files may be found in `./data/processed`