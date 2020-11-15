<div align="center">
    <img width="200" src="https://innsida.ntnu.no/documents/10157/2546401449/ntnu_hoeyde_eng.png/9130ea3c-828a-497e-b469-df0c54e16bb5?t=1578568440350" />
</div>

# COVID-19 Clustering comparison

![Python CI](https://github.com/batherk/covid-19-clustering/workflows/Python%20CI/badge.svg)

This is a part of the course TDT4173 - Machine Learning at NTNU.

## Data set

This project uses the "Mortality risk of COVID-19"-dataset from Our World in Data
[https://ourworldindata.org/mortality-risk-covid](https://ourworldindata.org/mortality-risk-covid). The dataset contains
country-by-country data on mortality risk of the COVID-19 pandemic.

# Installation guide

## Prerequisites

- Python 3.8 (or newer)
- pip

## Installing dependencies

If you are using conda, run the following at the command-line:

```
conda install --file requirements.txt
```

If you have a virtual environment (venv) activated (or none at all), run the following at the command-line:

```
pip install -r requirements.txt
```

# File strucure

The files found in the `notebooks` folder are jupyter notebooks.
`data` contains raw csv files, as well as processed.

```
📂covid-19-clustering
┣ 📁.github (CI config)folder
┣ 📁.vscode (editor config)
┣ 📁data (raw and processed csv files)
┣ 📁models (persisted models with metadata)
┣ 📁notebooks (jupyter notebooks)
┣ 📁results (clustering assignement and metrics for each model)
┣ 📁src
┃ ┣ 📂model (Python scripts for training and presisting the models)
┃ ┣ 📜__init__.py
┃ ┣ 📜utils.py
┃ ┣ 📜visualization.py
┣ 📁tests
┣ 📜.flake8
┣ 📜.gitignore
┣ 📜project_proposal.md
┣ 📜README.md (this file)
┣ 📜requirements.txt (3rd-party packages)

```
