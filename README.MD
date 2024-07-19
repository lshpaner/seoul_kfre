# An Independent Validation of the Kidney Failure Risk Equation (KFRE) on a Seoul University Hospital Population Using Python

## Overview

This repository contains the work focused on independently validating the Kidney Failure Risk Equation (KFRE) developed by Tangri et al. using an anonymized patient dataset from Seoul University Hospital. The primary goal is to reproduce the KFRE probabilities and evaluate their applicability to the South Korean population. This project utilizes a [custom Python library for KFRE calculations](https://pypi.org/project/kfre), complemented by exploratory data analysis (EDA) on demographics and clinical characteristics.

-----------

## Table of Contents

1. [Introduction](#introduction)
    - [Repository Structure](#repository-structure)
2. [KFRE Python Library](#kfre-python-library)
3. [Data Description](#data-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Data Exploration: Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/data_exp_prep.ipynb)
    - [KFRE Validation: Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/kfre_validation.ipynb)
5. [KFRE Validation](#kfre-validation)
6. [Installation](#installation)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)

-----------

## Introduction

The Kidney Failure Risk Equation (KFRE) is a predictive model developed by Tangri et al. to estimate the risk of kidney failure in patients with chronic kidney disease (CKD). This project aims to validate the KFRE on a different population, specifically patients from Seoul University Hospital, using an anonymized dataset. The validation process involves:

- Calculating KFRE probabilities using a custom Python library.  
- Performing exploratory data analysis (EDA) on patient demographics and clinical characteristics.  
- Evaluating the performance and applicability of the KFRE on this new dataset.

### Repository Structure

```bash
├── data_output/  
│ └── summary_tables.xlsx  
├── images/  
│ ├── png_images/  
│ └── svg_images/  
├── notebooks/  
│ ├── data_exp_prep.ipynb  
│ └── kfre_validation.ipynb  
├── py_scripts/  
│ ├── functions.py  
│ └── .gitignore  
└── README.MD  

```


## KFRE Python Library

The KFRE calculations are performed using a custom Python library developed to replicate the original KFRE equations. This library is designed to be easy to use and integrate into various projects involving kidney failure risk estimation. 

[KFRE Python Library](https://pypi.org/project/kfre)

## Data Description

The dataset used in this project is an anonymized 16,619 patient dataset from Seoul University Hospital with. It includes various demographic and clinical characteristics necessary for calculating the KFRE, such as `age`, `sex`, `GFR`, `hypertension`, `albuminuria`, and other relevant clinical parameters.

## Exploratory Data Analysis (EDA)

[Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/data_exploration.ipynb)

EDA is conducted to understand the demographics and clinical characteristics of the patient population. This involves:

- Summarizing patient demographics (age, gender, etc.).  
- Analyzing clinical characteristics (eGFR levels, albuminuria categories, etc.).  
- Visualizing distributions and relationships between different variables.


## KFRE Validation

[KFRE Validation: Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/kfre_validation.ipynb)

The validation process involves applying the KFRE equations to the patient dataset using the custom Python library and assessing the model's performance. This includes:

- Calculating KFRE probabilities for each patient.  
- Comparing the predicted probabilities with actual outcomes.  
- Evaluating the model's accuracy and applicability to the South Korean population.  

## Installation

- To use this repository, you need to have Python installed. You can then clone the repository and install the required dependencies.

```bash
git clone https://github.com/lshpaner/seoul_kfre.git
cd repository
pip install -r requirements.txt
```

## Results
The results of this project will be detailed in the results directory, including:

- Summary tables and visualizations from the EDA.
- Detailed KFRE validation results and performance metrics.

## Acknowledgements
This project is inspired by the work of Tangri et al. and supported by the data and support provided by Dr. Minwoo Kang and  Jung Pyo Lee. Special thanks to the individuals and institutions that have contributed to this project.



