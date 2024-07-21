# An Independent Validation of the Kidney Failure Risk Equation (KFRE) on a Seoul University Hospital Population Using Python

## Overview

This repository contains the work focused on independently validating the Kidney Failure Risk Equation (KFRE) developed by Tangri et al. using an anonymized patient dataset from Seoul University Hospital. The primary goal is to reproduce the KFRE probabilities and evaluate their applicability to the South Korean population. This project utilizes a [custom Python library for KFRE calculations](https://pypi.org/project/kfre), complemented by exploratory data analysis (EDA) on demographics and clinical characteristics.

-----------

## Table of Contents

1. [Introduction](#introduction)
    - [Repository Structure](#repository-structure)
2. [KFRE Python Library](#kfre-python-library)
3. [Data Description](#data-description)
4. [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
    - [Data Exploration and Preprocessing: Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/data_exp_prep.ipynb)
    - [KFRE Validation: Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/kfre_validation.ipynb)
5. [KFRE Validation](#kfre-validation)
6. [Installation](#installation)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)
9. [References](#references)

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
│ ├── esrd_functions.py  
│ └── .gitignore  
└── README.MD  


```


## KFRE Python Library

The KFRE calculations are performed using a custom Python library developed to replicate the original KFRE equations. This library is designed to be easy to use and integrate into various projects involving kidney failure risk estimation. 

[KFRE Python Library](https://pypi.org/project/kfre)

## Data Description

The dataset used in this project is an anonymized 16,619 patient dataset from Seoul University Hospital with. It includes various demographic and clinical characteristics necessary for calculating the KFRE, such as `age`, `sex`, `GFR`, `hypertension`, `albuminuria`, and other relevant clinical parameters.

## Data Exploration and Preprocessing

[Data Exploration and Preprocessing: Jupyter Notebook](https://github.com/lshpaner/seoul_kfre/blob/main/notebooks/data_exp_prep.ipynb)

EDA is conducted to understand the demographics and clinical characteristics of the patient population. This involves:

- Inspecting data types  
- Converting `ESRD` outcomes to years 
- Classifying CKD Stages based on `GFR`  
- Summarizing patient demographics (age, sex, etc.).  
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


## References

1. Tangri N, Grams ME, Levey AS, Coresh J, Appel LJ, Astor BC, Chodick G, Collins AJ, Djurdjev O, Elley CR, Evans M, Garg AX, Hallan SI, Inker LA, Ito S, Jee SH, Kovesdy CP, Kronenberg F, Heerspink HJL, Marks A, Nadkarni GN, Navaneethan SD, Nelson RG, Titze S, Sarnak MJ, Stengel B, Woodward M, Iseki K, for the CKD Prognosis Consortium. (2016). *Multinational assessment of accuracy of equations for predicting risk of kidney failure: A meta-analysis. JAMA,* **315**(2), 164–174. doi: 10.1001/jama.2015.18202.

2. Tangri, N., Stevens, L. A., Griffith, J., Tighiouart, H., Djurdjev, O., Naimark, D., Levin, A., & Levey, A. S. (2011). *A predictive model for progression of chronic kidney disease to kidney failure. JAMA,* **305**(15), 1553-1559. doi: 10.1001/jama.2011.451.  

3. Shpaner, L. (2024). *kfre: A Python library for reproducing Kidney Failure Risk Equations (KFRE)* (Version 0.1.8) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.11100222

4. Kang MW, Tangri N, Kim YC, An JN, Lee J, Li L, Oh YK, Kim DK, Joo KW, Kim YS, Lim CS, Lee JP. (2020). *An independent validation of the kidney failure risk equation in an Asian population. Scientific Reports,* **10**(1), 12920. doi: 10.1038/s41598-020-69715-3.



