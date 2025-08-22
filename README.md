# Repo Introduction

This repo provides tools and methodologies to address the issue of sample selection bias in credit risk modeling, based on the paper  
**“Fighting sampling bias: A framework for training and evaluating credit scoring models.”**

---

## Environment Requirements

- **Python Version**: Python 3.8 or higher is required. It is strongly recommended to use the version specified in the project's `requirements.txt` or documentation for compatibility.  
- **Operating System**: The project is designed to be cross-platform and should work on Windows and macOS.  
- **Core Libraries**: The project relies on standard Python libraries for data manipulation, numerical operations, machine learning, and visualization. These are listed in the `requirements.txt` file and include packages such as:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `scipy`
  - `statsmodels`
  - `matplotlib`
  - `seaborn`

---

## Core Modules

- **`reject_inference.py`**  
  Contains an abstract `RejectInference` class, from which several concrete model subclasses are derived.  
  It houses the Bias-Aware Self-Learning (BASL) method along with benchmark models like Label-All-Rejects-as-Bad, Heckman Two-Stage, Heckman Bivariate, and Reweighting.  
  Each model inherits a standardized `fit()`, `predict()`, and `predict_proba()` method.

- **`acceptance_loop.py`**

- **`evaluation.py`**

- **`data_generation_simplified.py`**

- …etc

---

## Implementation

The following files contain the implementation of the modules and a comparison of the results from various models:

- **`code_01_simulation_study.py`**
- **`code_02_simulation_results.py`**
- **`benchmark_experiment.py`**  
  Conducts a benchmark experiment with synthetic data to compare BASL performance against benchmark models.

---

## Constraint

- Figure **2.c** in the main paper cannot be replicated with the meta-parameters given in Appendix E.  
  Even when running the original R code with those parameters, the results deviate from the figure reported in the paper.

---

## Citation

- **Gemini**: BASL and Heckman models in `reject_inference.py` were drafted with the help of Gemini 2.5.
- 
