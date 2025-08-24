# Fighting Sampling Bias: Advanced ML Tools for Reject Inference

This repository contains the code and resources for a project developed as part of the **Applied Predictive Analysis** seminar at **Humboldt University** in the 2025 summer semester. Our objective is to **re-implement** the proposed methods and **replicate** the experimental results in Python from the paper [*Fighting sampling bias: A framework for training and evaluating credit scoring models( (Kozodoi, N., Lessmann, S., Alamgirf, M., Moreira-Matias, L., & Papakonstantinou, K., 2025)*](https://doi.org/10.1016/j.ejor.2025.01.040). 

## File Structure

The paper developed a **bias-aware self-labeling (BASL)** algorithm for scorecard training and a **Bayesian Evaluation (BE)** strategy for scorecard evaluation in sampling bias situation. A simulation framework is built to compare these two methods with benchmarks called **acceptance loop**. Following this main logic, we structure our repository as follow:

- `original R codes`: contains original R codes in of the paper
- `Python codes`:
    - **data_generator_simplified.py**: simulation data generator.
    - **reject_inference.py**: code for BASL and benchmark methods.
    - **Evaluation.py**: code for BE and evaluation metrics callable by benchmarks precessed in experiments.
    - **acceptance_loop**: simulation framework for experiments, which comparing BASL and BE with benchmarks in simulated credit scoring situation.
    - **scorecard_selection.py**: implement a evaluation loop to comparing scorecards based on xgboots.
    - **code_01_simulation_study.py**: our **main script** to implement experiments and generate plots.
    - **code_02_simulation_study.py**: distribution plots generator.
    - **benchmark_experiment.py**: conducts a benchmark experiment with synthetic data to compare BASL performance against benchmark models.
- `simulation data`: contains randomly generated initial population and holdout population data with two continuous variables.
- `results`: contains all plots we generated from Python codes.
- `README.md`

## How to run our code?

### Environment Requirements

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

### Code examples for running our Python codes:
The following files contain the implementation of the key modules and a comparison of the results from various models:

- **data_generator_simplified.py**:
The Data Generator creates synthetic applicant datasets for credit scoring research with controlled bias. It simulates continuous and binary features, assigns labels (GOOD/BAD) according to a specified bad rate, and allows adding noise and nonlinear transformations.
A generator is initialized with the desired number of applicants (n), the number of continuous (k_con) and binary (k_bin) features, and the target bad rate (bad_ratio). Additional parameters control the mean differences, variances, and probabilities that distinguish good and bad applicants, as well as random seeds for reproducibility. Optionally, one generator can be replicated to ensure consistent distributions when creating a holdout sample.
Calling .generate() produces a synthetic dataset stored in self.data, returned as a pandas DataFrame. The DataFrame contains one column for each continuous (X1, X2, …) and binary (B1, B2, …) feature, together with a BAD target column indicating whether an applicant is labeled "GOOD" or "BAD".

```python
from data_generation_simplified import DataGenerator

# Initialize generator
generator = DataGenerator(
    n=1000,
    k_con=5,
    k_bin=2,
    bad_ratio=0.3,
    seed=42
)

# Generate synthetic dataset
generator.generate()

# Access generated data
print(generator.data.head())
```
This example first creates a generator with 1,000 applicants, five continuous and two binary features, and a bad rate of 30%. After calling .generate(), the simulated dataset is accessible via generator.data. In practice, this dataset serves as the starting point for experiments and is later passed into the acceptance loop, which splits it into accepts, rejects, and a holdout sample for unbiased evaluation.
- **Evaluation.py**:  
This module provides an independent implementation of the **Bayesian Evaluation strategy**, which is designed to address sampling bias in model evaluation. For example, to calculate the **AUC metric** with the Bayesian strategy in a credit scoring scenario, you can use the following code. The example below uses `y_true_acc` (true targets for accepted applicants), `y_proba_acc` (predicted probabilities for accepted applicants), and `y_proba_rej` (predicted probabilities for rejected applicants).

```python
import numpy as np
from Evaluation import Metric, AUC, Evaluation, bayesianMetric

# Example Data (replace with your actual data)
y_true_acc = np.array([1, 0, 1, 0, 1])                  # True targets for accepted applicants
y_proba_acc = np.array([0.9, 0.2, 0.8, 0.3, 0.7])       # Predicted probabilities for accepted applicants
y_proba_rej = np.array([0.1, 0.2, 0.3, 0.4, 0.5])       # Predicted probabilities for rejected applicants

# Initialize the AUC metric
auc_metric = AUC()

# Use the Bayesian Metric class to calculate AUC
bm_auc = bayesianMetric(auc_metric)

# Calculate the AUC using the Bayesian Evaluation strategy
auc_bm = bm_auc.BM(y_true_acc = y_true_acc,
                   y_proba_acc = y_proba_acc,
                   y_proba_rej = y_proba_rej,
                   rejects_prior = y_proba_rej          # Can be a single float if rejects share the same rejection probability
                   )
print(f"The AUC metric based on Bayesian Evaluation strategy is {auc_bm:.4f}.")
```

- **reject_inference.py**  
  Contains an abstract `RejectInference` class, from which several concrete model subclasses are derived.  
  It houses the Bias-Aware Self-Learning (BASL) method along with benchmark models like Label-All-Rejects-as-Bad, Heckman Two-Stage, Heckman Bivariate, and Reweighting.  
  Each model inherits a standardized `fit()`, `predict()`, and `predict_proba()` method.

- **code_01_simulation_study.py**

- **code_02_simulation_results.py**

- **benchmark_experiment.py**  
  


## Constraint

- Figure **2.c** in the main paper cannot be replicated with the meta-parameters given in Appendix E.  
  Even when running the original R code with those parameters, the results deviate from the figure reported in the paper.


## Contribution and Acknowledgments 

## Citation

- **Gemini**: BASL and Heckman models in `reject_inference.py` were drafted with the help of Gemini 2.5.
- 
