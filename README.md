# Fighting Sampling Bias: Advanced ML Tools for Reject Inference

This repository contains the code and resources for a project developed as part of the **Applied Predictive Analysis** seminar at **Humboldt University** in the 2025 summer semester. Our objective is to **re-implement** the proposed methods and **replicate** the experimental results in Python from the paper [*Fighting sampling bias: A framework for training and evaluating credit scoring models( (Kozodoi, N., Lessmann, S., Alamgirf, M., Moreira-Matias, L., & Papakonstantinou, K., 2025)*](https://doi.org/10.1016/j.ejor.2025.01.040). 

## File Structure

The paper developed a **bias-aware self-labeling (BASL)** algorithm for scorecard training and a **Bayesian Evaluation (BE)** strategy for scorecard evaluation in sampling bias situation. A simulation framework is built to compare these two methods with benchmarks called **acceptance loop**. Following this main logic, we structure our repository as follow:

- `original R codes`: contains original R codes in of the paper
- `Python codes`:
    - **data_generator.py**: simulation data generator.
    - **reject_inference.py**: code for BASL and benchmark methods.
    - **Evaluation.py**: code for BE and evaluation metrics callable by benchmarks precessed in experiments.
    - **acceptance_loop**: simulation framework for experiments, which comparing BASL and BE with benchmarks in simulated credit scoring situation.
    - **scorecard_selection.py**: implement a evaluation loop to comparing scorecards based on xgboots.
    - **code_01_simulation_study.py**: our **main script** to implement experiments and generate plots.
    - **code_02_simulation_study.py**: distribution plots generator.
    - **benchmark_experiment.py**: conducts a benchmark experiment with synthetic data to compare BASL performance against benchmark models.
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
  This file contains the core building blocks for all reject inference models used in the project. At its center is the abstract RejectInference class, which sets a simple standard: every model needs to implement fit(), predict(), and predict_proba(). This makes sure that no matter which method you’re testing, it can plug into the benchmark experiments without extra adjustments.

    On top of this base, the file implements both the proposed Bias-Aware Self-Learning (BASL) method and three benchmark models: Label-All-Rejects-as-Bad (LARAB), Reweighting, and the Heckman Two-Stage model. Each subclass handles rejected applicants differently, but they all follow the same interface, so they can be swapped in and out     easily during experiments.

- **data_generator.py**:
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

- **acceptance_loop.py**:
This module implements the core **simulation framework** for both Experiment I and II from the paper. It is designed to be used in conjunction with our data generator **data_generator_simplified.py**, which provides the necessary simulation data.

  The module's main function is to simulate the credit scoring process over multiple iterations, using data for both accepted and rejected applicants (`accepts` and `rejects`) to train and evaluate models.
  - **Experiment I**: In each iteration, an accepts-based scorecard is evaluated using three distinct strategies: `accepts-based evaluation`, `oracle evaluation`, and `Bayesian evaluation`. The evaluation results are saved in `eval_stats`.
  - **Experiment II**: In each iteration, three different scorecards are trained and evaluated: an `accepts-based scorecard`, an `oracle-based scorecard`, and a `corrected scorecard`. The evaluation metrics for these models are saved in `stats_list`.

  The module also outputs the final predictions and the full set of collected `accepts` and `rejects` after all iterations are complete.

  In the following example, `init_accepts`, `init_rejects`, and `holdout_population` are generated by **data_generator.py**. The full data preparation process is detailed in our main script **code_01_simulation_study.py**.

```python
from data_generation_simplified import DataGenerator
from acceptance_loop import AcceptanceLoop

# Define a data generator to create a population for the simulation
res = data.DataGenerator(
  n = 100,
  bad_ratio = 0.7,
  k_con = 2,
  k_bin = 0,
  con_nonlinear = 0,
  con_mean_bad_dif = [2,1],
  con_var_bad_dif = 0.5,
  covars = [[1, 0.2, 0.2, 1], [1, -0.2, -0.2, 1]],
  seed = 77,
  verbose = True
  )

# Initialize acceptance loop
acceptance_loop = AcceptanceLoop(
  n_iter = 300,                          # iteration times
  current_accepts = init_accepts,        # initial accepts is the current accepts in the first iteration
  current_rejects = init_rejects,        # initial rejects is the current rejects in the first iteration
  holdout = holdout_population,          # holdout set used as evaluation set in every iteration
  res = res,                             # res is a hyper parameter set of data generator
  top_percent = 0.2                      # top_percent is the accept ratio in population
  )

# Implement the acceptance loop and access the results of two experiments
stats_list, holdout_accept_preds_bad, basl_preds_bad, oracel_preds_bad, current_accepts, current_rejects = acceptance_loop.run()
print(f"The results for Experiment I are: \n {acceptance_loop.eval_stats_list}")
print(f"The results for Experiment II are: \n {stats_list}")
print(f"After {acceptance_loop.n_iter} times iteration, {len(acceptance_loop.current_accepts)} accepts and {len(acceptance_loop.current_rejects)} are collected.")
```
- **scorecard_selection.py**:
This module creates an experiment for comparison of different evaluation strategies using data sample collected in **acceptance_loop**. It outputs AUC metric of different XGBoost models with following evaluation strategies:
  - **Accepts**: evaluate model on accepts only.
  - **Bayesian**: evaluate model on accepts and rejects using Bayesian Evaluation strategy.
  - **Oracle**: evaluate model on holdout set.
  - **Rejects BAD**: evaluate model on evaluation set augmented with rejects assumed to be BAD
  - **Rejects GOOD**: evaluate model on evaluation set augmented with rejects assumed to be GOOD
  - **Average**: Average of Rejects BAD and Rejects GOOD
  - **weights**: weighted metrics value based on BAD ratio of rejects in all collected samples
In the following example, `acceptance_loop` is an implemented by our code **acceptance_loop.py**, and `holdout_population` is generated by **data_generator.py**.

```python
from scorecard_selection import ScorecardSelector

# Initialize scorecard selection module
selector = ScorecardSelector(
  current_accepts = acceptance_loop.current_accepts,
  current_rejects = acceptance_loop.current_rejects,
  hold_population = holdout_population.drop(columns=["B1"])
  )

# Implement scorecard selection
eval_scores, models = selector.run_selection()
```

- **code_02_simulation_study.py**:
This module replicate all plots in R code **code_02_simulation_results.R**. It visualize all results we generated from experiments. After implementing acceptance loop and scorecard selection, the results are saved in `curr_accepts`, `curr_rejects`, `stat_list` and `eval_list`. and can be visualized as following example:

```python
from code_02_simulation_results import SimulationResults

# Initialize a simulation results test
test = SimulationResults(holdout, curr_accepts, curr_rejects, stat_list, eval_list)

# Access different plots in paper
test.feature_density()
test.target_density()
test.pairs_plot('holdouts')
test.basl_gain()
test.be_gain()
```
- **benchmark_experiment.py**:
The Benchmark Experiment script ties the whole framework together and runs the main simulation used in our study. It begins by generating a synthetic applicant population with the Data Generator. Using a simple business rule, this population is then split into accepted and rejected applicants, while a separate holdout sample is created to serve as an unbiased reference point.

    From there, the script simulates the selective lending process and prepares the data for model training. For the Heckman Two-Stage model, it also performs a feature selection step to separate variables relevant for the acceptance decision from those driving default outcomes. Once the datasets are ready, several benchmark models are     trained — including Label-All-Rejects-as-Bad (LARAB), Reweighting, and Heckman Two-Stage — alongside the proposed Bias-Aware Self-Learning (BASL) method. Each approach is fit on the biased training data (accepts + rejects) and then tested on the unbiased holdout set.

    Model performance is measured using the metrics implemented in evaluation.py: AUC, Brier Score (BS), Partial AUC (PAUC), and Acceptance-Based Risk (ABR). Results are collected in a pandas DataFrame and printed to the console, making it easy to directly compare how well BASL performs relative to standard reject inference techniques     under the same simulation settings.

## Constraint

- Figure **2.c** in the main paper cannot be replicated with the meta-parameters given in Appendix E.  
  Even when running the original R code with those parameters, the results deviate from the figure reported in the paper.
- `HeckmanTwoStage` model performs poorly on our synthetic data primarily because of the high correlations present among some generated features. The high correlations among some features make it difficult for the model to isolate the true effect of each variable. This leads to unstable predictions and unreliable results. The issue is especially pronounced in multi-stage models, where errors in the initial stage can cascade and negatively affect the final outcome.


## Contribution

This project was completed by our team. The following are all the members and their key contributions:

**Yuanyuan Li**:
* **Evaluation & Selection Modules**: Developed and tested the core scripts for `Evaluation.py` and `scorecard_selection.py`.
* **Integrated Functions**: Coded the evaluation part within `acceptance_loop`, the scorecard selection and visualization components in `code_01_simulation_study.py`.
* **Reviewing**: Reviewed BASL part in `reject_inference.py`.

**Wai Yung LAU**:
* **Code Development and Architecture:** Architected the class structure for the reject inference module. Developed and validated the `BASL` and `HeckmanTwoStage` models within `reject_inference.py`. Constructed the core code framework for `benchmark_experiment.py`, implementing the benchmark training for `BASL` and `HeckmanTwoStage`, and evaluation pipelines.
* **Code Reviewing / Debug:** Conducted a code review and debugging of `acceptance_loop.py`. Reviewed the `LabelAllRejectsAsBad` model in `reject_inference.py`.
* **Presentation:** Prepared and presented slides on the reject inference for the final presentation. Wrote and presented the project background for the first presentation. Also prepared a script for the team.
* **Documentation:** Prepared the initial draft of the project's `README.md`, detailing the project introduction, environment requirements, an overview of `reject_inference.py`, and constraints.


## Citation

- **Gemini**: BASL and Heckman models in `reject_inference.py` were drafted with the help of Gemini 2.5.
