import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_generation_simplified as data
from acceptance_loop import AcceptanceLoop
from reject_inference import RejectInference, BiasAwareSelfLearning, HeckmanBivariate, HeckmanTwoStage, LabelAllRejectsAsBad, Reweighting
from Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric
from sklearn.linear_model import LogisticRegression

######### INITIAL POPULATION ########
# generate data
res = data.DataGenerator(
    n = 100,
    bad_ratio = 0.7,
    k_con = 10,
    k_bin = 0,
    con_nonlinear = 0,
    #con_mean_bad_dif = [2,1],
    con_var_bad_dif = 0.5,
    #covars = [[1, 0.2, 0.2, 1],[1, 0.2, 0.2, 1]],
    seed = 77,
    verbose = True
)
res.generate()

# extract initial population
init_population = res.data
init_population.drop(columns=['B1'], inplace=True)
init_population['BAD'].map({'BAD':1, 'GOOD':0})

top_percent = 0.2

# accept applicants using a business rule on X1
# Identify indices of the top `top_percent` based on 'X1'
accepts_ind = init_population.index[init_population['X1'] >= init_population['X1'].quantile(1 - top_percent)]

# Select rows for current accepts and rejects
init_accepts = init_population.loc[accepts_ind]
init_rejects = init_population.drop(accepts_ind)

# check if both classes are present
if (init_accepts['BAD'] == 'BAD').sum() < 4:
    top_indices = init_population.index[init_population['X1'] >= init_population['X1'].quantile(1 - top_percent)]
    accepts_ind = list(top_indices[:len(accepts_ind) - 4]) + list(init_population.index[:4])
    init_accepts = init_population.loc[accepts_ind]
    init_rejects = init_population.drop(accepts_ind)

######### HOLDOUT POPULATION ########

holdout_sample = 3000

# generate holdout data
holdout = data.DataGenerator(
    n = holdout_sample,
    replicate = res,
    seed = 99999,
)
holdout.generate()
holdout_population = holdout.data

# prepare X and y
x_holdout = holdout_population.drop(columns=['BAD'])
y_holdout = holdout_population['BAD'].map({"BAD":1, "GOOD":0})

#######################
#
# ACCEPTANCE LOOP
#
#######################

acceptance_loop = AcceptanceLoop(
    n_iter = 300,
    current_accepts= init_accepts,
    current_rejects= init_rejects,
    holdout= holdout_population,
    res=res,
    top_percent= top_percent)

_, _, _, _, final_accepts, final_rejects = acceptance_loop.run()

# ============================================================
# Benchmark Models: LABAR + Reweighting + Heckman Model
# ============================================================

# Instantiate and fit LABAR
labar_model = LabelAllRejectsAsBad(
    strong_estimator=LogisticRegression(max_iter=1000, random_state=42),
    silent=False
)

labar_model.fit(
    accepts_x=final_accepts.drop(columns=["BAD"]),
    accepts_y=final_accepts["BAD"].map({"BAD": 1, "GOOD": 0}),
    rejects_x=final_rejects.drop(columns=["BAD"])
)

# Instantiate and fit Reweighting
reweighting_model = Reweighting(
    strong_estimator=LogisticRegression(max_iter=1000, random_state=42),
    silent=False
)

reweighting_model.fit(
    accepts_x=final_accepts.drop(columns=["BAD"]),
    accepts_y=final_accepts["BAD"].map({"BAD": 1, "GOOD": 0}),
    rejects_x=final_rejects.drop(columns=["BAD"])
)

# Instantiate and fit Heckman Model 
heckman_bivariate = HeckmanBivariate(top_k=5)
heckman_bivariate.fit(accepts_x=final_accepts.drop(columns=['BAD']),
                      accepts_y=final_accepts['BAD'].map({"BAD": 1, "GOOD": 0}),
                      rejects_x=final_rejects.drop(columns=['BAD']))

heckman_two_stage = HeckmanTwoStage(selection_classifier='Probit', outcome_classifier='XGB')
heckman_two_stage.fit(accepts_x=final_accepts.drop(columns=['BAD']),
                      accepts_y=final_accepts['BAD'].map({"BAD": 1, "GOOD": 0}),
                      rejects_x=final_rejects.drop(columns=['BAD']))

fitted_models = {
    "Heckman-Bivariate": heckman_bivariate,    
    "Heckman-TwoStage": heckman_two_stage,       
    "LARAB": labar_model,
    "Reweighting": reweighting_model
}

# Define metrics from Evaluation.py
metrics = {
    "AUC": AUC(),
    "BS": BS(),
    "PAUC": PAUC(),
    "ABR": ABR()
}


# Evaluation on holdout set with feature alignment
results = {}

for model_name, model in fitted_models.items():
    print(f"\nEvaluating: {model_name}")

    try:
        # Align features: ensure holdout uses same columns as accepts
        feature_cols = final_accepts.drop(columns=["BAD"]).columns
        x_holdout_aligned = x_holdout[feature_cols]

        # Predictions
        y_pred_holdout = model.predict(x_holdout_aligned.values)
        y_proba_holdout = model.predict_proba(x_holdout_aligned.values)

        # Positive class probabilities
        if y_proba_holdout.ndim == 2 and y_proba_holdout.shape[1] == 2:
            y_proba_positive_class = y_proba_holdout[:, 1]
        else:
            y_proba_positive_class = y_proba_holdout

        # Compute metrics
        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            if isinstance(metric_func, ABR):
                score = metric_func(
                    y_true=y_holdout.values,
                    y_proba=y_proba_positive_class,
                    y_pred=y_pred_holdout
                )
            else:
                score = metric_func(
                    y_true=y_holdout.values,
                    y_proba=y_proba_positive_class
                )
            results[model_name][metric_name] = score

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        results[model_name] = {"Error": str(e)}

# Display results
results_df = pd.DataFrame(results)
print(results_df.T)
