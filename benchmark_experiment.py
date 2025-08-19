import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_generation_simplified as data
from acceptance_loop import AcceptanceLoop
from reject_inference import RejectInference, BiasAwareSelfLearning,HeckmanBivariate,HeckmanTwoStage
from Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric


######### INITIAL POPULATION ########
# generate data
res = data.DataGenerator(
    n = 100,
    bad_ratio = 0.7,
    k_con = 2,
    k_bin = 0,
    con_nonlinear = 0,
    con_mean_bad_dif = [2, 1],
    con_var_bad_dif = 0.5,
    covars = [[1, 0.2, 0.2, 1], [1, -0.2, -0.2, 1]],
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
holdout =data.DataGenerator(n = holdout_sample, replicate = res, seed = 99999)
holdout.generate()
holdout_population = holdout.data

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

#######################
#
# Benchmark Comparasion
#
#######################
#Fitting benchmark
heckman_bivariate = HeckmanBivariate()
heckman_bivariate.fit(accepts_x=final_accepts.drop(columns=['BAD']).values,
                      accepts_y=final_accepts['BAD'].map({"BAD": 1, "GOOD": 0}).values,
                      rejects_x=final_rejects.drop(columns=['BAD']).values)
heckman_two_stage = HeckmanTwoStage(selection_classifier='Probit', outcome_classifier='XGB')
heckman_two_stage.fit(accepts_x=final_accepts.drop(columns=['BAD']).values,
                      accepts_y=final_accepts['BAD'].map({"BAD": 1, "GOOD": 0}).values,
                      rejects_x=final_rejects.drop(columns=['BAD']).values)



#Evaluating benchmark
metric_auc = AUC()
metric_bs = BS()
metric_pauc = PAUC()
metric_abr = ABR()

metrics = {
    "AUC": metric_auc,
    "BS": metric_bs,
    "PAUC": metric_pauc,
    "ABR": metric_abr
}

fitted_models = {
    "Heckman Bivariate": heckman_bivariate,
    "Heckman Two-Stage": heckman_two_stage
}

results = {}

for model_name, model in fitted_models.items():
    print(f"\nEvaluating: {model_name}")

    try:
        x_holdout_values = x_holdout.values if isinstance(x_holdout, pd.DataFrame) else x_holdout
        y_pred_holdout = model.predict(x_holdout_values)
        y_proba_holdout = model.predict_proba(x_holdout_values)

        # Ensure y_proba_holdout is probabilities for the positive class (if it's a 2D array)
        if y_proba_holdout.ndim == 2 and y_proba_holdout.shape[1] == 2:
             y_proba_positive_class = y_proba_holdout[:, 1]
        else:
             y_proba_positive_class = y_proba_holdout # Assume it's already positive class probabilities

        results[model_name] = {}
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_true=y_holdout.values, y_pred=y_pred_holdout, y_proba=y_proba_positive_class)
            results[model_name][metric_name] = score

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        results[model_name] = {"Error": str(e)}


# Display results as a DataFrame
results_df = pd.DataFrame(results)
results_df.T