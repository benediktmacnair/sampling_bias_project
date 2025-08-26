import pickle
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_generator as data
from acceptance_loop import AcceptanceLoop
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric
from reject_inference import RejectInference, BiasAwareSelfLearning
import random
from scorecard_selection import ScorecardSelector
from code_02_simulation_results import SimulationResults

#######################
#
#    Data Preparing
#
#######################

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
    # Adjust indices to ensure at least 4 'BAD' cases
    # threshold = init_population['X1'].quantile(1 - top_percent)
    # accepts_ind = init_population.index[init_population['X1'] >= threshold].tolist()
    # main_part = accepts_ind[:len(accepts_ind) - 4]
    # first_four = list(init_population.index[:4])
    # accepts_ind = main_part + first_four

    top_indices = init_population.index[init_population['X1'] >= init_population['X1'].quantile(1 - top_percent)]
    accepts_ind = list(top_indices[:len(accepts_ind) - 4]) + list(init_population.index[:4])
    init_accepts = init_population.loc[accepts_ind]
    init_rejects = init_population.drop(accepts_ind)
    

# if not os.path.exists('plots'):
#     os.makedirs('plots')
sns.kdeplot(init_accepts.iloc[:, 0], fill= True, color='lime', label='Accepts', alpha=0.1)
sns.kdeplot(init_rejects.iloc[:, 0], fill= True, color='orange', label='Rejects', alpha=0.1)
sns.kdeplot(init_population.iloc[:, 0], fill= True, color='blue', label='Initial Population', alpha=0.1)
plt.xlabel('Population of Feature X1')
plt.title('Bias in Data Population')
plt.legend()
plt.grid(True)
# plt.savefig(f'plots/population.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.close()



######### HOLDOUT POPULATION ########

holdout_sample = 3000

# generate holdout data
holdout = data.DataGenerator(n = holdout_sample, replicate = res, seed = 99999)
holdout.generate()
holdout_population = holdout.data

# x_holdout = holdout_population.drop(columns=['BAD', 'B1'])
# y_holdout = holdout_population['BAD'].map({"BAD":1, "GOOD":0})

######### Visualize Data Set #########
fig, ax = plt.subplots(1,3, figsize=(18, 6))
ax[0].scatter(holdout_population['X1'], holdout_population['X2'], c=holdout_population['BAD'].map({'BAD':1, 'GOOD':0}), cmap='plasma', alpha=0.6, s=50, label="BAD")
ax[0].set_title('Scatter Plot for Holdout Population')
ax[0].legend()
ax[0].grid()

ax[1].scatter(init_population['X1'], init_population['X2'], c=init_population['BAD'].map({'BAD':1, 'GOOD':0}), cmap='plasma', alpha=0.6, s=50, label="BAD")
ax[1].set_title('Scatter Plot for Initial Population')
ax[1].legend()
ax[1].grid()

ax[2].scatter(init_accepts['X1'], init_accepts['X2'], c=init_accepts['BAD'].map({'BAD':1, 'GOOD':0}), cmap='plasma', alpha=0.6, s=50, label="BAD")
ax[2].set_title('Scatter Plot for Initial Accepts')
ax[2].legend()
ax[2].grid()

plt.tight_layout()
plt.show()


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

stats_list, holdout_accept_preds_bad, basl_preds_bad, oracle_preds_bad, _, _ = acceptance_loop.run()
sns.kdeplot(holdout_accept_preds_bad, fill= True, color='#00FF00', label='Accepts', alpha=0.1)
sns.kdeplot(basl_preds_bad, fill= True, color='orange', label='BASL', alpha=0.1)
sns.kdeplot(oracle_preds_bad, fill= True, color='blue', label='Oracle', alpha=0.1)
plt.xlabel('Predicted P(BAD)')
plt.title('Score Distribution Density')
plt.legend()
plt.xlim(0, 1)
plt.grid(True)

if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig(f'Bias in prediction.png', dpi=300, bbox_inches='tight')

plt.show()
plt.close()

with open('stats_list.pkl', 'wb') as f:
    pickle.dump(stats_list, f)

########################
#
# SCORECARD SELECTION
#
########################

selector = ScorecardSelector(current_accepts = acceptance_loop.current_accepts,
                             current_rejects = acceptance_loop.current_rejects,
                             hold_population = holdout_population.drop(columns=["B1"]))

eval_scores, models = selector.run_selection()

with open('eval_scores.pkl', 'wb') as f:
    pickle.dump(eval_scores, f)

#################################################
#
# Visualize the Impact of Bias in Acceptance Loop
#
#################################################

#### Impact of Bias on Training ####
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6,5))
plt.plot(stats_list.index, stats_list["abr_inference"], label="BASL", color='orange')
plt.plot(stats_list.index, stats_list["abr_unbiased"], label="Oracle", color='blue')
plt.plot(stats_list.index, stats_list["abr_accepts"], label="Accepts", color='lime')
plt.legend()
plt.title('Impact of Bias on Training')
plt.xlabel('Acceptance Loop Iteration')
plt.ylabel('ABR on Holdout Sample')
plt.legend()
plt.tight_layout()
plt.show()

#### Impact of Bias on Evaluation ####
eval_stats = acceptance_loop.eval_stats_list
sns.set_theme(style="whitegrid")
plt.figure(figsize=(6,5))
plt.plot(eval_stats.index, eval_stats["abr_bayesian_eval"], label="BE", color='orange')
plt.plot(eval_stats.index, eval_stats["abr_oracle_eval"], label="Oracle", color='blue')
plt.plot(eval_stats.index, eval_stats["abr_accept_eval"], label="Accepts", color='lime')
plt.legend()
plt.title('Impact of Bias on Evaluation')
plt.xlabel('Acceptance Loop Iteration')
plt.ylabel('ABR on Holdout Sample')
plt.legend()
plt.tight_layout()
plt.show()

################################
#
# Visualize Distribution of data
#
################################

holdout = holdout_population.drop(columns=['B1'])
curr_accepts = acceptance_loop.current_accepts
curr_rejects = acceptance_loop.current_rejects

stat_list = acceptance_loop.stats_list
eval_list = eval_scores

test = SimulationResults(holdout, curr_accepts, curr_rejects, stat_list, eval_list)
test.feature_density()
test.target_density()
test.pairs_plot('holdouts')
test.basl_gain()
test.be_gain()