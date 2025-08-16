import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import data_generation_simplified as data
from acceptance_loop import AcceptanceLoop
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from basl.Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric
from basl.reject_inference_14june import RejectInference, BiasAwareSelfLearning
import random

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
    

sns.kdeplot(init_accepts.iloc[:, 0], fill= True, color='lime', label='Accepts', alpha=0.1)
sns.kdeplot(init_rejects.iloc[:, 0], fill= True, color='orange', label='Rejects', alpha=0.1)
sns.kdeplot(init_population.iloc[:, 0], fill= True, color='blue', label='Initial Population', alpha=0.1)
plt.xlabel('Population of Feature X1')
plt.title('Bias in Data Population')
plt.legend()
plt.show()
plt.close()



######### HOLDOUT POPULATION ########

holdout_sample = 3000

# generate holdout data
holdout = data.DataGenerator(n = holdout_sample, replicate = res, seed = 99999)
holdout.generate()
holdout_population = holdout.data

# x_holdout = holdout_population.drop(columns=['BAD', 'B1'])
# y_holdout = holdout_population['BAD'].map({"BAD":1, "GOOD":0})


#######################
#
# ACCEPTANCE LOOP
# 
#######################

acceptance_loop = AcceptanceLoop(
    n_iter = 5,
    current_accepts= init_accepts,
    current_rejects= init_rejects,
    holdout= holdout_population,
    res=res,
    top_percent= top_percent)

stats_list = acceptance_loop.run()


with open('stats_list.pkl', 'wb') as f:
    pickle.dump(stats_list, f)