import pandas as pd
import numpy as np
import data_generation_simplified as data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

init_population = res.data

top_percent = 0.2 

# Identify indices of the top `top_percent` based on 'X1'
threshold = init_population.iloc[:, 0].quantile(1 - top_percent)
accepts_ind = init_population.index[init_population.iloc[:, 0] >= threshold]

# Select rows for current accepts and rejects
current_accepts = init_population.loc[accepts_ind]
current_rejects = init_population.drop(accepts_ind)

if (current_accepts['BAD'] == 'BAD').sum() < 4:
    # Adjust indices to ensure at least 4 'BAD' cases

    '''
    
    MISSING CODE!!
    
    '''

    current_accepts = init_population.loc[accepts_ind]
    current_rejects = init_population.drop(accepts_ind)


holdout_sample = 3000

holdout = data.DataGenerator(n = holdout_sample, replicate = res, seed = 99999)
holdout.generate()
holdout_population = holdout.data
x_holdout = holdout_population.drop(columns=['BAD'])
y_holdout = holdout_population['BAD']

#######################
#
# ACCEPTANCE LOOP
# 
#######################
n_iter = 300

stats = {
    'generation': range(1, n_iter + 1),
    'sample_size': None,
    'accept_ratio': None,
    'bad_ratio_accepts': None,
    'bad_ratio_rejects': None,
    'bad_ratio_unbiased': None,
    'auc_accepts': None,
    'auc_unbiased': None,
    'auc_inference': None
}
stats_list = []

for i in range (n_iter):
    
    if i % 10 == 0:
        print("Iteration {} of {}, : {} accepts and {} rejects".format(i, n_iter, len(current_accepts), len(current_rejects)))

    stats['sample_size'] = len(current_accepts) + len(current_rejects)
    stats['accept_ratio'] = len(current_accepts) / stats['sample_size']
    stats['bad_ratio_accepts'] = (current_accepts['BAD'] == 'BAD').mean()
    stats['bad_ratio_rejects'] = (current_rejects['BAD'] == 'BAD').mean()
    stats['bad_ratio_unbiased'] = stats['bad_ratio_accepts'] * stats['accept_ratio'] + stats['bad_ratio_rejects'] * (1 - stats['accept_ratio'])

    

    new_applicants = data.DataGenerator(n = 100, replicate = res, seed = 77 + i)
    new_applicants.generate()
    new_applicants = new_applicants.data

    x_new_applicants = new_applicants.drop(columns=['BAD'])
    y_new_applicants = new_applicants['BAD']

    ##### ACCEPTS-BASED SCORECARD
    '''
    y_accepts_binary 
    '''
    x_accepts = current_accepts.drop(columns=['BAD'])
    y_accepts = current_accepts['BAD']
    y_accepts_binary = y_accepts.map({"BAD":1, "GOOD":0})

    clf = LogisticRegression()
    clf.fit(x_accepts, y_accepts_binary)

    preds = clf.predict_proba(x_new_applicants)
    holdout_preds = clf.predict_proba(x_holdout)

    # preds = clf.predict(x_new_applicants)
    # holdout_preds = clf.predict(x_holdout)

    stats['auc_accepts'] = roc_auc_score(holdout_preds, y_holdout.map({"BAD":1, "GOOD":0}))


    ##### ORACLE SCORECARD
    combined_data = pd.concat([current_accepts, current_rejects], axis=0, ignore_index=True)

    x_combined = combined_data.drop(columns=['BAD'])
    y_combined = combined_data['BAD']
    y_combined_binary = (y_combined == 'GOOD').astype(int)    
    
    clf_unbiased = LogisticRegression()
    clf_unbiased.fit(x_combined, y_combined_binary)

    clf_unbiased.fit(x_combined, y_combined_binary)

    preds_unbiased = clf_unbiased.predict(x_new_applicants)
    holdout_preds_unbiased = clf_unbiased.predict(x_holdout)

    stats['auc_unbiased'] = roc_auc_score(holdout_preds_unbiased, y_holdout.map({"BAD":1, "GOOD":0}))

    stats_list.append(stats)

print("Hallo")



