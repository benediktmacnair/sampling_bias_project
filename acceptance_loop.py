import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_generation_simplified import DataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from reject_inference import RejectInference, BiasAwareSelfLearning
from Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric

class AcceptanceLoop:
    def __init__(self, n_iter, current_accepts, current_rejects, holdout, res, top_percent):
        self.n_iter = n_iter,
        self.current_accepts = current_accepts,
        self.current_rejects = current_rejects,
        self.holdout = holdout,
        self.res = res,
        self.top_percent = top_percent,
        self.seed = 77,
        self._current_accepts = None,
        self._current_rejects = None,
        self.stats = {
            'sample_size': None,
            'accept_ratio': None,
            'bad_ratio_accepts': None,
            'bad_ratio_rejects': None,
            'bad_ratio_unbiased': None,
            'auc_accepts': None,
            'auc_unbiased': None,
            'auc_inference': None,
            'abr_accepts': None,
            'abr_unbiased': None,
            'abr_inference': None
            },
        self.stats_list = []
    
    
    def state_save(self):
        self.stats['sample_size'] = len(self.current_accepts) + len(self.current_rejects)
        self.stats['accept_ratio'] = len(self.current_accepts) / self.stats['sample_size']
        self.stats['bad_ratio_accepts'] = (self.current_accepts['BAD'] == 'BAD').mean()
        self.stats['bad_ratio_rejects'] = (self.current_rejects['BAD'] == 'BAD').mean()
        self.stats['bad_ratio_unbiased'] = self.stats['bad_ratio_accepts'] * self.stats['accept_ratio'] + self.stats['bad_ratio_rejects'] * (1 - self.stats['accept_ratio'])

    def clean_data(self, data):
        if 'B1' in data.columns:
            # Assuming 'BAD' is the target variable and 'B1' is a feature to drop
            data = data.drop(columns=['B1'])

        x_data = data.drop(columns=['BAD'])
        y_data = data['BAD'].map({"BAD": 1, "GOOD": 0})
        return x_data, y_data

    def generate_new_applicants(self, i):
        self.seed = self.seed + i + 1
        new_applicants = DataGenerator(n = 100, replicate = self.res, seed = self.seed)
        new_applicants.generate()
        x_new_applicants, y_new_applicants = self.clean_data(new_applicants.data)

        return x_new_applicants, y_new_applicants
    
    def train_clf(self, x_data, y_data):
        clf = LogisticRegression()
        clf.fit(x_data, y_data)

        return clf
    
    def basl_inference(self, accepts_x, accepts_y, rejects_x):
        filtering_beta=[0.01, 0.09]
        holdout_percent=0.1
        labeling_percent=0.1
        sampling_percent=0.8
        multiplier=2
        max_iterations=5
        early_stop=True
        weak_learner = LogisticRegression()
        strong_learner = LogisticRegression()

        basl = BiasAwareSelfLearning(strong_estimator = strong_learner,
                                    weak_learner_estimator = weak_learner,
                                    filtering_beta = filtering_beta,
                                    holdout_percent = holdout_percent,
                                    labeling_percent = labeling_percent,
                                    sampling_percent = sampling_percent,
                                    multiplier = multiplier,
                                    max_iterations = max_iterations,
                                    early_stop = early_stop, 
                                    silent=True)

        basl.fit(accepts_x= accepts_x,
                 accepts_y= accepts_y,
                 rejects_x= rejects_x)
        
        return basl
        
    
    def run(self):
        self.n_iter = self.n_iter[0]
        self.current_accepts = self.current_accepts[0]
        self.current_rejects = self.current_rejects[0]
        self.holdout = self.holdout[0]
        self.res = self.res[0]
        self.top_percent = self.top_percent[0]
        self.seed = self.seed[0]
        self.stats_list = []
        self.stats = self.stats[0]

        
        x_holdout, y_holdout = self.clean_data(self.holdout)
        
        for i in range(self.n_iter):
            self.state_save()
            x_curr_accept, y_curr_accept = self.clean_data(self.current_accepts)

            ####### NEW APPLICANTS GENERATION #######
            x_new_applicants, y_new_applicants = self.generate_new_applicants(i)

            ####### ACCEPT BASED SCORECARD #######
            clf = self.train_clf(x_curr_accept, y_curr_accept)
            accept_preds = clf.predict_proba(x_new_applicants)
            accept_preds_bad = accept_preds[:,1]
            accept_preds_good = accept_preds[:,0]
            # accept_preds_bad = accept_preds[:,0]
            # accept_preds = accept_preds
            holdout_accept_probs = clf.predict_proba(x_holdout)
            holdout_accept_preds_good = holdout_accept_probs[:,0]
            holdout_accept_preds_bad = holdout_accept_probs[:,1]
            # auc
            self.stats['auc_accepts'] = roc_auc_score(y_holdout, holdout_accept_preds_bad)
            # abr
            abr_metric = ABR()
            self.stats['abr_accepts'] = abr_metric(y_true=y_holdout, y_pred=None, y_proba=holdout_accept_preds_bad) 

            accept_preds = pd.DataFrame(accept_preds)
            accept_preds_good = pd.DataFrame(accept_preds_good)

            # coeff_accepts = clf.coef_[0]
            # intercept_accepts = clf.intercept_[0]


            ####### ORACLE BASED SCORECARD #######
            x_curr_reject, y_curr_reject = self.clean_data(self.current_rejects)
            x_oracle = pd.concat([x_curr_accept, x_curr_reject], axis=0, ignore_index=True)
            y_oracle = pd.concat([y_curr_accept, y_curr_reject], axis=0, ignore_index=True)

            clf_oracle = self.train_clf(x_oracle, y_oracle)
            oracle_preds = clf_oracle.predict_proba(x_holdout)
            oracle_preds_bad = oracle_preds[:,1]
            oracle_preds_good = oracle_preds[:,0]
            # auc
            self.stats['auc_unbiased'] = roc_auc_score(y_holdout, oracle_preds_bad)
            # abr
            abr_metric = ABR()
            self.stats['abr_unbiased'] = abr_metric(y_true = y_holdout, y_pred = None, y_proba = oracle_preds_bad)

            ####### CORRECTED SCORECARD #######
            clf_basl = self.basl_inference(x_curr_accept, y_curr_accept, x_curr_reject)
            basl_preds = clf_basl.predict_proba(x_holdout)
            basl_preds_bad = basl_preds[:,1]
            basl_preds_good = basl_preds[:,0]
            # auc
            self.stats['auc_inference'] = roc_auc_score(y_holdout, basl_preds_bad)
            # abr
            abr_metric = ABR()
            self.stats['abr_inference'] = abr_metric(y_true = y_holdout, y_pred = None, y_proba = basl_preds_bad)

            self.stats_list.append(self.stats.copy())

            ####### ACCEPTING NEW APPLICANTS #######
            if i <= self.n_iter-1:
                # select top accept applicants
                threshold = accept_preds_good.quantile(1 - self.top_percent)[0]
                accepts_ind = accept_preds_good.index[accept_preds_good[0] >= threshold].tolist()
                new_applicants = pd.concat([x_new_applicants, y_new_applicants], axis=1)
                new_accepts = new_applicants.loc[accepts_ind]

                n_accepts = round(self.top_percent * len(new_applicants))
                if len(accepts_ind) > n_accepts:
                    # Randomly sample the required number of rows from new_accepts
                    sampled_indices = np.random.choice(new_accepts.index, size=n_accepts, replace=False)
                    new_accepts = new_accepts.loc[sampled_indices]

                # select rejected applicants
                rejects_ind = new_applicants.index.difference(new_accepts.index)
                new_rejects = new_applicants.loc[rejects_ind]

                # transform boolean to label
                new_accepts['BAD'] = new_accepts['BAD'].map({0: 'GOOD', 1: 'BAD'})
                new_rejects['BAD'] = new_rejects['BAD'].map({0: 'GOOD', 1: 'BAD'})

                # update current accepts and rejects
                self.current_accepts = pd.concat([self.current_accepts, new_accepts], axis=0, ignore_index=True)
                self.current_rejects = pd.concat([self.current_rejects, new_rejects], axis=0, ignore_index=True)

        self.stats_list = pd.DataFrame(self.stats_list)
        return self.stats_list, holdout_accept_preds_bad, basl_preds_bad,oracle_preds_bad, self.current_accepts, self.current_rejects