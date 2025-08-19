import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric
from itertools import product
import random

class ScorecardSelector:
    '''
    A class to perform scorecard and evaluation, accounting for sample selection bias.
    '''
    def __init__(self, 
                 current_accepts, 
                 current_rejects, 
                 hold_population, 
                 cv_folds = 4,
                 acceptance_rate = 0.2,
                 seed = 1):
        self.current_accepts = current_accepts
        self.current_rejects = current_rejects
        self.hold_population = hold_population
        self.cv_folds = cv_folds
        self.acceptance_rate = acceptance_rate
        self.seed = seed

        # Results placeholders
        self.eval_results = []
        self.trained_models = []
        self.rej_scores_prior = None
        self.rejects_subset = None

        # Set a fixed random seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _prepare_data(self):
        '''
        Prepares the data for evaluation by sampling a subset of rejects which contains the rejects number matches accepts number.
        This rejects subset will be used in Bayesian Evaluation.
        '''
        num_accepts_per_fold = len(self.current_accepts) // self.cv_folds
        num_rejects_subset = int(num_accepts_per_fold * (1 - self.acceptance_rate) / self.acceptance_rate)

        rejects_indicies = np.random.choice(self.current_rejects.index, size=num_rejects_subset, replace=False)
        self.rejects_subset = self.current_rejects.loc[rejects_indicies].copy()

        # Simulate unknown labels by setting them to NA
        self.rejects_subset['BAD'] = np.nan
        print(f"Preparing {len(self.current_accepts)} accepts and {len(self.rejects_subset)} rejects for evaluation loop.")

    def _train_rejects_prior_model(self):
        '''
        Trains a simple Logistic Regression model on accepts to score rejects, creating a 'prior' probability for the Bayesian metric.
        '''
        print("Training Logistic Regression for reject priors...")
        features = [col for col in self.current_accepts.columns if col not in ['BAD']]
        X_accepts = self.current_accepts[features]
        y_accepts = self.current_accepts['BAD'].apply(lambda x: 1 if x == 'BAD' else 0)

        # Train a simple logistic regression on accepts
        logreg_model = LogisticRegression(random_state=self.seed)
        logreg_model.fit(X_accepts, y_accepts)

        # Score all rejects to get the prior probabilities
        X_rejects_subset = self.current_rejects.loc[self.rejects_subset.index, features]
        self.rej_scores_prior = pd.Series(logreg_model.predict_proba(X_rejects_subset)[:,1],
                                          index = self.rejects_subset.index)
        print("Finished scoring rejects for prior probabilities.")

    def _train_and_evaluate_models(self):
        '''
        Performs the main training and evaluation loop using K-fold cross-validation and grid search for XGBoost models.
        '''
        print("Starting XGBoost model training and evaluation...")
        
        # Get features and target
        features = [col for col in self.current_accepts.columns if col not in ['BAD']]
        X_accepts = self.current_accepts[features]
        y_accepts = self.current_accepts['BAD'].apply(lambda x: 1 if x == 'BAD' else 0)

        X_holdout = self.hold_population[features]
        y_holdout = self.hold_population['BAD'].apply(lambda x: 1 if x == 'BAD' else 0)

        # Define parameter grid
        param_grid = {'n_estimators': [10, 15, 20],
                      'max_depth': [1, 2, 3]}
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

        results_by_fold = []

        # Outer loop for cross-validation
        for fold, (train_index, valid_index) in enumerate(kf.split(X_accepts)):
            print(f"--- Processing Fold {fold + 1}/{self.cv_folds} ---")

            X_train, X_val = X_accepts.iloc[train_index], X_accepts.iloc[valid_index]
            y_train, y_val = y_accepts.iloc[train_index], y_accepts.iloc[valid_index]

            fold_metrics = []

            # Inner loop for hyperparameter combinations
            for params in param_combinations:
                model = XGBClassifier(objective='binary:logistic',
                                      eval_metric = 'auc',
                                      random_state = self.seed,
                                      **params)
                model.fit(X_train, y_train)

                # Store the trained model and its parameters
                self.trained_models.append({"model": model, 'params': params, "fold": fold+1})

                # Make predictions for evaluation
                y_pred_val = model.predict_proba(X_val)[:,1]
                y_pred_holdout = model.predict_proba(X_holdout)[:,1]

                # --- Calculate various AUC metrics --

                # 1. Accepts AUC (on the validation set)
                auc_accepts = roc_auc_score(y_val, y_pred_val)

                # 2. Holdout AUC (on the unbiased holdout set)
                auc_holdout = roc_auc_score(y_holdout, y_pred_holdout)

                # 3. Assumed Rejects are BAD
                # Combine the accepts validation set with the rejects subset
                y_val_rej_bad = np.concatenate([y_val, np.ones(len(self.rejects_subset))])
                y_val_pred_rej_bad = np.concatenate([y_pred_val, model.predict_proba(self.rejects_subset[features])[:,1]])
                auc_rej_bad = roc_auc_score(y_val_rej_bad, y_val_pred_rej_bad)

                # 4. Assumed Rejects are GOOD
                y_val_rej_good = np.concatenate([y_val, np.zeros(len(self.rejects_subset))])
                auc_rej_good = roc_auc_score(y_val_rej_good, y_val_pred_rej_bad)

                # 5. Simple Average AUC (between the two extreme assumptions)
                auc_average = (auc_rej_bad + auc_rej_good) / 2

                # 6. Weighted AUC (using the true ratio from the full collected rejects dataset from acceptance loop)
                w_bad = (self.current_rejects['BAD'] == 'BAD').mean()
                auc_weighted = auc_rej_bad * w_bad + auc_rej_good * (1 - w_bad)

                # 7. Bayesian Evaluation AUC
                metric_auc = AUC()
                bm_auc = bayesianMetric(metric_auc)
                auc_bayesian = bm_auc.BM(y_true_acc=y_val,
                                         y_proba_acc=model.predict_proba(X_val)[:,1],
                                         y_proba_rej=model.predict_proba(self.rejects_subset[features])[:,1],
                                         rejects_prior=self.rej_scores_prior,
                                         acc_rate=[0.2, 0.4])
                
                fold_metrics.append({'params': tuple(params.items()),
                                     'auc_accepts': auc_accepts,
                                     'auc_bayesian': auc_bayesian,
                                     'auc_holdout': auc_holdout,
                                     'auc_rej_bad': auc_rej_bad,
                                     'auc_rej_good': auc_rej_good,
                                     'auc_average': auc_average,
                                     'auc_weighted': auc_weighted
                                     })
                
            results_by_fold.append(pd.DataFrame(fold_metrics))

        # Concatenate results from all folds
        self.eval_results = pd.concat(results_by_fold)
        print("Finished model training and evaluation.")

    def run_selection(self):
        '''
        Run the full scorecard selection and evaluation pipeline.
        '''
        self._prepare_data()
        self._train_rejects_prior_model()
        self._train_and_evaluate_models()

        # Aggregate the results for each parameter combination
        final_results = self.eval_results.groupby(['params']).mean().reset_index()

        print("\n--- Final Evaluation Results (Averaged Across Folds) ---")
        print(final_results)

        return final_results, self.trained_models