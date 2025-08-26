import pandas as pd
import numpy as np
import data_generator as data
from reject_inference import RejectInference, BiasAwareSelfLearning, HeckmanTwoStage, LabelAllRejectsAsBad, Reweighting
from Evaluation import Metric, AUC, BS, PAUC, ABR, Evaluation, bayesianMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance


######### INITIAL POPULATION ########
# generate data
res = data.DataGenerator(
    n = 20000,
    bad_ratio = 0.7,
    k_con = 100,
    k_bin = 0,
    con_nonlinear = 0,
    con_var_bad_dif = 0.5,
    seed = 77,
    verbose = True
)
res.generate()

# extract initial population
init_population = res.data
init_population.drop(columns=['B1'], inplace=True)
# FIX: Assign the mapped values back to the DataFrame
init_population['BAD'] = init_population['BAD'].map({'BAD':1, 'GOOD':0})

## Classification of initial population into accepts and rejects
top_percent = 0.2

# accept applicants using a business rule on X1
accepts_ind = init_population.index[init_population['X1'] >= init_population['X1'].quantile(1 - top_percent)]
final_accepts = init_population.loc[accepts_ind].copy()
final_rejects = init_population.drop(accepts_ind).copy()

# FIX: Combine the accepts and rejects data to fit the scaler once
all_training_x = pd.concat([final_accepts.drop(columns=["BAD"]), 
                           final_rejects.drop(columns=["BAD"])], 
                           ignore_index=True)

# Scaling the input features
scaler = StandardScaler()
# FIX: Fit the scaler on the entire training population
scaler.fit(all_training_x)

# FIX: Use the fitted scaler to transform both accepts and rejects data
accepts_x_np = scaler.transform(final_accepts.drop(columns=["BAD"]))
rejects_x_np = scaler.transform(final_rejects.drop(columns=["BAD"]))

# Convert the NumPy arrays back to DataFrames, using the original column names
accepts_x = pd.DataFrame(accepts_x_np, columns=all_training_x.columns)
rejects_x = pd.DataFrame(rejects_x_np, columns=all_training_x.columns)

######### HOLDOUT POPULATION ########
holdout_sample = 3000
holdout = data.DataGenerator(
    n = holdout_sample,
    replicate = res,
    seed = 99999,
)
holdout.generate()
holdout_population = holdout.data
holdout_population.drop(columns=['B1'], inplace=True)
x_holdout = holdout_population.drop(columns=['BAD'])
y_holdout = holdout_population['BAD'].map({"BAD":1, "GOOD":0})
x_holdout_np = scaler.transform(x_holdout)
x_holdout = pd.DataFrame(x_holdout_np, columns=x_holdout.columns)


# ============================================================
# FEATURE SELECTION FOR TWO-STAGE HECKMAN MODELS
# ============================================================

print("\nPerforming two-step feature selection for the Heckman Two-Stage model...")

# Step 1: Feature selection for the selection equation (accepted vs. rejected)
full_data_for_fs = pd.concat([accepts_x, rejects_x], ignore_index=True)
selection_y_for_fs = pd.Series([1] * len(accepts_x) + [0] * len(rejects_x))

xgb_selection_fs = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb_selection_fs.fit(full_data_for_fs, selection_y_for_fs)

selection_importance_result = permutation_importance(
    xgb_selection_fs, full_data_for_fs, selection_y_for_fs, n_repeats=10, random_state=42)
selection_fs_indices = selection_importance_result.importances_mean.argsort()[::-1]

# Step 2: Feature selection for the outcome equation (bad vs. good in accepts)
xgb_outcome_fs = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb_outcome_fs.fit(accepts_x, final_accepts['BAD'])

outcome_importance_result = permutation_importance(
    xgb_outcome_fs, accepts_x, final_accepts['BAD'], n_repeats=10, random_state=42)
outcome_fs_indices = outcome_importance_result.importances_mean.argsort()[::-1]

# Define the features for the Two-Stage model (uses the top features from the selection model)
k_features_twostage_selection = 10
k_features_twostage_outcome = 50 
heckman_selection_features = full_data_for_fs.columns[selection_fs_indices[:k_features_twostage_selection]].tolist()
heckman_outcome_features = accepts_x.columns[outcome_fs_indices[:k_features_twostage_outcome]].tolist()

# Get the list of all feature columns for converting names to indices
all_cols = all_training_x.columns.tolist()
# Convert the selection feature names to their numeric indices
heckman_selection_features_idx = [all_cols.index(f) for f in heckman_selection_features]
# Convert the outcome feature names to their numeric indices
heckman_outcome_features_idx = [all_cols.index(f) for f in heckman_outcome_features]

# ============================================================
# Benchmark Models: LABAR + Reweighting + Heckman Model + BASL
# ============================================================

# Instantiate and fit LABAR
labar_model = LabelAllRejectsAsBad(
    strong_estimator=LogisticRegression(max_iter=1000, random_state=42),
    silent=False
)
labar_model.fit(
    accepts_x=accepts_x,
    accepts_y=final_accepts["BAD"],
    rejects_x=rejects_x)


# Instantiate and fit Reweighting
reweighting_model = Reweighting(
    strong_estimator=LogisticRegression(max_iter=1000, random_state=42),
    silent=False
)
reweighting_model.fit(
    accepts_x=accepts_x,
    accepts_y=final_accepts["BAD"],
    rejects_x=rejects_x)

# Instantiate and fit Bias-Aware Self-Learning
basl_model = BiasAwareSelfLearning(strong_estimator = LogisticRegression(penalty='l1',solver='liblinear',max_iter=10000, random_state=42),
                            weak_learner_estimator = LogisticRegression(penalty='l1',solver='liblinear',max_iter=10000, random_state=42),
                            filtering_beta = [0.1, 0.9],
                            holdout_percent = 0.1,
                            labeling_percent = 0.2,
                            sampling_percent = 0.5,
                            multiplier = 4,
                            max_iterations = 5,
                            early_stop = True, 
                            silent=True)
basl_model.fit(accepts_x= accepts_x,
            accepts_y= final_accepts["BAD"],
            rejects_x= rejects_x)

# Instantiate and fit Heckman Two-Stage Model
heckman_two_stage = HeckmanTwoStage(
    selection_classifier='XGB', 
    outcome_classifier='XGB',
    selection_features_idx=heckman_selection_features_idx,
    outcome_features_idx=heckman_outcome_features_idx)

heckman_two_stage.fit(accepts_x=accepts_x,
                      accepts_y=final_accepts['BAD'],
                      rejects_x=rejects_x)

fitted_models = {
    "Heckman-TwoStage": heckman_two_stage,       
    "LARAB": labar_model,
    "Reweighting": reweighting_model,
    "BASL": basl_model
}

# Define metrics from Evaluation.py
metrics = {
    "AUC": AUC(),
    "BS": BS(),
    "PAUC": PAUC(),
    "ABR": ABR()
}

# Evaluation on holdout set
results = {}
for model_name, model in fitted_models.items():
    print(f"\nEvaluating: {model_name}")
    try:
        # Align features
        feature_cols = all_training_x.columns
        x_holdout_aligned = x_holdout[feature_cols]

        y_pred_holdout = model.predict(x_holdout_aligned)
        y_proba_holdout = model.predict_proba(x_holdout_aligned)
        y_proba_positive_class = y_proba_holdout[:, 1]

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
