import pandas as pd
import numpy as np
import sys
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import clone

from numpy.random import choice

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.functions import MLWorkflow

# Data Loading and Preparation
dataFolder = "../data/"
devPath = os.path.join(dataFolder, "development_final_data.csv")
evalPath = os.path.join(dataFolder, "evaluation_final_data.csv")

print("Loading development (training) data now...")
devData = pd.read_csv(devPath)
print("Loading evaluation data now...")
evalData = pd.read_csv(evalPath)

print("Preparing data for modeling...")
# Using the last column as target
targetCol = devData.columns[-1]
print("Target variable is: " + str(targetCol))

# Get features and target for development data
X_dev = devData.drop(columns=[targetCol]).values
y_dev = devData[targetCol].values

# Get features and target for evaluation data
X_eval = evalData.drop(columns=[targetCol]).values
y_eval = evalData[targetCol].values

#test
print("Dev set shape: " + str(X_dev.shape))
print("Eval set shape: " + str(X_eval.shape))

# Baseline Models (No FS, No Tuning)
print("\n--- Generating Baseline Models ---")
baseModels = {}  # dictionary to store baseline models
# Create a workflow instance to use its models later
myWorkflow = MLWorkflow()

# Create fresh instances for baseline models
algoDict = {}
algoDict["elastic_net"] = myWorkflow.models["elastic_net"].__class__()
algoDict["svr"] = myWorkflow.models["svr"].__class__()
algoDict["bayesian_ridge"] = myWorkflow.models["bayesian_ridge"].__class__()

#test
#print("AlgoDict: " + str(algoDict))

modelFolder = "../models"
os.makedirs(modelFolder, exist_ok=True)

# Train each baseline model
for nm in algoDict:
    print("\nTraining baseline model for " + nm + " ...")
    mdl = algoDict[nm]
    mdl.fit(X_dev, y_dev)
    modPath = os.path.join(modelFolder, nm + "_baseline.joblib")
    joblib.dump(mdl, modPath)
    print("Saved baseline model for " + nm + " to " + modPath)
    baseModels[nm] = mdl

# Evaluate baseline models
print("\n--- Evaluating Baseline Models ---")
baseResList = []  # list to store results
for nm in baseModels:
    print("Evaluating " + nm + " model...")
    modl = baseModels[nm]
    yPred = modl.predict(X_eval)
    rmse_val = np.sqrt(mean_squared_error(y_eval, yPred))
    mae_val = mean_absolute_error(y_eval, yPred)
    r2_val = r2_score(y_eval, yPred)
    print(
        nm
        + ": RMSE="
        + str(round(rmse_val, 4))
        + ", MAE="
        + str(round(mae_val, 4))
        + ", R²="
        + str(round(r2_val, 4))
    )
    # Add result dict 
    baseResList.append(
        {
            "Model": nm,
            "RMSE": rmse_val,
            "MAE": mae_val,
            "R²": r2_val,
            "Approach": "Baseline",
        }
    )

baseDF = pd.DataFrame(baseResList)
print("\n--- Baseline Model Comparison ---")
print(baseDF.to_string(index=False))

# Feature Selection Stage
print("\n--- Feature Selection on Development Set ---")
selFeatMask, numFeat, fs_rmse = myWorkflow.select_features(X_dev, y_dev)
print(
    "Selected "
    + str(numFeat)
    + " features with CV RMSE: "
    + str(round(fs_rmse, 4))
)
#test
temp_mask = selFeatMask[:20]
print("Feature mask (first 20): " + str(temp_mask))

# Train models on selected features (still no tuning here)
print("\n--- Training Models on Selected Features ---")
fsModels = {}
for nm in algoDict:
    print("Cloning model for " + nm + " and training on selected features...")
    freshModel = clone(algoDict[nm])  # clone to get a new model
    X_dev_fs = X_dev[:, selFeatMask]
    freshModel.fit(X_dev_fs, y_dev)
    fsModPath = os.path.join(modelFolder, nm + "_fs.joblib")
    joblib.dump(freshModel, fsModPath)
    print("Saved feature-selected model for " + nm + " to " + fsModPath)
    fsModels[nm] = freshModel

print("\n--- Evaluating Feature-Selected Models ---")
fsResList = []
for nm in fsModels:
    print("Evaluating feature-selected model for " + nm + " ...")
    mdl_fs = fsModels[nm]
    X_eval_fs = X_eval[:, selFeatMask]
    yPred_fs = mdl_fs.predict(X_eval_fs)
    rmse_fs = np.sqrt(mean_squared_error(y_eval, yPred_fs))
    mae_fs = mean_absolute_error(y_eval, yPred_fs)
    r2_fs = r2_score(y_eval, yPred_fs)
    print(
        nm
        + ": RMSE="
        + str(round(rmse_fs, 4))
        + ", MAE="
        + str(round(mae_fs, 4))
        + ", R²="
        + str(round(r2_fs, 4))
    )
    fsResList.append(
        {
            "Model": nm,
            "RMSE": rmse_fs,
            "MAE": mae_fs,
            "R²": r2_fs,
            "Approach": "Feature Selected",
        }
    )

fsDF = pd.DataFrame(fsResList)
print("\n--- Feature-Selected Model Comparison ---")
print(fsDF.to_string(index=False))

# Combine Baseline and Feature-Selected results for comparison
compDF = pd.concat([baseDF, fsDF])
print("\n--- Comparison of Baseline vs. Feature-Selected Models ---")
print(compDF.to_string(index=False))

# FS + Tuning Stage (Hyperparameter Tuning)
# ===============================
print("\n--- Hyperparameter Tuning on Feature-Selected Models ---")
# Define parameter grids 
paramGrids = {}
paramGrids["elastic_net"] = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}
paramGrids["svr"] = {
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.2],
    "kernel": ["linear", "rbf"],
}
paramGrids["bayesian_ridge"] = {
    "alpha_1": [1e-7, 1e-6, 1e-5],
    "alpha_2": [1e-7, 1e-6, 1e-5],
    "lambda_1": [1e-7, 1e-6, 1e-5],
    "lambda_2": [1e-7, 1e-6, 1e-5],
}

tunedModels = {}
tuningResList = []
# Use selected features from dev set
X_dev_fs = X_dev[:, selFeatMask]
# Create a KFold object
kfoldObj = KFold(n_splits=myWorkflow.k_folds, shuffle=True, random_state=42)

for nm in algoDict:
    print("\nTuning model " + nm + " on feature-selected data...")
    # Use clone to get a fresh model
    tempModel = clone(algoDict[nm])
    gridSearch = GridSearchCV(
        tempModel,
        paramGrids[nm],
        cv=kfoldObj,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gridSearch.fit(X_dev_fs, y_dev)
    bestMod = gridSearch.best_estimator_
    bestScr = -gridSearch.best_score_
    bestPrms = gridSearch.best_params_
    print(
        "Best "
        + nm
        + " model: RMSE="
        + str(round(bestScr, 4))
        + ", Params="
        + str(bestPrms)
    )
    tunedModels[nm] = bestMod
    tuningResList.append(
        {"Model": nm, "RMSE": bestScr, "Best Params": bestPrms, "Approach": "FS+Tuning"}
    )

tuningDF = pd.DataFrame(tuningResList)
print("\n--- Tuning Results ---")
print(tuningDF.to_string(index=False))

# Save tuned models to a final_models folder
finalModelsFolder = "../final_models"
os.makedirs(finalModelsFolder, exist_ok=True)
for nm in tunedModels:
    finalModPath = os.path.join(finalModelsFolder, nm + "_final_model.joblib")
    joblib.dump(tunedModels[nm], finalModPath)
    print("Saved final tuned model for " + nm + " to " + finalModPath)

# Evaluate Tuned Models on Evaluation Set
print("\n--- Evaluating Tuned Models on Evaluation Set ---")
tunedResList = []
for nm in tunedModels:
    print("Evaluating tuned model for " + nm + " ...")
    mod_tuned = tunedModels[nm]
    X_eval_fs = X_eval[:, selFeatMask]
    yPred_tuned = mod_tuned.predict(X_eval_fs)
    rmse_tuned = np.sqrt(mean_squared_error(y_eval, yPred_tuned))
    mae_tuned = mean_absolute_error(y_eval, yPred_tuned)
    r2_tuned = r2_score(y_eval, yPred_tuned)
    print(
        nm
        + ": RMSE="
        + str(round(rmse_tuned, 4))
        + ", MAE="
        + str(round(mae_tuned, 4))
        + ", R²="
        + str(round(r2_tuned, 4))
    )
    tunedResList.append(
        {
            "Model": nm,
            "RMSE": rmse_tuned,
            "MAE": mae_tuned,
            "R²": r2_tuned,
            "Approach": "FS+Tuning",
        }
    )

tunedDF = pd.DataFrame(tunedResList)
print("\n--- Tuned Model Comparison ---")
print(tunedDF.to_string(index=False))

# Combine all results for the final  comparison
finalCompDF = pd.concat([baseDF, fsDF, tunedDF])
print("\n--- Final Comparison of All Approaches ---")
print(finalCompDF.to_string(index=False))


# Bootstrap Evaluation & Boxplots (for all stages)
def bootstrap_evaluation(mod, X_data, y_data, n_bootstraps=100, feature_mask=None):
    rmse_list = []  # list for rmse values
    mae_list = []  # list for mae values
    r2_list = []  # list for r2 values
    tot_samples = len(y_data)
    for i in range(n_bootstraps):
        # Get random indices with replacement (super random)
        rand_idx = np.random.choice(tot_samples, tot_samples, replace=True)
        X_sample = X_data[rand_idx]
        if feature_mask is not None:
            X_sample = X_sample[:, feature_mask]
        y_sample = y_data[rand_idx]
        y_pred = mod.predict(X_sample)
        # Calculate metrics 
        cur_rmse = np.sqrt(mean_squared_error(y_sample, y_pred))
        cur_mae = mean_absolute_error(y_sample, y_pred)
        cur_r2 = r2_score(y_sample, y_pred)
        rmse_list.append(cur_rmse)
        mae_list.append(cur_mae)
        r2_list.append(cur_r2)
        print("Bootstrap iteration", (i + 1), "completed with RMSE =", cur_rmse)
    return rmse_list, mae_list, r2_list


# empty list to store bootstrap results
bootstrap_results = []
n_boot = 100  # Number of bootstrap iterations

# Evaluate Baseline models (using full features)
for nm in baseModels:
    mdl = baseModels[nm]
    rmse_vals, mae_vals, r2_vals = bootstrap_evaluation(
        mdl, X_eval, y_eval, n_bootstraps=n_boot
    )
    bootstrap_results.append(
        {
            "Model": nm,
            "Stage": "Baseline",
            "RMSE": rmse_vals,
            "MAE": mae_vals,
            "R2": r2_vals,
        }
    )
    print("Completed bootstrap for baseline model", nm)

# Evaluate Feature-selected models
for nm in fsModels:
    mdl = fsModels[nm]
    rmse_vals, mae_vals, r2_vals = bootstrap_evaluation(
        mdl, X_eval, y_eval, n_bootstraps=n_boot, feature_mask=selFeatMask
    )
    bootstrap_results.append(
        {
            "Model": nm,
            "Stage": "Feature Selected",
            "RMSE": rmse_vals,
            "MAE": mae_vals,
            "R2": r2_vals,
        }
    )
    print("Completed bootstrap for feature-selected model", nm)

# Evaluate Tuned models (FS+Tuning)
for nm in tunedModels:
    mdl = tunedModels[nm]
    rmse_vals, mae_vals, r2_vals = bootstrap_evaluation(
        mdl, X_eval, y_eval, n_bootstraps=n_boot, feature_mask=selFeatMask
    )
    bootstrap_results.append(
        {
            "Model": nm,
            "Stage": "FS+Tuning",
            "RMSE": rmse_vals,
            "MAE": mae_vals,
            "R2": r2_vals,
        }
    )
    print("Completed bootstrap for tuned model", nm)

# Organize bootstrap results into a DataFrame for plotting
bootstrap_data = []
for res in bootstrap_results:
    for j in range(n_boot):
        bootstrap_data.append(
            {
                "Model": res["Model"],
                "Stage": res["Stage"],
                "RMSE": res["RMSE"][j],
                "MAE": res["MAE"][j],
                "R2": res["R2"][j],
            }
        )
bootstrap_df = pd.DataFrame(bootstrap_data)
bootstrap_df["Group"] = bootstrap_df["Model"] + " (" + bootstrap_df["Stage"] + ")"
print("Bootstrap data organized into DataFrame with shape", bootstrap_df.shape)

# Create boxplots for each metric
metric_list = ["RMSE", "MAE", "R2"]
for met in metric_list:
    plt.figure(figsize=(10, 6))
    groups = bootstrap_df["Group"].unique()
    data_to_plot = []
    for grp in groups:
        # Get all values for the current group and metric
        grp_data = bootstrap_df[bootstrap_df["Group"] == grp][met]
        data_to_plot.append(grp_data)
    plt.boxplot(data_to_plot, labels=groups)
    plt.title("Bootstrap " + met + " Comparison")
    plt.ylabel(met)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("Boxplot for", met, "displayed.")

final_models_dir = "../final_models"

# Save each best tuned model from our tuned_models into final_models directory
for nm in tunedModels:
    mod_obj = tunedModels[nm]
    final_model_path = os.path.join(final_models_dir, nm + "_final_model.joblib")
    joblib.dump(mod_obj, final_model_path)
    print("Saved final tuned model", nm, "to", final_model_path)


# Final Comparison and Best Model Selection
print("\n--- Final Tuned Model Comparison ---")
print(tunedDF.to_string(index=False))

# Pick the best overall model based on the lowest RMSE
best_model_row = tunedDF.loc[tunedDF["RMSE"].idxmin()]
print("\n--- Best Overall Model Based on RMSE ---")
print(best_model_row)

print("\nFinal Decision:")
print(
    "The best overall model is "
    + str(best_model_row["Model"])
    + " with an RMSE of "
    + str(round(best_model_row["RMSE"], 4))
    + "."
)
print(
    "This model was chosen because it has the lowest RMSE and a good balance of complexity and interpretability."
)

# Retrain Best Overall Model on the Full Dataset (Development + Evaluation)
# ------------
print("\n--- Retraining Best Overall Model on the Full Dataset ---")
# Combine development and evaluation datasets 
X_full = np.vstack((X_dev, X_eval))
y_full = np.concatenate((y_dev, y_eval))
print("Full dataset shape is", X_full.shape)

# Use the best tuned model and clone it for retraining
from sklearn.base import clone

winner_model = clone(tunedModels["bayesian_ridge"])
winner_model.fit(X_full[:, selFeatMask], y_full)
print("Winner model retrained on the full dataset.")

# Save the retrained winner model as "winner.joblib"
winner_model_path = os.path.join(final_models_dir, "winner.joblib")
joblib.dump(winner_model, winner_model_path)
print("Final winner model saved to", winner_model_path)
