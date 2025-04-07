from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import numpy as np
import joblib
import os


class MLWorkflow:
    def __init__(self, k_folds=5):
        self.k_folds = k_folds
        self.models = {}
        self.models["elastic_net"] = ElasticNet()
        self.models["svr"] = SVR()
        self.models["bayesian_ridge"] = BayesianRidge()
        self.best_model = None
        self.best_features = None
        self.best_params = None
        self.best_model_name = None

    def establish_baseline(self, X_train, y_train):
        resDict = {}  # storing baseline results
        k_fold_obj = KFold(n_splits=(self.k_folds), shuffle=True, random_state=42)
        for mName in self.models:
            mdl = self.models[mName]
            scoresArr = cross_val_score(
                mdl,
                X_train,
                y_train,
                cv=k_fold_obj,
                scoring="neg_root_mean_squared_error",
            )
            tot = 0
            cnt = 0
            for s in scoresArr:
                tot = tot + s
                cnt = cnt + 1
            if cnt != 0:
                meanScore = tot / cnt
            else:
                meanScore = 0
            resDict[mName] = -meanScore  # converting negative to positive RMSE
            print("Baseline for", mName, "is", (-meanScore))
        return resDict

    def select_features(self, X_train, y_train, max_features=None):
        if max_features == None:
            max_features = X_train.shape[1]
        bestScore = 999999999  
        best_n = 0
        bestMask = None
        for num in range(1, (max_features + 1)):
            sel = SelectKBest(score_func=f_regression, k=num)
            X_sel = sel.fit_transform(X_train, y_train)
            for key in self.models:
                mdl = self.models[key]
                scArr = cross_val_score(
                    mdl,
                    X_sel,
                    y_train,
                    cv=self.k_folds,
                    scoring="neg_root_mean_squared_error",
                )
                s = 0
                c = 0
                for val in scArr:
                    s = s + val
                    c = c + 1
                if c > 0:
                    mean_sc = s / c
                else:
                    mean_sc = 0
                curScore = -mean_sc
                print(
                    "Trying", num, "features for", key, "gives score", curScore
                )
                if curScore < bestScore:
                    bestScore = curScore
                    best_n = num
                    bestMask = sel.get_support()
        return bestMask, best_n, bestScore

    def tune_model(self, X_train, y_train, sel_feat):
        # Tuning models with GridSearchCV
        Xsel = X_train[:, sel_feat]
        gridParams = {}
        gridParams["elastic_net"] = {}
        gridParams["elastic_net"]["alpha"] = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        gridParams["elastic_net"]["l1_ratio"] = [0.1, 0.3, 0.5, 0.7, 0.9]
        gridParams["svr"] = {}
        gridParams["svr"]["C"] = [0.1, 1, 10, 100]
        gridParams["svr"]["epsilon"] = [0.01, 0.1, 0.2]
        gridParams["svr"]["kernel"] = ["linear", "rbf"]
        gridParams["bayesian_ridge"] = {}
        gridParams["bayesian_ridge"]["alpha_1"] = [1e-7, 1e-6, 1e-5]
        gridParams["bayesian_ridge"]["alpha_2"] = [1e-7, 1e-6, 1e-5]
        gridParams["bayesian_ridge"]["lambda_1"] = [1e-7, 1e-6, 1e-5]
        gridParams["bayesian_ridge"]["lambda_2"] = [1e-7, 1e-6, 1e-5]

        bestScr = 999999999  
        bestModName = ""
        bestPrms = None
        for key in self.models:
            print("Tuning model:", key)
            mdl = self.models[key]
            gridSrch = GridSearchCV(
                mdl,
                gridParams[key],
                cv=self.k_folds,
                scoring="neg_root_mean_squared_error",
                n_jobs=(-1),
            )
            gridSrch.fit(Xsel, y_train)
            curBestScr = -gridSrch.best_score_
            print(" Current best score for", key, "is", curBestScr)
            if curBestScr < bestScr:
                bestScr = curBestScr
                bestModName = key
                bestPrms = gridSrch.best_params_
                self.best_model = gridSrch.best_estimator_
                self.best_model_name = key
                print("New best model is", key)
        return bestModName, bestPrms, bestScr

    def fit(self, X_train, y_train, models_dir="../models"):
        baseRes = self.establish_baseline(X_train, y_train)
        print("Baseline results computed:")
        print(baseRes)

        selFeats, numFeats, score_fs = self.select_features(X_train, y_train)
        self.best_features = selFeats
        print(
            "Feature selection done. Number of features: "
            + str(numFeats)
            + " with score: "
            + str(score_fs)
        )

        bestMod, bestPrm, bestScr = self.tune_model(X_train, y_train, selFeats)
        self.best_params = bestPrm
        print(" Model tuning finished.")
        print("Best Model: " + str(self.best_model_name))
        print("Best Parameters: " + str(bestPrm))
        print("Best Score (RMSE): " + str(bestScr))

        if self.best_model != None:
            try:
                os.makedirs(models_dir, exist_ok=True)
                fileName = os.path.join(
                    models_dir, (str(self.best_model_name) + "_final_model.joblib")
                )
                joblib.dump(self.best_model, fileName)
                print(" Final model saved at " + fileName)
            except Exception as ex:
                print(" Error saving model: " + str(ex))
        else:
            print("No best model found to save!")
        return self

    def predict(self, X_tst):
        # Make predictions with our best model
        if (self.best_model == None) or (self.best_features == None):
            raise ValueError("Model must be fitted before making predictions")
        X_sel = X_tst[:, self.best_features]
        pred = self.best_model.predict(X_sel)
        return pred

    def evaluate(self, X_eval, y_eval, n_bootstraps=1000, confidence_level=0.95):
        # Evaluate the model 
        if (self.best_model == None) or (self.best_features == None):
            raise ValueError("Model must be fitted before evaluation.")
        Xsel_eval = X_eval[:, self.best_features]
        totSamples = len(y_eval)
        met = {"rmse": [], "mae": [], "r2": []}
        print(
            "Starting evaluation with "
            + str(n_bootstraps)
            + " bootstrap samples..."
        )
        for j in range(n_bootstraps):
            allIdx = np.arange(totSamples)
            idxs = resample(allIdx, n_samples=totSamples, replace=True)
            X_bootstrap = Xsel_eval[idxs]
            y_bootstrap = y_eval[idxs]
            uniqVals = np.unique(y_bootstrap)
            if len(uniqVals) < 2:
                print(
                    "Skipping bootstrap sample "
                    + str(j + 1)
                    + " due to uniform y values."
                )
                continue
            try:
                y_pred = self.best_model.predict(X_bootstrap)
                rmse_val = np.sqrt(mean_squared_error(y_bootstrap, y_pred))
                mae_val = mean_absolute_error(y_bootstrap, y_pred)
                r2_val = r2_score(y_bootstrap, y_pred)
                met["rmse"].append(rmse_val)
                met["mae"].append(mae_val)
                met["r2"].append(r2_val)
            except Exception as ex:
                continue

        if len(met["rmse"]) == 0:
            return None

        resDict = {}
        a_val = (1.0 - confidence_level) / 2.0
        for mKey in met:
            vals = met[mKey]
            if len(vals) == 0:
                resDict[mKey] = {
                    "mean": np.nan,
                    "median": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
                continue
            s = 0
            cnt = 0
            for v in vals:
                s = s + v
                cnt = cnt + 1
            mean_val = s / cnt
            median_val = np.median(vals)
            ci_low = np.percentile(vals, a_val * 100)
            ci_high = np.percentile(vals, (1.0 - a_val) * 100)
            resDict[mKey] = {
                "mean": mean_val,
                "median": median_val,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }
            print(
                mKey.upper()
                + " -> Mean: "
                + str(round(mean_val, 4))
                + ", Median: "
                + str(round(median_val, 4))
                + ", "
                + str(int(confidence_level * 100))
                + "% CI: ("
                + str(round(ci_low, 4))
                + ", "
                + str(round(ci_high, 4))
                + ")"
            )
        return resDict
