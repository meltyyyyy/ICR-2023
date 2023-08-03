import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from utils.constants import INT_DENOMINATORS
from utils.metrics import balanced_log_loss, lgb_metric
from utils.model_selection import MultilabelStratifiedKFold

warnings.filterwarnings("ignore")


COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"


def load_data():
    train = pd.read_csv(f"{COMP_PATH}/train.csv")
    test = pd.read_csv(f"{COMP_PATH}/test.csv")
    sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
    greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")
    return train, test, sample_submission, greeks


def feature_eng(train, test):
    train.fillna(train.median(), inplace=True)
    test.fillna(test.median(), inplace=True)

    train = to_int(train.copy())
    test = to_int(test.copy())

    train["EJ"] = train["EJ"].map({"A": 0, "B": 1})
    test["EJ"] = test["EJ"].map({"A": 0, "B": 1})

    scaler = StandardScaler()
    df, test_df = train.copy(), test.copy()
    new_num_cols = train.select_dtypes(include=["float64"]).columns
    df[new_num_cols] = scaler.fit_transform(train[new_num_cols])
    test_df[new_num_cols] = scaler.transform(test[new_num_cols])

    return df, test_df


def to_int(df):
    for k, v in INT_DENOMINATORS.items():
        df[k] = np.round(df[k] / v, 1)
    return df


def training(df, feat_cols, target, greeks):
    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1

    for fold, (_, test_idx) in enumerate(kf.split(X=df, y=greeks.iloc[:, 1:3])):
        df.loc[test_idx, "fold"] = fold

    metric = balanced_log_loss
    outer_cv_score = []  # store all cv scores of outer loop inference
    inner_cv_score = []  # store all cv scores of inner loop training
    feat_imps = []
    models = []
    weights = []

    for fold in range(5):
        train_df = df[df["fold"] != fold]
        valid_df = df[df["fold"] == fold]

        X_train, y_train = (
            train_df.drop(["Id", "Class", "fold"], axis=1).loc[:, feat_cols],
            train_df[target],
        )
        X_valid, y_valid = (
            valid_df.drop(["Id", "Class", "fold"], axis=1).loc[:, feat_cols],
            valid_df[target],
        )

        lgb = LGBMClassifier(
            boosting_type="goss",
            learning_rate=0.06733232950390658,
            n_estimators=50000,
            early_stopping_round=300,
            random_state=42,
            subsample=0.6970532011679706,
            colsample_bytree=0.6055755840633003,
            class_weight="balanced",
            metric="none",
            is_unbalance=True,
            max_depth=8,
        )

        holdout = pd.concat([X_valid, y_valid], axis=1)
        oof_inner = np.zeros(len(X_train))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_models = []
        print(
            f"Outer Loop fold {fold}, Inner Loop Training with {X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        for fold, (fit_idx, val_idx) in enumerate(cv.split(X=X_train, y=y_train)):
            X_fit = X_train.iloc[fit_idx]
            X_val = X_train.iloc[val_idx]
            y_fit = y_train.iloc[fit_idx]
            y_val = y_train.iloc[val_idx]

            model = lgb.fit(
                X_fit,
                y_fit,
                eval_set=(X_valid, y_valid),
                verbose=0,
                eval_metric=lgb_metric,
            )
            inner_models.append(model)
            val_preds = model.predict(X_val)
            oof_inner[val_idx] = val_preds

            val_score = balanced_log_loss(y_val, val_preds)
            best_iter = model.booster_.best_iteration
            feat_imps.append(model.feature_importances_)
            print(
                f"Fold: {fold:>3}| {metric.__name__}: {val_score:.5f}"
                f" | Best iteration: {best_iter:>4}"
            )

        mean_cv_score = metric(y_train, oof_inner)
        print(f"80% data CV score: {metric.__name__}: {mean_cv_score:.5f}")
        print(f'{"*" * 50}\n')
        inner_cv_score.append(mean_cv_score)

        preds = np.zeros(len(holdout))
        for model in inner_models:
            preds += model.predict(X_valid)
        preds = preds / len(inner_models)
        cv_score = metric(y_valid, preds)

        print(f"20% data CV score: {metric.__name__}: {cv_score:.5f}")
        print(f'{"*" * 50}\n')

        outer_cv_score.append(cv_score)
        models.append(inner_models)
        weights.append(1 / cv_score)
    print(
        f"80% data average CV score: {metric.__name__}: {np.mean(inner_cv_score):.5f}"
    )
    print(f'{"*" * 50}\n')

    print(
        f"20% data average CV score: {metric.__name__}: {np.mean(outer_cv_score):.5f}"
    )
    print(f'{"*" * 50}\n')

    return (
        models,
        weights,
        np.mean(feat_imps, axis=0),
        np.mean(inner_cv_score),
        np.mean(outer_cv_score),
    )


def inferring(X, models, weights):
    y = np.zeros(len(X))
    for i, model_list in enumerate(models):
        for model in model_list:
            y += weights[i] * model.predict(X)
    return y / sum([len(inner_models) for inner_models in models])


def plot_importance(feat_imps, feat_cols):
    feat_imps = np.squeeze(feat_imps)  # flatten feat_imps to 1D array
    feat_df = pd.DataFrame(feat_imps, index=feat_cols, columns=["importance"])
    feat_df = feat_df.sort_values(
        by="importance", ascending=True
    )  # sort DataFrame by importance

    plt.figure(figsize=(12, 6))
    plt.title("Mean feature importance")
    sns.barplot(
        x=feat_df["importance"], y=feat_df.index
    )  # use sorted DataFrame for plotting
    plt.savefig("feat_imps.png")
    plt.close()


def main():
    train, test, _, greeks = load_data()
    df, test_df = feature_eng(train, test)
    feat_cols = df.columns[1:-1]
    target = "Class"

    _, _, feat_imps, inner_cv_score, outer_cv_score = training(
        df, feat_cols, target, greeks
    )
    plot_importance(feat_imps, feat_cols)
    icv_scores = []
    ocv_scores = []
    feat_col_list = []
    best_feat_cols = feat_cols

    while len(feat_cols) > 10:
        _, _, feat_imps, inner_cv_score, outer_cv_score = training(
            df, feat_cols, target, greeks
        )
        if len(ocv_scores) > 1 and outer_cv_score < np.min(ocv_scores):
            best_feat_cols = feat_cols

        icv_scores.append(inner_cv_score)
        ocv_scores.append(outer_cv_score)
        feat_col_list.append(feat_cols)

        feat_df = pd.DataFrame(feat_imps, index=feat_cols, columns=["importance"])
        feat_df = feat_df.sort_values(by="importance", ascending=True)
        feat_cols = feat_df.iloc[1:].index

    _, _, feat_imps, inner_cv_score, outer_cv_score = training(
        df, best_feat_cols, target, greeks
    )
    plot_importance(feat_imps, best_feat_cols)

    plt.figure(figsize=(12, 6))
    plt.plot(icv_scores, label="Inner CV score")
    plt.plot(ocv_scores, label="Outer CV score")
    plt.xlabel("Number of features dropped")
    plt.ylabel("Metric score")
    plt.legend()
    plt.savefig("feat_reduction.png")
    plt.close()

    lowest_score_indices = np.argsort(ocv_scores)[:5]
    for rank, i in enumerate(lowest_score_indices):
        with open(f"top{rank}_feat_cols.txt", "w") as f:
            f.write(str(feat_col_list[i]))
            f.write(f"\nInner CV score: {icv_scores[i]:.5f}")
            f.write(f"\nOuter CV score: {ocv_scores[i]:.5f}")


if __name__ == "__main__":
    main()
