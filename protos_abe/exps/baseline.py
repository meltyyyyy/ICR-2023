import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from utils.metrics import balanced_log_loss, lgb_metric

warnings.filterwarnings("ignore")

COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"
COLS = [
    "DE",
    "EL",
    "GH",
    "FE",
    "DY",
    "EE",
    "EU",
    "CH",
    "CD ",
    "CC",
    "GL",
    "DL",
    "EB",
    "AF",
    "FI",
    "DN",
    "DA",
    "FL",
    "CR",
    "FR",
    "AB",
    "BQ",
    "DU",
]


def load_data():
    train = pd.read_csv(f"{COMP_PATH}/train.csv")
    test = pd.read_csv(f"{COMP_PATH}/test.csv")
    sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
    greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")

    return train, test, sample_submission, greeks


def feature_eng(train, test):
    train.fillna(train.median(), inplace=True)
    test.fillna(test.median(), inplace=True)

    train["EJ"] = train["EJ"].map({"A": 0, "B": 1})
    test["EJ"] = test["EJ"].map({"A": 0, "B": 1})

    scaler = StandardScaler()
    df, test_df = train.copy(), test.copy()
    new_num_cols = train.select_dtypes(include=["float64"]).columns
    df[new_num_cols] = scaler.fit_transform(train[new_num_cols])
    test_df[new_num_cols] = scaler.transform(test[new_num_cols])

    return df, test_df


def training(df, test_df, greeks):
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    df["fold"] = -1

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, greeks["Alpha"])):
        df.loc[test_idx, "fold"] = fold

    final_valid_predictions = {}
    final_test_predictions = []
    scores = []
    log_losses = []
    balanced_log_losses = []
    weights = []

    for fold in range(5):
        train_df = df[df["fold"] != fold]
        valid_df = df[df["fold"] == fold]
        valid_ids = valid_df.Id.values.tolist()

        X_train, y_train = (
            train_df.drop(["Id", "Class", "fold"], axis=1).loc[:, COLS],
            train_df["Class"],
        )
        X_valid, y_valid = (
            valid_df.drop(["Id", "Class", "fold"], axis=1).loc[:, COLS],
            valid_df["Class"],
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

        lgb.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            verbose=1000,
            eval_metric=lgb_metric,
        )

        y_pred = lgb.predict_proba(X_valid)
        preds_test = lgb.predict_proba(test_df.drop(["Id"], axis=1).loc[:, COLS].values)

        final_test_predictions.append(preds_test)
        final_valid_predictions.update(dict(zip(valid_ids, y_pred)))

        logloss = log_loss(y_valid, y_pred)
        balanced_logloss = balanced_log_loss(y_valid, y_pred[:, 1])
        log_losses.append(logloss)
        balanced_log_losses.append(balanced_logloss)
        weights.append(1 / balanced_logloss)

        print(
            f"Fold: {fold}, log loss: {round(logloss, 3)}, balanced los loss: {round(balanced_logloss, 3)}"
        )

    return (
        final_valid_predictions,
        final_test_predictions,
        log_losses,
        balanced_log_losses,
        weights,
    )


def inference(test_df, final_test_predictions, weights):
    test_preds = np.zeros((test_df.shape[0], 2))
    for i in range(5):
        test_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
        test_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
    test_preds /= sum(weights)
    return test_preds


def main():
    train, test, _, greeks = load_data()
    df, test_df = feature_eng(train, test)
    (
        final_valid_predictions,
        final_test_predictions,
        log_losses,
        balanced_log_losses,
        weights,
    ) = training(df, test_df, greeks)
    test_preds = inference(test_df, final_test_predictions, weights)

    print()
    print("Log Loss")
    print(log_losses)
    print(np.mean(log_losses), np.std(log_losses))
    print()
    print("Balanced Log Loss")
    print(balanced_log_losses)
    print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
    print()
    print("Weights")
    print(weights)

    final_valid_predictions = pd.DataFrame.from_dict(
        final_valid_predictions, orient="index"
    ).reset_index()
    final_valid_predictions.columns = ["Id", "class_0", "class_1"]
    final_valid_predictions.to_csv("oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), test_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ["Id", "class_0", "class_1"]

    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
