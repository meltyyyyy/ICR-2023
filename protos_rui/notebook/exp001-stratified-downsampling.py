

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import math

from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold

from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")





COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"
train = pd.read_csv(f"{COMP_PATH}/train.csv")
test = pd.read_csv(f"{COMP_PATH}/test.csv")
sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")




train.head()




train.info()




train_droped = train.dropna(subset=["BQ"], how="any")
train_bq_isnull = train[train["BQ"].isnull()]
train_bq_isnull.reset_index(inplace=True)
train_bq_isnull.drop(["index"], axis=1, inplace=True)
print(len(train_droped))
train_droped.reset_index(inplace=True)
train_droped.drop(["index"], axis=1, inplace=True)
train_droped.head()




test_droped = test.dropna(subset=["BQ"], how="any")
test_bq_isnull = train[train["BQ"].isnull()]
test_bq_isnull.reset_index(inplace=True)
test_bq_isnull.drop(["index"], axis=1, inplace=True)
print(len(test_droped))
test_droped.reset_index(inplace=True)
test_droped.drop(["index"], axis=1, inplace=True)
test_droped.head()




train.fillna(train.median(), inplace=True)
test.fillna(test_droped.median(), inplace=True)
train.head()




cols = train_droped.columns
feat_cols = cols[1:]
num_cols = train_droped.select_dtypes(include=['float64']).columns
print("No of Columns:", len(cols))





def competition_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1
    return (log_loss_0 + log_loss_1)/2

def balanced_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    return balanced_log_loss/(N_0+N_1)




def lgb_metric(y_true, y_pred):
    return 'balanced_log_loss', balanced_log_loss(y_true, y_pred), False





train_droped['EJ'] = train_droped['EJ'].map({'A': 0, 'B': 1})
test['EJ']  = test['EJ'].map({'A': 0, 'B': 1})




"""
scaler = StandardScaler()
df, test_df = train.copy(), test.copy()
new_num_cols = train.select_dtypes(include=['float64']).columns
df[new_num_cols] = scaler.fit_transform(train[new_num_cols])
test_df[new_num_cols] = scaler.transform(test[new_num_cols])
df
"""






BN_SCALER = 0.3531
train_droped["BN"] = train["BN"]/BN_SCALER
train_droped["BN"].unique()




train_droped["BN"].describe()





train_droped["BN_label"] = 0
for i in range(len(train_droped)):
    if (train_droped.loc[i, "BN"] > 40) & (train_droped.loc[i, "BN"] <= 60):
        train_droped.loc[i, "BN_label"] = 1
    elif (train_droped.loc[i, "BN"] > 60) & (train_droped.loc[i, "BN"] <= 70):
        train_droped.loc[i, "BN_label"] = 2
    elif (train_droped.loc[i, "BN"] > 70) & (train_droped.loc[i, "BN"] <= 80):
        train_droped.loc[i, "BN_label"] = 3
train_droped["BN_label"].unique()





from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.75, random_state=0)




class0_train = train_droped[train_droped["Class"]==0]
for i, (train_index, test_index) in enumerate(sss.split(class0_train, class0_train["BN_label"])):
    print(train_index)
    print(len(train_index))




train_droped["Class"].sum()




under_sampled_df = class0_train.iloc[test_index]
len(under_sampled_df)




df = pd.concat([under_sampled_df, train_droped[train_droped["Class"] == 1]])




df.reset_index(inplace=True)
df.head()




df.drop(["index"], axis=1, inplace=True)
len(df)





df.info()




kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
df['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(kf.split(df, df["BN_label"])):
    df.loc[test_idx, 'fold'] = fold

df.groupby('fold')["Class"].value_counts()





final_valid_predictions = {}
final_test_predictions = []
scores = []
log_losses = []
balanced_log_losses = []
weights = []

for fold in range(5):
    train_df = df[df['fold'] != fold]
    valid_df = df[df['fold'] == fold]
    valid_ids = valid_df.Id.values.tolist()

    X_train, y_train = train_df.drop(['Id', 'Class', 'fold'], axis=1), train_df['Class']
    X_valid, y_valid = valid_df.drop(['Id', 'Class', 'fold'], axis=1), valid_df['Class']
    X_train.drop(["BN_label"], axis=1, inplace=True)
    X_valid.drop(["BN_label"], axis=1, inplace=True)
    
    lgb = LGBMClassifier(boosting_type='goss', learning_rate=0.06733232950390658, n_estimators = 50000, 
                         early_stopping_round = 300, random_state=42,
                        subsample=0.6970532011679706,
                        colsample_bytree=0.6055755840633003,
                         class_weight='balanced',
                         metric='none', is_unbalance=True, max_depth=8)
    
    lgb.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=1000,
            eval_metric=lgb_metric)
    
    y_pred = lgb.predict_proba(X_valid)
    preds_test  = lgb.predict_proba(test.drop(['Id'], axis=1).values)
    
    final_test_predictions.append(preds_test)
    final_valid_predictions.update(dict(zip(valid_ids, y_pred)))

    logloss = log_loss(y_valid, y_pred)
    balanced_logloss = balanced_log_loss(y_valid, y_pred[:, 1])
    log_losses.append(logloss)
    balanced_log_losses.append(balanced_logloss)
    weights.append(1/balanced_logloss)
    
    print(f"Fold: {fold}, log loss: {round(logloss, 3)}, balanced los loss: {round(balanced_logloss, 3)}")

print()
print("Log Loss")
print(log_losses)
print(np.mean(log_losses), np.std(log_losses))
print()
print("Balanced Log Loss")
print(balanced_log_losses)
print("MEAN : ", np.mean(balanced_log_losses), "STD : ", np.std(balanced_log_losses))
print()
print("Weights")
print(weights)




test_preds = np.zeros((test.shape[0],2))
for i in range(5):
    test_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    test_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
test_preds /= sum(weights)
test_preds




final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
final_valid_predictions.to_csv(r"oof.csv", index=False)

test_dict = {}
test_dict.update(dict(zip(test.Id.values.tolist(), test_preds)))
submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
submission.columns = ['Id', 'class_0', 'class_1']
submission.loc[test["BQ"].isnull(), 'class_0'] = 1
submission.loc[test["BQ"].isnull(), 'class_1'] = 0

submission.to_csv(r"submission.csv", index=False)
submission






