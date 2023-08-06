#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")


# ### select submission

# In[2]:


LGB = False
XGB = False
PFN = False
CAT = False
LOG = False
ENSEMBLE = True


# ### Data Prep

# In[3]:


COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"
train = pd.read_csv(f"{COMP_PATH}/train.csv")
test = pd.read_csv(f"{COMP_PATH}/test.csv")
sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")


# In[4]:


use_cols = ['Id', 'DE', 'EL', 'GH', 'FE', 'DY', 'EE', 'EU', 'CH', 'CD ', 'CC', 'GL', 'DL',
            'EB', 'AF', 'FI', 'DN', 'DA', 'FL', 'CR', 'FR', 'AB', 'BQ', 'DU', 'Class']
use_cols_test = ['Id', 'DE', 'EL', 'GH', 'FE', 'DY', 'EE', 'EU', 'CH', 'CD ', 'CC', 'GL', 'DL',
            'EB', 'AF', 'FI', 'DN', 'DA', 'FL', 'CR', 'FR', 'AB', 'BQ', 'DU']


# In[5]:


# 特徴量の削減
train = train[use_cols]
test = test[use_cols_test]


# In[6]:


train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
train.head()


# In[7]:


cols = train.columns
feat_cols = cols[1:]
num_cols = train.select_dtypes(include=['float64']).columns
print("No of Columns:", len(cols))


# ### Metric

# In[8]:


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


# In[9]:


def lgb_metric(y_true, y_pred):
    return 'balanced_log_loss', balanced_log_loss(y_true, y_pred), False


# ### Feature Engineering

# In[10]:


# Label encoding
#train['EJ'] = train['EJ'].map({'A': 0, 'B': 1})
#test['EJ']  = test['EJ'].map({'A': 0, 'B': 1})


# In[11]:


df, test_df = train.copy(), test.copy()
df


# In[12]:


# スケーリング
scaler = StandardScaler()
new_num_cols = train.select_dtypes(include=['float64']).columns
df[new_num_cols] = scaler.fit_transform(train[new_num_cols])
test_df[new_num_cols] = scaler.transform(test[new_num_cols])


# ### CV

# In[13]:


kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
df['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(kf.split(df, greeks['Alpha'])):
    df.loc[test_idx, 'fold'] = fold

df.groupby('fold')["Class"].value_counts()


# ### Training

# ### LGBM

# In[14]:


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

    X_train, y_train = train_df.drop(['Id','Class', 'fold'], axis=1), train_df['Class']
    X_valid, y_valid = valid_df.drop(['Id', 'Class', 'fold'], axis=1), valid_df['Class']
    
    lgb = LGBMClassifier(boosting_type='goss', learning_rate=0.06733232950390658, n_estimators = 50000, 
                         early_stopping_round = 300, random_state=42,
                        subsample=0.6970532011679706,
                        colsample_bytree=0.6055755840633003,
                         class_weight='balanced',
                         metric='none', is_unbalance=True, max_depth=8)
    
    lgb.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=1000,
            eval_metric=lgb_metric)
    
    y_pred = lgb.predict_proba(X_valid)
    preds_test  = lgb.predict_proba(test_df.drop(['Id'], axis=1).values)
    
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
print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
print()
print("Weights")
print(weights)


# In[15]:


lgb_preds = np.zeros((test_df.shape[0],2))
for i in range(5):
    lgb_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    lgb_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
lgb_preds /= sum(weights)
lgb_preds


# In[16]:


if LGB is True:
    final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
    final_valid_predictions.to_csv(r"oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), lgb_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ['Id', 'class_0', 'class_1']                       

    submission.to_csv(r"submission.csv", index=False)
    submission
else:
    pass


# ###  XGB

# In[17]:


from sklearn.utils.class_weight import compute_sample_weight


# In[18]:


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

    X_train, y_train = train_df.drop(['Id','Class', 'fold'], axis=1), train_df['Class']
    X_valid, y_valid = valid_df.drop(['Id', 'Class', 'fold'], axis=1), valid_df['Class']
    
    # 各foldでのweightを定義
    weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    
    xgb = XGBClassifier(n_estimators=10000, n_jobs=-1, max_depth=4, eta=0.1, colsample_bytree=0.67)
    xgb.fit(X_train, y_train, sample_weight=weight_train, eval_set=[(X_train, y_train),(X_valid, y_valid)], early_stopping_rounds=300, verbose=300)
    
    y_pred = xgb.predict_proba(X_valid)
    preds_test  = xgb.predict_proba(test_df.drop(['Id'], axis=1).values)
    
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
print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
print()
print("Weights")
print(weights)


# In[19]:


xgb_preds = np.zeros((test_df.shape[0],2))
for i in range(5):
    xgb_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    xgb_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
xgb_preds /= sum(weights)
xgb_preds


# In[20]:


if XGB is True:
    #final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    #final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
    #final_valid_predictions.to_csv(r"oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), xgb_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ['Id', 'class_0', 'class_1']                       

    submission.to_csv(r"submission.csv", index=False)
    print(submission)
else:
    pass


# # CatBoost

# In[21]:


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

    X_train, y_train = train_df.drop(['Id','Class', 'fold'], axis=1), train_df['Class']
    X_valid, y_valid = valid_df.drop(['Id', 'Class', 'fold'], axis=1), valid_df['Class']
    
    train_pool = Pool(X_train, y_train)
    validate_pool = Pool(X_valid, y_valid)
    
    cat = CatBoostClassifier(random_state = 91, depth=4, l2_leaf_reg = 5, objective= "Logloss", auto_class_weights = "Balanced", one_hot_max_size=10)
    cat.fit(train_pool, 
            eval_set = validate_pool, 
            verbose = 200, 
            early_stopping_rounds=200,  # 10回以上精度が改善しなければ中止
            use_best_model=True,       # 最も精度が高かったモデルを使用するかの設定
            plot=True)
    
    y_pred = cat.predict_proba(X_valid)
    preds_test  = cat.predict_proba(test_df.drop(['Id'], axis=1).values)
    
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
print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
print()
print("Weights")
print(weights)


# In[22]:


cat_preds = np.zeros((test_df.shape[0],2))
for i in range(5):
    cat_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    cat_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
cat_preds /= sum(weights)
cat_preds


# In[23]:


if CAT is True:
    #final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    #final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
    #final_valid_predictions.to_csv(r"oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), cat_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ['Id', 'class_0', 'class_1']                       

    submission.to_csv(r"submission.csv", index=False)
    print(submission)
else:
    pass


# # TabPFN

# In[24]:


get_ipython().system('pip install tabpfn --no-index --find-links=file:///kaggle/input/pip-packages-icr/pip-packages')
get_ipython().system('mkdir -p /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff')
get_ipython().system('cp /kaggle/input/pip-packages-icr/pip-packages/prior_diff_real_checkpoint_n_0_epoch_100.cpkt /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff/')


# In[25]:


from tabpfn import TabPFNClassifier


# In[26]:


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

    X_train, y_train = train_df.drop(['Id','Class', 'fold'], axis=1), train_df['Class']
    X_valid, y_valid = valid_df.drop(['Id', 'Class', 'fold'], axis=1), valid_df['Class']
    
    # 各foldでのweightを定義
    #weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    
    pfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
    pfn.fit(X_train, y_train)
    
    y_pred = pfn.predict_proba(X_valid)
    preds_test  = pfn.predict_proba(test_df.drop(['Id'], axis=1).values)
    
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
print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
print()
print("Weights")
print(weights)


# In[27]:


pfn_preds = np.zeros((test_df.shape[0],2))
for i in range(5):
    pfn_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    pfn_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
pfn_preds /= sum(weights)
pfn_preds


# In[28]:


if PFN is True:
    #final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    #final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
    #final_valid_predictions.to_csv(r"oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), pfn_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ['Id', 'class_0', 'class_1']                       

    submission.to_csv(r"submission.csv", index=False)
    print(submission)
else:
    pass


# # Logistic Classification

# In[29]:


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

    X_train, y_train = train_df.drop(['Id','Class', 'fold'], axis=1), train_df['Class']
    X_valid, y_valid = valid_df.drop(['Id', 'Class', 'fold'], axis=1), valid_df['Class']
    
    # 各foldでのweightを定義
    #weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    
    log = LogisticRegression(penalty = 'l2', max_iter = 15000, class_weight='balanced', random_state = 91, solver = 'liblinear')
    log.fit(X_train, y_train)
    
    y_pred = log.predict_proba(X_valid)
    preds_test  = log.predict_proba(test_df.drop(['Id'], axis=1).values)
    
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
print(np.mean(balanced_log_losses), np.std(balanced_log_losses))
print()
print("Weights")
print(weights)


# In[30]:


log_preds = np.zeros((test_df.shape[0],2))
for i in range(5):
    log_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    log_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
log_preds /= sum(weights)
log_preds


# In[31]:


if LOG is True:
    #final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    #final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
    #final_valid_predictions.to_csv(r"oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), log_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ['Id', 'class_0', 'class_1']                       

    submission.to_csv(r"submission.csv", index=False)
    print(submission)
else:
    pass


# ### Ensemble

# In[32]:


ensemble_preds = ((2 * lgb_preds) + xgb_preds + cat_preds + log_preds + pfn_preds)/6
ensemble_preds


# In[33]:


if ENSEMBLE is True:
    #final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    #final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
    #final_valid_predictions.to_csv(r"oof.csv", index=False)

    test_dict = {}
    test_dict.update(dict(zip(test.Id.values.tolist(), ensemble_preds)))
    submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
    submission.columns = ['Id', 'class_0', 'class_1']                       

    submission.to_csv(r"submission.csv", index=False)
    print(submission)
else:
    pass

