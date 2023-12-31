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
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


# ### Data Prep

# In[3]:


COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"
train = pd.read_csv(f"{COMP_PATH}/train.csv")
test = pd.read_csv(f"{COMP_PATH}/test.csv")
sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")


# In[4]:


train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
train.head()


# In[5]:


cols = train.columns
feat_cols = cols[1:]
num_cols = train.select_dtypes(include=['float64']).columns
print("No of Columns:", len(cols))


# ### Metric

# In[6]:


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


# In[7]:


def lgb_metric(y_true, y_pred):
    return 'balanced_log_loss', balanced_log_loss(y_true, y_pred), False


# ### Feature Engineering

# In[8]:


# Label encoding
train['EJ'] = train['EJ'].map({'A': 0, 'B': 1})
test['EJ']  = test['EJ'].map({'A': 0, 'B': 1})


# In[9]:


scaler = StandardScaler()
df, test_df = train.copy(), test.copy()
new_num_cols = train.select_dtypes(include=['float64']).columns
df[new_num_cols] = scaler.fit_transform(train[new_num_cols])
test_df[new_num_cols] = scaler.transform(test[new_num_cols])
df


# ## K-Means

# In[10]:


# エルボー法によるK-meansの可視化関数の定義
def plot_kmeans(input_data, num_cluster):
    dist_list =[]
    for i in range(1,num_cluster):
        kmeans= KMeans(n_clusters=i, init='random', random_state=0)
        kmeans.fit(input_data)
        dist_list.append(kmeans.inertia_)
    
    # グラフを表示
    plt.plot(range(1,num_cluster), dist_list,marker='+')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')


# In[11]:


# K-Meansの可視化の上限設定
NUM_CLUSTER = 50


# In[12]:


# エルボー法によるK-Meansの実施
plot_kmeans(df.drop(["Id", "EJ", "Class"], axis=1), NUM_CLUSTER)


# In[13]:


# クラスを含めない状態でクラスター数7でラベリング
num_cluster = 5

# kmeansクラスをインスタンス化
kmeans = KMeans(init='random', n_clusters=num_cluster, random_state=0)

# train dataでトレーニング
kmeans.fit(df.drop(["Id", "EJ", "Class"], axis=1))

# predict
km_label_train =pd.Series(kmeans.labels_, name='cluster_number_{}'.format(num_cluster))
km_label_test =pd.Series(kmeans.predict(test_df.drop(["Id", "EJ"], axis=1)), name='cluster_number_{}'.format(num_cluster))


# In[14]:


# 本番データに結合
df["cluster_label"] = km_label_train
test_df["cluster_label"] = km_label_test


# In[15]:


df.groupby(["cluster_label"])["Class"].sum()


# ## aggrigationの実装

# In[16]:


test_df["cluster_label"]


# In[17]:


agg_cols = ['min', 'max', 'mean', 'std']
cat_cols = ["cluster_label"]

for col in cat_cols:
    grp_df = df.groupby(col)[num_cols].agg(agg_cols)
    grp_df.columns = [f'{col}_' + '_'.join(c) for c in grp_df.columns]
    df = df.merge(grp_df, on=col, how='left')
    test_df = test_df.merge(grp_df, on=col, how='left')


# ## 特徴選択

# In[18]:


# thresholdを超える相関を持つカラムの削除
threshold = 0.8
print(len(df.columns))

feat_corr = set()
corr_matrix = df.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feat_name = corr_matrix.columns[i]
            feat_corr.add(feat_name)

print(len(set(feat_corr)))

df.drop(labels=feat_corr, axis='columns', inplace=True)
test_df.drop(labels=feat_corr, axis='columns', inplace=True)

print(len(df.columns))


# ### CV

# In[19]:


kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
df['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(kf.split(df, greeks['Alpha'])):
    df.loc[test_idx, 'fold'] = fold

df.groupby('fold')["Class"].value_counts()


# ### Training

# In[20]:


final_valid_predictions = {}
final_test_predictions = []
scores = []
log_losses = []
balanced_log_losses = []
weights = []
importance = None

for fold in range(5):
    train_df = df[df['fold'] != fold]
    valid_df = df[df['fold'] == fold]
    valid_ids = valid_df.Id.values.tolist()

    X_train, y_train = train_df.drop(['Id', 'Class', 'fold'], axis=1), train_df['Class']
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
    
    # feature importanceの算出
    if importance is None:
        importance = pd.DataFrame(lgb.feature_importances_, index=X_train.columns, columns=['fold{}_importance'.format(fold)])
    else:
        importance['fold{}_importance'.format(fold)] = lgb.feature_importances_

    
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


# ## 特徴選択

# In[21]:


fold_num = 5
for i in range(fold_num):
    if i == 0:
        importance["sum_imp"] = importance["fold{}_importance".format(i)]
    else:
        importance["sum_imp"] += importance["fold{}_importance".format(i)]


# In[22]:


# 上位X個のカラムを使用
use_col_rank = 25


# In[23]:


importance.sort_values('sum_imp', ascending=False, inplace=True)
importance.head()


# In[24]:


len(importance.iloc[:use_col_rank])


# In[25]:


selected_cols = importance.iloc[:use_col_rank].index.tolist()


# In[26]:


final_valid_predictions = {}
final_test_predictions = []
scores = []
log_losses = []
balanced_log_losses = []
weights = []
importance = None

selected_df = df[selected_cols]
selected_df["fold"] = df["fold"]
selected_df["Class"] = df["Class"]
selected_df["Id"] = df["Id"]
selected_test_df = test_df[selected_cols]
selected_test_df["Id"] = test_df["Id"]

for fold in range(5):
    train_df = selected_df[df['fold'] != fold]
    valid_df = selected_df[df['fold'] == fold]
    valid_ids = valid_df.Id.values.tolist()

    X_train, y_train = train_df.drop(['Id', 'Class', 'fold'], axis=1), train_df['Class']
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
    preds_test  = lgb.predict_proba(selected_test_df.drop(['Id'], axis=1).values)
    
    final_test_predictions.append(preds_test)
    final_valid_predictions.update(dict(zip(valid_ids, y_pred)))

    logloss = log_loss(y_valid, y_pred)
    balanced_logloss = balanced_log_loss(y_valid, y_pred[:, 1])
    log_losses.append(logloss)
    balanced_log_losses.append(balanced_logloss)
    weights.append(1/balanced_logloss)
    
    # feature importanceの算出
    if importance is None:
        importance = pd.DataFrame(lgb.feature_importances_, index=X_train.columns, columns=['fold{}_importance'.format(fold)])
    else:
        importance['fold{}_importance'.format(fold)] = lgb.feature_importances_

    
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


# ## submission

# In[27]:


test_preds = np.zeros((test_df.shape[0],2))
for i in range(5):
    test_preds[:, 0] += weights[i] * final_test_predictions[i][:, 0]
    test_preds[:, 1] += weights[i] * final_test_predictions[i][:, 1]
test_preds /= sum(weights)
test_preds


# In[28]:


final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ['Id', 'class_0', 'class_1']
final_valid_predictions.to_csv(r"oof.csv", index=False)

test_dict = {}
test_dict.update(dict(zip(test.Id.values.tolist(), test_preds)))
submission = pd.DataFrame.from_dict(test_dict, orient="index").reset_index()
submission.columns = ['Id', 'class_0', 'class_1']                       

submission.to_csv(r"submission.csv", index=False)
submission


# In[ ]:





# In[ ]:




