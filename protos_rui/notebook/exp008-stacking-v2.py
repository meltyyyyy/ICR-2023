#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tabpfn --no-index --find-links=file:///kaggle/input/pip-packages-icr/pip-packages')
get_ipython().system('mkdir -p /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff')
get_ipython().system('cp /kaggle/input/pip-packages-icr/pip-packages/prior_diff_real_checkpoint_n_0_epoch_100.cpkt /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff/')


# In[2]:


import warnings

import seaborn as sns
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier
import numpy as np
from sklearn.model_selection._split import (
    BaseShuffleSplit,
    _BaseKFold,
    _RepeatedSplits,
    _validate_shuffle_split,
)
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_array


# In[3]:


class config:
    fold = 5


# In[4]:


warnings.filterwarnings("ignore")
COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"


# # Utils

# In[5]:


INT_DENOMINATORS = {
    "AB": 0.004273,
    "AF": 0.00242,
    "AH": 0.008709,
    "AM": 0.003097,
    "AR": 0.005244,
    "AX": 0.008859,
    "AY": 0.000609,
    "AZ": 0.006302,
    "BC": 0.007028,
    "BD ": 0.00799,
    "BN": 0.3531,
    "BP": 0.004239,
    "BQ": 0.002605,
    "BR": 0.006049,
    "BZ": 0.004267,
    "CB": 0.009191,
    "CC": 6.12e-06,
    "CD ": 0.007928,
    "CF": 0.003041,
    "CH": 0.000398,
    "CL": 0.006365,
    "CR": 7.5e-05,
    "CS": 0.003487,
    "CU": 0.005517,
    "CW ": 9.2e-05,
    "DA": 0.00388,
    "DE": 0.004435,
    "DF": 0.000351,
    "DH": 0.002733,
    "DI": 0.003765,
    "DL": 0.00212,
    "DN": 0.003412,
    "DU": 0.0013794,
    "DV": 0.00259,
    "DY": 0.004492,
    "EB": 0.007068,
    "EE": 0.004031,
    "EG": 0.006025,
    "EH": 0.006084,
    "EL": 0.000429,
    "EP": 0.009269,
    "EU": 0.005064,
    "FC": 0.005712,
    "FD ": 0.005937,
    "FE": 0.007486,
    "FI": 0.005513,
    "FR": 0.00058,
    "FS": 0.006773,
    "GB": 0.009302,
    "GE": 0.004417,
    "GF": 0.004374,
    "GH": 0.003721,
    "GI": 0.002572,
}


# In[6]:


def competition_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1
    return (log_loss_0 + log_loss_1) / 2


def balanced_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    balanced_log_loss = 2 * (w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    return balanced_log_loss / (N_0 + N_1)


def lgb_metric(y_true, y_pred):
    return "balanced_log_loss", balanced_log_loss(y_true, y_pred), False


# # Training

# In[7]:


def load_data():
    train = pd.read_csv(f"{COMP_PATH}/train.csv")
    test = pd.read_csv(f"{COMP_PATH}/test.csv")
    sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
    greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")
    return train, test, sample_submission, greeks


# In[8]:


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


# In[9]:


def to_int(df):
    for k, v in INT_DENOMINATORS.items():
        df[k] = np.round(df[k] / v, 1)
    return df


# In[10]:


def plot_predictions_distribution(true_values, predictions, title="pred_dist.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    indexes = np.arange(len(predictions))  # Create an array of index values
    true_values = np.array(true_values)  # Ensure true_values is an array for element-wise comparison
    predictions = np.array(predictions)  # Ensure predictions is an array for element-wise comparison

    # Plot all predictions and true values
    ax.scatter(indexes, true_values, label='True Values', s=10)

    # Create a mask for incorrect predictions where prob > 0.5 but true label is 0
    incorrect_mask_1 = (predictions > 0.5) & (true_values == 0)

    # Create a mask for incorrect predictions where prob < 0.5 but true label is 1
    incorrect_mask_2 = (predictions < 0.5) & (true_values == 1)

    # Create a mask for correct predictions
    correct_mask = (predictions >= 0.5) & (true_values == 1) | (predictions < 0.5) & (true_values == 0)

    # Plot the predictions using the masks for coloring
    ax.scatter(indexes[correct_mask], predictions[correct_mask], label='Correct Predictions', alpha=0.5, s=10, color='blue')
    ax.scatter(indexes[incorrect_mask_1], predictions[incorrect_mask_1], label='prob > 0.5 but true label is 0', alpha=0.5, s=10, color='red')
    ax.scatter(indexes[incorrect_mask_2], predictions[incorrect_mask_2], label='prob < 0.5 but true label is 1', alpha=0.5, s=10, color='green')

    ax.set_xlabel('Index')
    ax.set_ylabel('Probability')
    plt.title('Scatter plot of Predictions and True Values')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(title, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# # Models

# In[11]:


def training_l1(train, test, models_list, model_name, use_cols):
    oof_list = []
    test_list = []
    
    X_train = train[use_cols]
    y_train = train["Class"]
    X_test = test[use_cols]
    
    weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    
    for i, model_l1 in enumerate(models_list):
        print(f'{"*" * 50}\n{model_name[i]}')
        if model_name[i] == "XGB":
            model = model_l1.fit(X_train, y_train, sample_weight=weight_train)
        if model_name[i] == "CAT":
            model = model_l1.fit(X_train, y_train, sample_weight=weight_train, verbose=False)
        else:
            model = model_l1.fit(X_train, y_train)
            
        oof_preds = model.predict_proba(X_train)[:, 1]
        oof_score = balanced_log_loss(y_train, oof_preds)
        test_preds = model.predict_proba(X_test)[:, 1]
        print(oof_score)
        plot_predictions_distribution(y_train, oof_preds, f"pred_dist{model_name[i]}_80.png")
        
        oof_list.append(oof_preds)
        test_list.append(test_preds)
    return oof_list, test_list


# In[12]:


def make_stacking_dataset(oof_list, test_list, y_train):
    train_l2 = pd.DataFrame({
        'lgb_pred': oof_list[0],
        'xgb_pred': oof_list[1],
        'cat_pred': oof_list[2],
        'pfn_pred': oof_list[3],
        'log_pred': oof_list[4],
        'train_y': y_train,
        })
    
    test_l2 = pd.DataFrame({
        'lgb_pred': test_list[0],
        'xgb_pred': test_list[1],
        'cat_pred': test_list[2],
        'pfn_pred': test_list[3],
        'log_pred': test_list[4],
        })
    return train_l2, test_l2


# In[13]:


def get_l2_data(models_list, model_name):
    train, test, _, greeks = load_data()
    df, test_df = feature_eng(train, test)
    use_cols = ['DE', 'EL', 'GH', 'FE', 'DY', 'EE', 'EU', 'CH', 'CD ', 'CC', 'GL', 'DL','EB', 'AF', 'FI', 'DN', 'DA', 'FL', 'CR', 'FR', 'AB', 'BQ', 'DU']
    target = "Class"
    oof_list, test_list = training_l1(df, test_df, models_list, model_name, use_cols)
    train_l2, test_l2 = make_stacking_dataset(oof_list, test_list, train[target])
    
    return train_l2, test_l2, test


# In[14]:


def training_l2(train_l2, test_l2):
    X_train = train_l2.drop("train_y", axis=1)
    y_train = train_l2["train_y"]
    X_test = test_l2
    
    log = LogisticRegression(penalty = 'l2', max_iter = 1000, class_weight='balanced', random_state = 91, solver = 'liblinear')
    model = log.fit(X_train, y_train)
            
    oof_preds = model.predict_proba(X_train)[:, 1]
    oof_score = balanced_log_loss(y_train, oof_preds)
    test_preds = model.predict_proba(X_test)[:, 1]
    print(oof_score)
    plot_predictions_distribution(y_train, oof_preds, f"pred_dist_l2_80.png")

    return test_preds


# In[15]:


def main():
    N_ESTIMATOR = 100
    lgb = LGBMClassifier(
                boosting_type="goss",
                learning_rate=0.1,
                n_estimators=N_ESTIMATOR,
                random_state=42,
                subsample=1,
                colsample_bytree=1,
                class_weight="balanced",
                metric="none",
                is_unbalance=True,
                max_depth=4,
            )
    xgb = XGBClassifier(n_estimators=N_ESTIMATOR, n_jobs=-1, max_depth=4, eta=0.1, colsample_bytree=0.67)
    cat = CatBoostClassifier(random_state = 91, depth=4, n_estimators = N_ESTIMATOR, l2_leaf_reg = 5, objective= "Logloss", auto_class_weights = "Balanced", one_hot_max_size=10)
    pfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
    log = LogisticRegression(penalty = 'l2', max_iter = 15000, class_weight='balanced', random_state = 91, solver = 'liblinear')
    models_list = [lgb, xgb, cat, pfn, log]
    model_name = ["LGB", "XGB", "CAT", "PFN", "LOG"]
    
    train_l2, test_l2, test = get_l2_data(models_list, model_name)
    test_pred = training_l2(train_l2, test_l2)
    submission = pd.DataFrame(test.Id)
    submission["class_0"] = 1-test_pred
    submission["class_1"] = test_pred
    submission.to_csv(r"submission.csv", index=False)
    return submission


# In[16]:


submission = main()


# In[17]:


submission.head()


# In[ ]:




