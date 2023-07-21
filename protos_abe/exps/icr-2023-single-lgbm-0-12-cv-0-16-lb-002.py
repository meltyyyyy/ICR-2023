import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


COMP_PATH = "/kaggle/input/icr-identify-age-related-conditions"
train = pd.read_csv(f"{COMP_PATH}/train.csv")
test = pd.read_csv(f"{COMP_PATH}/test.csv")
sample_submission = pd.read_csv(f"{COMP_PATH}/sample_submission.csv")
greeks = pd.read_csv(f"{COMP_PATH}/greeks.csv")


int_denominators = {
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


def to_int(df):
    for k, v in int_denominators.items():
        df[k] = np.round(df[k] / v, 1)
    return df


train = to_int(train.copy())
test = to_int(test.copy())

train.head()


train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)


cols = train.columns
feat_cols = cols[1:]
num_cols = train.select_dtypes(include=["float64"]).columns
print("No of Columns:", len(cols))


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


train["EJ"] = train["EJ"].map({"A": 0, "B": 1})
test["EJ"] = test["EJ"].map({"A": 0, "B": 1})


scaler = StandardScaler()
df, test_df = train.copy(), test.copy()
new_num_cols = train.select_dtypes(include=["float64"]).columns
df[new_num_cols] = scaler.fit_transform(train[new_num_cols])
test_df[new_num_cols] = scaler.transform(test[new_num_cols])
df


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


def IterativeStratification(labels, r, random_state):
    """This function implements the Iterative Stratification algorithm described
    in the following paper:
    Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of
    Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M.
    (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD
    2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin,
    Heidelberg.
    """

    n_samples = labels.shape[0]
    test_folds = np.zeros(n_samples, dtype=int)

    c_folds = r * n_samples

    c_folds_labels = np.outer(r, labels.sum(axis=0))

    labels_not_processed_mask = np.ones(n_samples, dtype=bool)

    while np.any(labels_not_processed_mask):
        num_labels = labels[labels_not_processed_mask].sum(axis=0)

        if num_labels.sum() == 0:
            sample_idxs = np.where(labels_not_processed_mask)[0]

            for sample_idx in sample_idxs:
                fold_idx = np.where(c_folds == c_folds.max())[0]

                if fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[random_state.choice(fold_idx.shape[0])]

                test_folds[sample_idx] = fold_idx
                c_folds[fold_idx] -= 1

            break

        label_idx = np.where(num_labels == num_labels[np.nonzero(num_labels)].min())[0]
        if label_idx.shape[0] > 1:
            label_idx = label_idx[random_state.choice(label_idx.shape[0])]

        sample_idxs = np.where(
            np.logical_and(labels[:, label_idx].flatten(), labels_not_processed_mask)
        )[0]

        for sample_idx in sample_idxs:
            label_folds = c_folds_labels[:, label_idx]
            fold_idx = np.where(label_folds == label_folds.max())[0]

            if fold_idx.shape[0] > 1:
                temp_fold_idx = np.where(c_folds[fold_idx] == c_folds[fold_idx].max())[
                    0
                ]
                fold_idx = fold_idx[temp_fold_idx]

                if temp_fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[random_state.choice(temp_fold_idx.shape[0])]

            test_folds[sample_idx] = fold_idx
            labels_not_processed_mask[sample_idx] = False

            c_folds_labels[fold_idx, labels[sample_idx]] -= 1
            c_folds[fold_idx] -= 1

    return test_folds


class MultilabelStratifiedKFold(_BaseKFold):
    """Multilabel stratified K-Folds cross-validator
    Provides train/test indices to split multilabel data into train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds for multilabel data. The folds are made by preserving
    the percentage of samples for each label.
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Unlike StratifiedKFold that only uses random_state
        when ``shuffle`` == True, this multilabel implementation
        always uses the random_state since the iterative stratification
        algorithm breaks ties randomly.
    Examples
    --------
    >>> from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    >>> import numpy as np
    >>> X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    >>> y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])
    >>> mskf = MultilabelStratifiedKFold(n_splits=2, random_state=0)
    >>> mskf.get_n_splits(X, y)
    2
    MultilabelStratifiedKFold(n_splits=2, random_state=0, shuffle=False)
    >>> for train_index, test_index in mskf.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 3 4 6] TEST: [1 2 5 7]
    TRAIN: [1 2 5 7] TEST: [0 3 4 6]
    Notes
    -----
    Train and test sizes may be slightly different in each fold.
    See also
    --------
    RepeatedMultilabelStratifiedKFold: Repeats Multilabel Stratified K-Fold
    n times.
    """

    def __init__(self, n_splits=3, *, shuffle=False, random_state=None):
        super(MultilabelStratifiedKFold, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _make_test_folds(self, X, y):
        y = np.asarray(y, dtype=bool)
        type_of_target_y = type_of_target(y)

        if type_of_target_y != "multilabel-indicator":
            raise ValueError(
                "Supported target type is: multilabel-indicator. Got {!r} instead.".format(
                    type_of_target_y
                )
            )

        num_samples = y.shape[0]

        rng = check_random_state(self.random_state)
        indices = np.arange(num_samples)

        if self.shuffle:
            rng.shuffle(indices)
            y = y[indices]

        r = np.asarray([1 / self.n_splits] * self.n_splits)

        test_folds = IterativeStratification(labels=y, r=r, random_state=rng)

        return test_folds[np.argsort(indices)]

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples, n_labels)
            The target variable for supervised learning problems.
            Multilabel stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(MultilabelStratifiedKFold, self).split(X, y, groups)


class RepeatedMultilabelStratifiedKFold(_RepeatedSplits):
    """Repeated Multilabel Stratified K-Fold cross validator.
    Repeats Mulilabel Stratified K-Fold n times with different randomization
    in each repetition.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : None, int or RandomState, default=None
        Random state to be used to generate random state for each
        repetition as well as randomly breaking ties within the iterative
        stratification algorithm.
    Examples
    --------
    >>> from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
    >>> import numpy as np
    >>> X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    >>> y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])
    >>> rmskf = RepeatedMultilabelStratifiedKFold(n_splits=2, n_repeats=2,
    ...     random_state=0)
    >>> for train_index, test_index in rmskf.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [0 3 4 6] TEST: [1 2 5 7]
    TRAIN: [1 2 5 7] TEST: [0 3 4 6]
    TRAIN: [0 1 4 5] TEST: [2 3 6 7]
    TRAIN: [2 3 6 7] TEST: [0 1 4 5]
    See also
    --------
    RepeatedStratifiedKFold: Repeats (Non-multilabel) Stratified K-Fold
    n times.
    """

    def __init__(self, n_splits=5, *, n_repeats=10, random_state=None):
        super(RepeatedMultilabelStratifiedKFold, self).__init__(
            MultilabelStratifiedKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )


class MultilabelStratifiedShuffleSplit(BaseShuffleSplit):
    """Multilabel Stratified ShuffleSplit cross-validator
    Provides train/test indices to split data into train/test sets.
    This cross-validation object is a merge of MultilabelStratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds for multilabel
    data. The folds are made by preserving the percentage of each label.
    Note: like the ShuffleSplit strategy, multilabel stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.
    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.1.
        The default will change in version 0.21. It will remain 0.1 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    train_size : float, int, or None, default is None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Unlike StratifiedShuffleSplit that only uses
        random_state when ``shuffle`` == True, this multilabel implementation
        always uses the random_state since the iterative stratification
        algorithm breaks ties randomly.
    Examples
    --------
    >>> from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    >>> import numpy as np
    >>> X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    >>> y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])
    >>> msss = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=0.5,
    ...    random_state=0)
    >>> msss.get_n_splits(X, y)
    3
    MultilabelStratifiedShuffleSplit(n_splits=3, random_state=0, test_size=0.5,
                                     train_size=None)
    >>> for train_index, test_index in msss.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 2 5 7] TEST: [0 3 4 6]
    TRAIN: [2 3 6 7] TEST: [0 1 4 5]
    TRAIN: [1 2 5 6] TEST: [0 3 4 7]
    Notes
    -----
    Train and test sizes may be slightly different from desired due to the
    preference of stratification over perfectly sized folds.
    """

    def __init__(
        self, n_splits=10, *, test_size="default", train_size=None, random_state=None
    ):
        super(MultilabelStratifiedShuffleSplit, self).__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

    def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        y = np.asarray(y, dtype=bool)
        type_of_target_y = type_of_target(y)

        if type_of_target_y != "multilabel-indicator":
            raise ValueError(
                "Supported target type is: multilabel-indicator. Got {!r} instead.".format(
                    type_of_target_y
                )
            )

        n_train, n_test = _validate_shuffle_split(
            n_samples, self.test_size, self.train_size
        )

        n_samples = y.shape[0]
        rng = check_random_state(self.random_state)
        y_orig = y.copy()

        r = np.array([n_train, n_test]) / (n_train + n_test)

        for _ in range(self.n_splits):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            y = y_orig[indices]

            test_folds = IterativeStratification(labels=y, r=r, random_state=rng)

            test_idx = test_folds[np.argsort(indices)] == 1
            test = np.where(test_idx)[0]
            train = np.where(~test_idx)[0]

            yield train, test

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples, n_labels)
            The target variable for supervised learning problems.
            Multilabel stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(MultilabelStratifiedShuffleSplit, self).split(X, y, groups)


kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1

for fold, (train_idx, test_idx) in enumerate(kf.split(X=train, y=greeks.iloc[:, 1:3])):
    df.loc[test_idx, "fold"] = fold

df.groupby("fold")["Class"].value_counts()


metric = balanced_log_loss
final_valid_predictions = {}
final_test_predictions = []
scores = []
log_losses = []
balanced_log_losses = []
weights = []
outer_cv_score = []  # store all cv scores of outer loop inference
inner_cv_score = []  # store all cv scores of inner loop training
models = []
DU_threshold = 0.266  # The upper whisker value

for fold in range(5):
    train_df = df[df["fold"] != fold]
    valid_df = df[df["fold"] == fold]
    valid_ids = valid_df.Id.values.tolist()

    X_train, y_train = train_df.drop(["Id", "Class", "fold"], axis=1), train_df["Class"]
    X_valid, y_valid = valid_df.drop(["Id", "Class", "fold"], axis=1), valid_df["Class"]

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
    models_ = []

    print(
        f"Outer Loop fold {fold}, Inner Loop Training with {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    for fold, (fit_idx, val_idx) in enumerate(cv.split(X=X_train, y=y_train)):
        X_fit = X_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]
        y_fit = y_train.iloc[fit_idx]
        y_val = y_train.iloc[val_idx]

        model = lgb.fit(
            X_fit, y_fit, eval_set=(X_valid, y_valid), verbose=0, eval_metric=lgb_metric
        )
        models_.append(model)
        val_preds = model.predict(X_val)
        
        # Apply rule-based approach
        rule_based_preds = np.where(X_val['DU'] > DU_threshold, 1, 0)
        val_preds = np.where(rule_based_preds == 1, 1, val_preds)
        
        oof_inner[val_idx] = val_preds
        val_score = balanced_log_loss(y_val, val_preds)
        best_iter = model.booster_.best_iteration
        print(
            f"Fold: {fold:>3}| {metric.__name__}: {val_score:.5f}"
            f" | Best iteration: {best_iter:>4}"
        )

    mean_cv_score = metric(y_train, oof_inner)
    print(f"80% data CV score: {metric.__name__}: {mean_cv_score:.5f}")
    print(f'{"*" * 50}\n')
    inner_cv_score.append(mean_cv_score)

    preds = np.zeros(len(holdout))
    for model in models_:
        rule_based_preds = np.where(X_valid['DU'] > DU_threshold, 1, 0)
        model_preds = model.predict(X_valid)
        # Apply rule-based approach
        model_preds = np.where(rule_based_preds == 1, 1, model_preds)
        preds += model_preds
    preds = preds / len(models_)
    cv_score = metric(y_valid, preds)
    print(f"20% data CV score: {metric.__name__}: {cv_score:.5f}")
    print(f'{"*" * 50}\n')
    outer_cv_score.append(cv_score)
    models.append(models_)
print(f"80% data average CV score: {metric.__name__}: {np.mean(inner_cv_score):.5f}")
print(f'{"*" * 50}\n')

print(f"20% data average CV score: {metric.__name__}: {np.mean(outer_cv_score):.5f}")
print(f'{"*" * 50}\n')

def predict(X):
    rule_based_preds = np.where(X['DU'] > DU_threshold, 1, 0)
    y = np.zeros(len(X))
    for model_list in models:
        for model in model_list:
            model_preds = model.predict(X)
            # Apply rule-based approach
            model_preds = np.where(rule_based_preds == 1, 1, model_preds)
            y += model_preds
    return y / sum([len(models_) for models_ in models])

predictions = predict(test[feat_cols[:-1]])

test['class_1'] = predictions
test['class_0'] = 1 - predictions

test_2 = pd.read_csv(f"{COMP_PATH}/test.csv")
test['Id'] = test_2['Id']
df_submission = test.loc[:, ['Id', 'class_0', 'class_1']]
df_submission.to_csv('submission.csv', index=False)