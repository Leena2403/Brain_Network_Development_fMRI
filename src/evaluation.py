"""
Evaluation: train a child-vs-adult classifier on connectivity features and
measure how well it works on people it never saw.

Used by notebooks 03 and 04.
"""

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(C=1.0):
    """Scale each feature, then logistic regression.
    class_weight='balanced' because there are ~3 children for every adult."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def regress_out(X_train, X_test, c_train, c_test):
    """Remove the part of each feature that a confound (e.g. head motion) can explain.
    The fit uses training people only, then the same correction is applied to test people."""
    c_train, c_test = np.asarray(c_train).reshape(-1, 1), np.asarray(c_test).reshape(-1, 1)
    reg = LinearRegression().fit(c_train, X_train)
    return X_train - reg.predict(c_train), X_test - reg.predict(c_test)


def cross_validate(time_series, y, kind="correlation", n_splits=5, n_repeats=10, confound=None):
    """Repeated stratified k-fold cross-validation.

    Everything that learns from data (the connectivity measure, the scaler, the
    confound regression and the classifier) is fitted on the training people only.

    Returns a dict with:
      scores      balanced accuracy for each fold (n_splits * n_repeats values)
      confusion   summed confusion matrix (rows = true class, columns = predicted)
      prob_adult  each person's out-of-fold P(adult), averaged over repeats
      weights     model weights from each fold (n_folds x n_connections)
    """
    y = np.asarray(y)
    scores, weights = [], []
    cm = np.zeros((2, 2))
    prob = np.zeros((len(y), n_repeats))

    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for train, test in cv.split(np.zeros(len(y)), y):
            conn = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True,
                                       standardize="zscore_sample")
            X_train = conn.fit_transform([time_series[i] for i in train])
            X_test = conn.transform([time_series[i] for i in test])

            if confound is not None:
                X_train, X_test = regress_out(X_train, X_test, confound[train], confound[test])

            model = make_classifier().fit(X_train, y[train])
            pred = model.predict(X_test)

            scores.append(balanced_accuracy_score(y[test], pred))
            cm += confusion_matrix(y[test], pred, labels=[0, 1])
            prob[test, rep] = model.predict_proba(X_test)[:, 1]
            weights.append(model[-1].coef_.ravel())

    return {"scores": np.array(scores), "confusion": cm,
            "prob_adult": prob.mean(axis=1), "weights": np.array(weights)}


def baseline_scores(X, y, model, n_splits=5, n_repeats=10):
    """Same repeated CV for a simple model on simple features (e.g. head motion only)."""
    scores = []
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        scores += list(cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy"))
    return np.array(scores)


def permutation_test(X, y, n_permutations=1000, seed=0):
    """How often does a model trained on shuffled labels do as well as the real one?

    Only valid for features computed per person (e.g. correlation), where
    computing them before the split leaks nothing between people.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    score, perm_scores, p = permutation_test_score(
        make_classifier(), X, y, cv=cv, scoring="balanced_accuracy",
        n_permutations=n_permutations, random_state=seed, n_jobs=-1)
    return score, perm_scores, p


def top_k_frequency(weights, k=20):
    """For each connection: share of CV folds in which it was among the k largest |weights|.
    A value near 1 means the model relies on that connection whichever people it trains on."""
    ranks = np.argsort(-np.abs(weights), axis=1)[:, :k]
    counts = np.zeros(weights.shape[1])
    for fold_top in ranks:
        counts[fold_top] += 1
    return counts / len(weights)
