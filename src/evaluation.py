"""Cross-validated child vs adult classification, baselines and permutation test."""

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(C=1.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def regress_out(X_train, X_test, c_train, c_test):
    """Remove a confound from every feature. Fit on train, apply to both."""
    c_train, c_test = np.asarray(c_train).reshape(-1, 1), np.asarray(c_test).reshape(-1, 1)
    reg = LinearRegression().fit(c_train, X_train)
    return X_train - reg.predict(c_train), X_test - reg.predict(c_test)


def cross_validate(time_series, y, kind="correlation", n_splits=5, n_repeats=10, confound=None):
    """Repeated stratified k-fold. The connectivity measure (needed for tangent), the
    confound regression and the classifier are all fit on the training fold only.

    Returns per-fold scores, the summed confusion matrix, each person's out-of-fold
    P(adult) averaged over repeats, and the model weights from every fold.
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
    scores = []
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        scores += list(cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy"))
    return np.array(scores)


def permutation_test(X, y, n_permutations=1000, seed=0):
    """Only for features computed per person (e.g. correlation), so precomputing X doesn't leak."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return permutation_test_score(make_classifier(), X, y, cv=cv, scoring="balanced_accuracy",
                                  n_permutations=n_permutations, random_state=seed, n_jobs=-1)


def top_k_frequency(weights, k=20):
    """Share of folds in which each feature is among the k largest |weights|."""
    counts = np.zeros(weights.shape[1])
    for fold_top in np.argsort(-np.abs(weights), axis=1)[:, :k]:
        counts[fold_top] += 1
    return counts / len(weights)
