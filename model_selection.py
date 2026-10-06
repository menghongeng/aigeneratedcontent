import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_table
# features.py needs to be in the same folder as this file
# dont forget to pip install pandas numpy matplotlib scikit-learn joblib


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# CONFIGURATION - edit file paths as needed - same OUT_DIR as the data processing script
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

OUT_DIR = "C:/Users/U-ser/Downloads/aigeneratedcontent/datasets"
TRAIN_PATH = f"{OUT_DIR}/train.csv"      # made by the data processing script (80% split)
TEST_PATH = f"{OUT_DIR}/test.csv"        # made by the data processing script (20% split)
TEXT_COL = "text"
LABEL_COL = "is_ai_generated"            # 1 = AI, 0 = human (same as the data processing script)
SOURCE_COL = "source"                    # only used to check results per source, never a model feature
RANDOM_STATE = 42                        # same as data processing so results are repeatable
PRECISION_TARGET = 0.90                  # we care most about NOT falsely accusing a human of using AI
N_CLUSTERS = 4                           # picked using the silhouette table printed in the clustering section

os.makedirs(OUT_DIR, exist_ok=True)


def save_plot(name, dpi=150):
    """Saves the current figure into OUT_DIR and closes it."""
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}", dpi=dpi, bbox_inches="tight")
    plt.close()


def make_pipeline(model):
    """Scaler + model in one pipeline so the scaler is only fitted on training rows (no test leakage)."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


#----------------------------------------------------------------------------
# LOAD DATA + BUILD FEATURES
#----------------------------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"train: {train_df.shape} | test: {test_df.shape}")
print(f"train {train_df[LABEL_COL].mean():.1%} AI | test {test_df[LABEL_COL].mean():.1%} AI")
# split was already stratified by source + label in the data processing script, so these should match

# raw text -> feature table (code lives in features.py so it can be reused for prediction later)
train_df = build_feature_table(train_df, TEXT_COL)
test_df = build_feature_table(test_df, TEXT_COL)
train_df.to_csv(f"{OUT_DIR}/train_features.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test_features.csv", index=False)

# uses every column that starts with feature_ so new features get picked up automatically
FEATURES = [c for c in train_df.columns if c.startswith("feature_")]
print("\nFeatures in use:", FEATURES)
print(train_df[FEATURES].describe().round(2))

X_train, y_train = train_df[FEATURES], train_df[LABEL_COL]
X_test, y_test = test_df[FEATURES], test_df[LABEL_COL]


#----------------------------------------------------------------------------
# CLASSIFICATION - compare candidate models on the test set
#----------------------------------------------------------------------------

# logistic regression = simple, interpretable baseline. random forest = can pick up non-linear patterns
# and feature interactions. dummy = always guesses the majority class, so it is the score to beat.
models = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

fitted = {}
print(f"\n{'model':<10} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'roc_auc':>8} {'pr_auc':>7}")
for name, m in models.items():
    pipe = make_pipeline(m).fit(X_train, y_train)
    fitted[name] = pipe
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    print(f"{name:<10} {accuracy_score(y_test, pred):>6.3f} "
          f"{precision_score(y_test, pred, zero_division=0):>6.3f} "
          f"{recall_score(y_test, pred):>6.3f} {f1_score(y_test, pred):>6.3f} "
          f"{roc_auc_score(y_test, proba):>8.3f} {average_precision_score(y_test, proba):>7.3f}")
# accuracy alone hides which mistakes are being made, so precision/recall/F1/AUC are shown too

# confusion matrices - rows are the truth, columns are the prediction
for name in ["logistic", "forest"]:
    pred = fitted[name].predict(X_test)
    print(f"\n--- {name} ---")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["human (0)", "AI (1)"]))

# log loss judges the predicted probabilities, not just the final 0/1 decision
for name in ["logistic", "forest"]:
    print(f"{name} log loss: {log_loss(y_test, fitted[name].predict_proba(X_test)[:, 1]):.4f}")

# class_weight: "balanced" pushes the model to catch more of the smaller class, usually at the cost of precision.
# shown so the choice for the final model is deliberate, not just sklearn's default
print(f"\n{'class_weight':<14} {'precision':>10} {'recall':>8} {'f1':>6}")
for cw in [None, "balanced"]:
    pipe = make_pipeline(LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE))
    pred = pipe.fit(X_train, y_train).predict(X_test)
    print(f"{str(cw):<14} {precision_score(y_test, pred):>10.3f} {recall_score(y_test, pred):>8.3f} "
          f"{f1_score(y_test, pred):>6.3f}")


#----------------------------------------------------------------------------
# CROSS VALIDATION + TUNING - TRAIN SET ONLY so the test set stays untouched until the very end
#----------------------------------------------------------------------------

# a single split is one number. 5-fold CV gives 5, so a gap between models smaller than the
# fold-to-fold spread can be treated as noise rather than a real difference
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
cv_f1 = {}

print(f"\n{'model':<10} {'mean F1':>8}   folds")
for name in ["logistic", "forest"]:
    scores = cross_val_score(make_pipeline(models[name]), X_train, y_train, cv=cv, scoring="f1")
    cv_f1[name] = scores.mean()
    print(f"{name:<10} {scores.mean():.3f} +/- {scores.std():.3f}   {scores.round(3)}")

# grid search for the random forest
grid = {"model__n_estimators": [100, 300], "model__max_depth": [None, 10]}
gs = GridSearchCV(make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
                  grid, cv=cv, scoring="f1", n_jobs=-1)
gs.fit(X_train, y_train)

print("\nGrid search results:")
for params, score in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
    print(f"{score:.3f}   {params}")
print(f"best: {gs.best_params_} | untuned CV F1 {cv_f1['forest']:.3f} -> tuned {gs.best_score_:.3f}")
# if every combination is within the fold-to-fold spread above, tuning didnt really help - say so in the report

cv_f1["forest_tuned"] = gs.best_score_
fitted["forest_tuned"] = gs.best_estimator_

# final model is picked using CV on the train set, NOT the test set
final_name = max(cv_f1, key=cv_f1.get)
final_model = fitted[final_name]
print(f"\nFinal model: {final_name} (CV F1 {cv_f1[final_name]:.3f})")


#----------------------------------------------------------------------------
# DECISION THRESHOLD - predict() is just (probability >= 0.5), but 0.5 isnt automatically the best cut
#----------------------------------------------------------------------------

# the threshold is chosen from out-of-fold predictions on the TRAIN set (never the test set):
# lowest threshold that still reaches PRECISION_TARGET, which keeps recall as high as possible
oof_probs = cross_val_predict(clone(final_model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.arange(0.2, 0.96, 0.01)
oof_precision = np.array([precision_score(y_train, oof_probs >= t, zero_division=0) for t in thresholds])
oof_recall = np.array([recall_score(y_train, oof_probs >= t) for t in thresholds])

reached = np.where(oof_precision >= PRECISION_TARGET)[0]
threshold = float(thresholds[reached[0]]) if len(reached) else 0.5
if not len(reached):
    print(f"No threshold reached precision {PRECISION_TARGET}, falling back to 0.5")
print(f"Chosen threshold: {threshold:.2f}")

plt.plot(thresholds, oof_precision, label="precision (out-of-fold, train)")
plt.plot(thresholds, oof_recall, label="recall (out-of-fold, train)")
plt.axvline(0.5, color="grey", ls="--", label="default 0.5")
plt.axvline(threshold, color="red", ls="--", label="chosen")
plt.xlabel("decision threshold")
plt.legend()
save_plot("threshold_tradeoff.png")

# the test set is scored ONCE here with the chosen model and threshold
test_probs = final_model.predict_proba(X_test)[:, 1]
final_pred = (test_probs >= threshold).astype(int)
print(f"\nFinal test results ({final_name}, threshold {threshold:.2f}):")
print(confusion_matrix(y_test, final_pred))
print(classification_report(y_test, final_pred, target_names=["human (0)", "AI (1)"]))

# accuracy per source: if one source is far better than the others, the model may be
# recognising the source rather than AI-ness (same check as the baseline in data processing)
print("Accuracy per source:")
print((y_test == final_pred).groupby(test_df[SOURCE_COL]).mean().round(3))


#----------------------------------------------------------------------------
# FEATURE IMPORTANCE - which features did the work (importance = useful for splitting, not causation)
#----------------------------------------------------------------------------

forest = fitted["forest"].named_steps["model"]
importances = pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nRandom forest feature importances:")
print(importances.round(3))

importances[::-1].plot.barh()
plt.xlabel("importance")
plt.title("Random forest feature importances")
save_plot("feature_importance.png")

#----------------------------------------------------------------------------
# BEYOND-THE-UNIT MODEL 1: ISOLATION FOREST
#----------------------------------------------------------------------------

# Trained only to use human-written text.
# Anything that is somewhat different during the test will be flagged.
# No scaling is needed here since tree-based splits dont care about feature scale.

from sklearn.ensemble import IsolationForest

human_train = X_train[y_train == 0]

iso = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
iso.fit(human_train)

# IsolationForest outputs 1 = normal (human-like), -1 = anomaly (AI-like).
# Flip it to match our usual 0/1 labels so the same metric functions work.
iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)

print("\n--- Isolation Forest (beyond-the-unit, trained on human text only) ---")
print(confusion_matrix(y_test, iso_pred))
print(classification_report(y_test, iso_pred, target_names=["human (0)", "AI (1)"]))


#----------------------------------------------------------------------------
# BEYOND-THE-UNIT MODEL 2: NAIVE BAYES
#----------------------------------------------------------------------------

# Assumes every feature is independent of the others given the class so that it is not strictly true here, since longer words and lower type-token ratio likely go together.
# This simplification makes it fast with nothing to tune, and gives a probabilistic contrast to the linear and tree models.

from sklearn.naive_bayes import GaussianNB

nb = make_pipeline(GaussianNB())
nb.fit(X_train, y_train)
fitted["naive_bayes"] = nb

nb_pred = nb.predict(X_test)

print("\n--- Naive Bayes (beyond-the-unit) ---")
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred, target_names=["human (0)", "AI (1)"]))

#----------------------------------------------------------------------------
# COMPARE ALL MODELS SIDE BY SIDE
#----------------------------------------------------------------------------

log_pred = fitted["logistic"].predict(X_test)
forest_pred = fitted["forest"].predict(X_test)

print(f"\n{'model':<20} {'precision':>10} {'recall':>8} {'f1':>6}")
print(f"{'logistic (taught)':<20} "
      f"{precision_score(y_test, log_pred, zero_division=0):>10.3f} "
      f"{recall_score(y_test, log_pred):>8.3f} "
      f"{f1_score(y_test, log_pred):>6.3f}")
print(f"{'forest (taught)':<20} "
      f"{precision_score(y_test, forest_pred, zero_division=0):>10.3f} "
      f"{recall_score(y_test, forest_pred):>8.3f} "
      f"{f1_score(y_test, forest_pred):>6.3f}")
print(f"{'isolation forest':<20} "
      f"{precision_score(y_test, iso_pred, zero_division=0):>10.3f} "
      f"{recall_score(y_test, iso_pred):>8.3f} "
      f"{f1_score(y_test, iso_pred):>6.3f}")
print(f"{'naive bayes':<20} "
      f"{precision_score(y_test, nb_pred, zero_division=0):>10.3f} "
      f"{recall_score(y_test, nb_pred):>8.3f} "
      f"{f1_score(y_test, nb_pred):>6.3f}")


#----------------------------------------------------------------------------
# CLUSTERING - KMeans inside the AI class only, WITHOUT using the label
#----------------------------------------------------------------------------

# every row here is already label 1 and only feature columns go into KMeans, so the label plays no part.
# clustering is unsupervised so train + test are pooled
all_df = pd.concat([train_df, test_df], ignore_index=True)
ai_df = all_df[all_df[LABEL_COL] == 1].copy().reset_index(drop=True)
# cap extreme values at the 1st/99th percentile so a few broken texts (e.g. 700-word "sentences") dont take over a cluster
X_ai = ai_df[FEATURES].clip(ai_df[FEATURES].quantile(0.01), ai_df[FEATURES].quantile(0.99), axis=1)
X_ai_scaled = StandardScaler().fit_transform(X_ai)

# silhouette score for different k (higher = tighter, better separated clusters) to justify N_CLUSTERS
print("\nSilhouette score by k:")
for k in range(2, 7):
    labels_k = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_ai_scaled)
    sil = silhouette_score(X_ai_scaled, labels_k, sample_size=min(5000, len(ai_df)), random_state=RANDOM_STATE)
    print(f"k={k}: {sil:.3f}")

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
ai_df["cluster"] = kmeans.fit_predict(X_ai_scaled)
print("\nCluster sizes:")
print(ai_df["cluster"].value_counts().sort_index())

# composition checks: label will be 100% AI (we only clustered the AI class), so source is also shown -
# KMeans never saw it, so it shows whether the clusters line up with anything real
print("\nLabel composition per cluster (share that is AI):")
print(ai_df.groupby("cluster")[LABEL_COL].mean())
print("\nSource composition per cluster (top 3):")
print(ai_df.groupby("cluster")[SOURCE_COL].value_counts(normalize=True).groupby(level=0).head(3).round(3))

# composition doesnt say what a cluster IS, so each cluster is described by how its features differ from the
# AI-class average. values are z-scores (standard deviations from the average) so features with big raw units
# like text length dont drown out the others
z_means = pd.DataFrame(X_ai_scaled, columns=FEATURES).groupby(ai_df["cluster"]).mean()
raw_means = ai_df.groupby("cluster")[FEATURES].mean()
overall_means = ai_df[FEATURES].mean()
dist_to_centroids = kmeans.transform(X_ai_scaled)

for c in sorted(ai_df["cluster"].unique()):
    print(f"\nCluster {c} (n={(ai_df['cluster'] == c).sum()})")
    top3 = z_means.loc[c].sort_values(key=abs, ascending=False).head(3)
    for feat, z in top3.items():
        print(f"  {feat}: z={z:+.2f} (cluster mean {raw_means.loc[c, feat]:.2f} vs overall {overall_means[feat]:.2f})")

    # real examples = the 2 texts closest to this cluster's centre
    members = ai_df.index[ai_df["cluster"] == c]
    for i in members[np.argsort(dist_to_centroids[members, c])[:2]]:
        print(f"  example: {str(ai_df.loc[i, TEXT_COL])[:300]!r}")
# use the z-scores + examples above to write what each cluster actually is in the report,
# including anything that is notably MISSING (e.g. very low sentence length variety)


#----------------------------------------------------------------------------
# SAVE THE TRAINED MODEL - so predict.py (and later the web app) can load it without retraining
#----------------------------------------------------------------------------

# the whole Pipeline is saved so the scaler travels with the model (saving only the bare model means
# redoing the exact scaling by hand later). the threshold and feature names are saved with it because
# predict() on its own would use 0.5 and the column order has to match
MODEL_PATH = f"{OUT_DIR}/ai_detector.joblib"
joblib.dump({"pipeline": final_model, "threshold": threshold, "features": FEATURES}, MODEL_PATH)
print(f"\nsaved {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024:.0f} KB)")
# a model saved under one scikit-learn version may not load under another, so pin the version in the readme