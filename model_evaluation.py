import json
import os
import re
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_table


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datasets"
OUTPUT_DIR = PROJECT_DIR / "evaluation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_PATH = DATA_DIR / "base_dataset.csv"
SECONDARY_PATH = DATA_DIR / "secondary_dataset.xlsx"
COMPARISON_PATH = DATA_DIR / "comparison_dataset.jsonl"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = DATA_DIR / "ai_detector.joblib"

TEXT_COL = "text"
LABEL_COL = "is_ai_generated"
PROMPT_COL = "prompt_name"
SOURCE_COL = "source"
TARGET_COLUMNS = [TEXT_COL, LABEL_COL, PROMPT_COL, SOURCE_COL]

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_WORDS = 20
PRECISION_TARGET = 0.90
ERROR_EXAMPLES_PER_TYPE = 5


# -----------------------------------------------------------------------------
# SMALL HELPERS
# -----------------------------------------------------------------------------

def save_current_plot(filename):
    """Save the current matplotlib figure into evaluation_outputs."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def make_pipeline(model):
    """Scale the feature columns before fitting the classifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def clean_text(text):
    """Use the same text-cleaning idea as datatransformation.py."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(
        str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_json_any(path):
    """Read either a JSON array or JSON-lines file."""
    with open(path, encoding="utf-8") as file:
        first_char = file.read(1)
        file.seek(0)
        if first_char == "[":
            return pd.DataFrame(json.load(file))
    return pd.read_json(path, lines=True)


def check_raw_files():
    """Explain exactly what is missing if the raw project data is incomplete."""
    missing = [p for p in [BASE_PATH, SECONDARY_PATH, COMPARISON_PATH] if not p.exists()]
    if missing:
        message = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "The evaluation script could not find the raw project datasets:\n"
            f"{message}\n\n"
            "Put model_evaluation.py in the MAIN aigeneratedcontent folder, not inside datasets."
        )


# -----------------------------------------------------------------------------
# BUILD THE CLEAN COMBINED DATASET FROM THE PROJECT'S THREE RAW DATASETS
# -----------------------------------------------------------------------------

def load_and_prepare_combined_data():
    check_raw_files()

    base = pd.read_csv(BASE_PATH)
    secondary = pd.read_excel(SECONDARY_PATH)
    comparison = load_json_any(COMPARISON_PATH)

    # Base dataset: label -> is_ai_generated
    base = base.rename(columns={"label": LABEL_COL})
    base = base[TARGET_COLUMNS].copy()

    # Secondary dataset: project code defines label_id 1 as human and 0 as AI,
    # so flip it to keep 0 = human, 1 = AI.
    secondary[LABEL_COL] = 1 - secondary["label_id"]
    secondary[PROMPT_COL] = "no_prompt"
    secondary = secondary[TARGET_COLUMNS].copy()

    # Comparison dataset: each question contains a list of human answers and ChatGPT answers.
    def explode_answers(df, answer_column, label):
        out = df[["question", answer_column]].explode(answer_column)
        out = out.rename(columns={answer_column: TEXT_COL, "question": PROMPT_COL})
        out[LABEL_COL] = label
        return out

    comparison_long = pd.concat(
        [
            explode_answers(comparison, "human_answers", 0),
            explode_answers(comparison, "chatgpt_answers", 1),
        ],
        ignore_index=True,
    )
    comparison_long[SOURCE_COL] = "wiki_questions"
    comparison_long = comparison_long[TARGET_COLUMNS]

    combined = pd.concat([base, secondary, comparison_long], ignore_index=True)

    # Clean and remove unusable or contradictory rows.
    combined = combined.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    combined[TEXT_COL] = combined[TEXT_COL].map(clean_text)
    combined = combined[combined[TEXT_COL] != ""].copy()

    labels_per_text = combined.groupby(TEXT_COL)[LABEL_COL].nunique()
    contradictory_texts = labels_per_text[labels_per_text > 1].index
    combined = combined[~combined[TEXT_COL].isin(contradictory_texts)].copy()

    combined = combined.drop_duplicates(subset=TEXT_COL, keep="first").copy()
    word_count = combined[TEXT_COL].str.split().str.len()
    combined = combined[word_count >= MIN_WORDS].copy()

    combined[LABEL_COL] = combined[LABEL_COL].astype(int)
    combined[PROMPT_COL] = combined[PROMPT_COL].fillna("unknown_prompt").astype(str)
    combined[SOURCE_COL] = combined[SOURCE_COL].fillna("unknown_source").astype(str)
    combined = combined.reset_index(drop=True)

    # Stable row id is useful when constructing hard tests without leaking rows into training.
    combined["_row_id"] = np.arange(len(combined))
    return combined


def get_standard_split(combined):
    """Use existing train/test files if present; otherwise reproduce the project's 80/20 split."""
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        print("Using existing datasets/train.csv and datasets/test.csv")
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        return train_df, test_df

    print("train.csv/test.csv were not found - creating the 80/20 split in memory.")
    strata = combined[SOURCE_COL].astype(str) + "_" + combined[LABEL_COL].astype(str)
    train_df, test_df = train_test_split(
        combined[TARGET_COLUMNS],
        test_size=TEST_SIZE,
        stratify=strata,
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# MODEL PREPARATION
# -----------------------------------------------------------------------------

def choose_and_train_model(train_feature_df, features):
    """
    Train the same two main classifier families used by model_selection.py and
    select between them using 5-fold training-set F1.
    """
    X_train = train_feature_df[features]
    y_train = train_feature_df[LABEL_COL]

    candidates = {
        "logistic": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "forest": make_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        ),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}

    print("\nSelecting an evaluation model using 5-fold CV on TRAIN only:")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"  {name:<9} mean F1 = {scores.mean():.3f} +/- {scores.std():.3f}")

    best_name = max(cv_scores, key=cv_scores.get)
    final_model = candidates[best_name].fit(X_train, y_train)

    # Choose a decision threshold using out-of-fold TRAIN probabilities only.
    oof_probs = cross_val_predict(
        clone(candidates[best_name]),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.arange(0.20, 0.96, 0.01)
    precisions = np.array([
        precision_score(y_train, oof_probs >= t, zero_division=0) for t in thresholds
    ])
    reached = np.where(precisions >= PRECISION_TARGET)[0]
    threshold = float(thresholds[reached[0]]) if len(reached) else 0.50

    print(f"Chosen model: {best_name}")
    print(f"Chosen threshold: {threshold:.2f}")
    return final_model, threshold, best_name


def get_model(train_feature_df, features):
    """Load the project's saved final model if available; otherwise train one for evaluation."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle["pipeline"]
            threshold = float(bundle.get("threshold", 0.5))
            saved_features = bundle.get("features", features)

            if list(saved_features) == list(features):
                print(f"Loaded existing model: {MODEL_PATH.name}")
                return model, threshold, "saved_final_model"

            print("Saved model uses different feature columns, so a fresh evaluation model will be trained.")
        except Exception as exc:
            print(f"Could not load ai_detector.joblib ({exc}). A fresh evaluation model will be trained.")

    return choose_and_train_model(train_feature_df, features)


# -----------------------------------------------------------------------------
# METRICS + ERROR ANALYSIS
# -----------------------------------------------------------------------------

def calculate_metrics(y_true, probabilities, predictions, evaluation_name):
    """Return the main classification metrics in one row."""
    row = {
        "evaluation": evaluation_name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }

    # AUC needs both classes in the evaluation set.
    if pd.Series(y_true).nunique() == 2:
        row["roc_auc"] = roc_auc_score(y_true, probabilities)
        row["pr_auc"] = average_precision_score(y_true, probabilities)
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan

    return row


def save_confusion_matrix(y_true, predictions, title, filename):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Human", "AI"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(title)
    save_current_plot(filename)


def collect_error_examples(df, probabilities, predictions, evaluation_name):
    """Collect a few confident false positives and false negatives for manual inspection."""
    results = df[[TEXT_COL, LABEL_COL, PROMPT_COL, SOURCE_COL]].copy().reset_index(drop=True)
    results["ai_probability"] = probabilities
    results["predicted_label"] = predictions
    results["evaluation"] = evaluation_name

    false_positives = results[
        (results[LABEL_COL] == 0) & (results["predicted_label"] == 1)
    ].sort_values("ai_probability", ascending=False).head(ERROR_EXAMPLES_PER_TYPE)
    false_positives = false_positives.copy()
    false_positives["error_type"] = "false_positive"

    false_negatives = results[
        (results[LABEL_COL] == 1) & (results["predicted_label"] == 0)
    ].sort_values("ai_probability", ascending=True).head(ERROR_EXAMPLES_PER_TYPE)
    false_negatives = false_negatives.copy()
    false_negatives["error_type"] = "false_negative"

    return pd.concat([false_positives, false_negatives], ignore_index=True)


def evaluate_model(model, threshold, feature_df, features, name, file_prefix):
    X = feature_df[features]
    y = feature_df[LABEL_COL]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(confusion_matrix(y, predictions))
    print(classification_report(y, predictions, target_names=["human (0)", "AI (1)"], zero_division=0))

    metrics = calculate_metrics(y, probabilities, predictions, name)
    save_confusion_matrix(y, predictions, name, f"{file_prefix}_confusion_matrix.png")
    errors = collect_error_examples(feature_df, probabilities, predictions, name)
    return metrics, errors


# -----------------------------------------------------------------------------
# INNOVATION TEST 1: UNSEEN AI SOURCE
# -----------------------------------------------------------------------------

def build_unseen_source_test(combined):
    """
    Hold out one whole AI source from training, then pair it with unseen human
    examples. This tests whether the detector generalises to an AI generator it
    never encountered during training.
    """
    source_stats = combined.groupby(SOURCE_COL)[LABEL_COL].agg(["count", "mean"])
    ai_only_sources = source_stats[(source_stats["mean"] == 1.0) & (source_stats["count"] >= 100)]

    if ai_only_sources.empty:
        return None, None, None

    held_source = ai_only_sources.sort_values("count", ascending=False).index[0]
    ai_test = combined[combined[SOURCE_COL] == held_source].copy()

    human_pool = combined[combined[LABEL_COL] == 0].copy()
    human_test = human_pool.sample(
        n=min(len(ai_test), len(human_pool)),
        random_state=RANDOM_STATE,
    )

    hard_test = pd.concat([ai_test, human_test], ignore_index=True)
    held_ids = set(ai_test["_row_id"]) | set(human_test["_row_id"])
    hard_train = combined[~combined["_row_id"].isin(held_ids)].copy()

    return hard_train, hard_test, held_source


# -----------------------------------------------------------------------------
# INNOVATION TEST 2: UNSEEN PROMPT/TOPIC
# -----------------------------------------------------------------------------

def build_unseen_prompt_test(combined):
    """
    Hold out one complete prompt that contains both human and AI writing.
    This tests topic generalisation rather than a random-row split.
    """
    prompt_stats = combined.groupby(PROMPT_COL)[LABEL_COL].agg(["count", "nunique"])
    valid = prompt_stats[(prompt_stats["nunique"] == 2) & (prompt_stats["count"] >= 100)]

    if valid.empty:
        return None, None, None

    held_prompt = valid.sort_values("count", ascending=False).index[0]
    hard_test = combined[combined[PROMPT_COL] == held_prompt].copy()
    hard_train = combined[combined[PROMPT_COL] != held_prompt].copy()
    return hard_train, hard_test, held_prompt


def train_hard_test_model(train_df, features):
    """Train a cloneable classifier for a hard test without touching the held-out rows."""
    featured = build_feature_table(train_df[TARGET_COLUMNS], TEXT_COL)
    X_train = featured[features]
    y_train = featured[LABEL_COL]

    # Random forest matches one of the project's main models and handles the small
    # hand-engineered feature set well without needing extra feature extraction.
    model = make_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    )
    model.fit(X_train, y_train)

    # Use a neutral threshold for hard tests so the comparison is easy to interpret.
    return model, 0.50


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("Project folder:", PROJECT_DIR)
    print("Datasets folder:", DATA_DIR)
    print("Outputs folder:", OUTPUT_DIR)

    combined = load_and_prepare_combined_data()
    print(f"\nPrepared combined dataset: {len(combined):,} rows")
    print(combined[LABEL_COL].value_counts().rename(index={0: "human", 1: "AI"}).to_string())

    # ---------------- Standard evaluation ----------------
    train_df, test_df = get_standard_split(combined)
    train_features = build_feature_table(train_df, TEXT_COL)
    test_features = build_feature_table(test_df, TEXT_COL)

    features = [c for c in train_features.columns if c.startswith("feature_")]
    if not features:
        raise RuntimeError("No feature_* columns were produced by features.py")

    print("\nFeatures used:")
    for feature in features:
        print("  -", feature)

    model, threshold, model_name = get_model(train_features, features)
    print(f"Evaluation model: {model_name}")

    all_metrics = []
    all_errors = []

    metrics, errors = evaluate_model(
        model,
        threshold,
        test_features,
        features,
        "Standard 80/20 Test",
        "standard_test",
    )
    all_metrics.append(metrics)
    all_errors.append(errors)

    # Save standard per-source accuracy because source imbalance can reveal weaknesses.
    standard_probs = model.predict_proba(test_features[features])[:, 1]
    standard_pred = (standard_probs >= threshold).astype(int)
    source_results = test_features[[SOURCE_COL, LABEL_COL]].copy()
    source_results["prediction"] = standard_pred
    source_accuracy = (
        source_results.assign(correct=source_results[LABEL_COL] == source_results["prediction"])
        .groupby(SOURCE_COL)["correct"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "accuracy"})
        .sort_values("accuracy")
    )
    source_accuracy.to_csv(OUTPUT_DIR / "accuracy_by_source.csv")

    # ---------------- Hard test: unseen AI source ----------------
    source_train, source_test, held_source = build_unseen_source_test(combined)
    if source_train is not None:
        print(f"\nHard test 1 held-out AI source: {held_source}")
        hard_model, hard_threshold = train_hard_test_model(source_train, features)
        source_test_features = build_feature_table(source_test[TARGET_COLUMNS], TEXT_COL)
        metrics, errors = evaluate_model(
            hard_model,
            hard_threshold,
            source_test_features,
            features,
            f"Unseen AI Source: {held_source}",
            "unseen_source",
        )
        all_metrics.append(metrics)
        all_errors.append(errors)
    else:
        print("\nSkipped unseen-source test: no suitable AI-only source was found.")

    # ---------------- Hard test: unseen prompt/topic ----------------
    prompt_train, prompt_test, held_prompt = build_unseen_prompt_test(combined)
    if prompt_train is not None:
        print(f"\nHard test 2 held-out prompt: {held_prompt}")
        hard_model, hard_threshold = train_hard_test_model(prompt_train, features)
        prompt_test_features = build_feature_table(prompt_test[TARGET_COLUMNS], TEXT_COL)
        metrics, errors = evaluate_model(
            hard_model,
            hard_threshold,
            prompt_test_features,
            features,
            f"Unseen Prompt: {held_prompt}",
            "unseen_prompt",
        )
        all_metrics.append(metrics)
        all_errors.append(errors)
    else:
        print("\nSkipped unseen-prompt test: no suitable prompt with both classes was found.")

    # ---------------- Save comparison results ----------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

    if all_errors:
        error_df = pd.concat(all_errors, ignore_index=True)
        error_df.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)

    # Metric comparison chart.
    chart_cols = ["accuracy", "precision", "recall", "f1"]
    plot_df = metrics_df.set_index("evaluation")[chart_cols]
    ax = plot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Standard vs Hard-Test Model Performance")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    save_current_plot("evaluation_comparison.png")

    # Small summary text file for quick inspection.
    with open(OUTPUT_DIR / "evaluation_summary.txt", "w", encoding="utf-8") as file:
        file.write("MODEL EVALUATION SUMMARY\n")
        file.write("=" * 60 + "\n")
        file.write(f"Features: {features}\n")
        file.write(f"Standard evaluation model: {model_name}\n")
        file.write(f"Standard threshold: {threshold:.2f}\n")
        if held_source is not None:
            file.write(f"Held-out AI source: {held_source}\n")
        if held_prompt is not None:
            file.write(f"Held-out prompt: {held_prompt}\n")
        file.write("\nMetrics:\n")
        file.write(metrics_df.round(4).to_string(index=False))
        file.write("\n\nManual error analysis:\n")
        file.write(
            "Open error_analysis.csv and inspect the selected false positives and false negatives. "
            "Look for repeated writing characteristics, source patterns, prompt/topic effects, text length, "
            "sentence structure, vocabulary variety, or other traits shared by the mistakes.\n"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Results saved in:", OUTPUT_DIR)
    print("  - evaluation_metrics.csv")
    print("  - evaluation_comparison.png")
    print("  - standard_test_confusion_matrix.png")
    if source_train is not None:
        print("  - unseen_source_confusion_matrix.png")
    if prompt_train is not None:
        print("  - unseen_prompt_confusion_matrix.png")
    print("  - accuracy_by_source.csv")
    print("  - error_analysis.csv")
    print("  - evaluation_summary.txt")


if __name__ == "__main__":
    main()
