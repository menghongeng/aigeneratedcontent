import json
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
# dont forget to pip install pandas matplotlib seaborn scikit-learn openpyxl


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# CONFIGURATION - edit file paths as needed - maybe use a local version so we dont keep pushing/pulling different paths to git
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

BASE_PATH = "C:/Users/arion/OneDrive - Swinburne University/Year 2 Sem 2/Tech Innovation/Assignment2/datasets/base_dataset.csv"
DS2_PATH = "C:/Users/arion/OneDrive - Swinburne University/Year 2 Sem 2/Tech Innovation/Assignment2/datasets/secondary_dataset.xlsx"
DS3_PATH = "C:/Users/arion/OneDrive - Swinburne University/Year 2 Sem 2/Tech Innovation/Assignment2/datasets/comparison_dataset.jsonl"
OUT_DIR = "C:/Users/arion/OneDrive - Swinburne University/Year 2 Sem 2/Tech Innovation/Assignment2/datasets"
DS3_SOURCE_NAME = "wiki_questions"     # renames sources in 3rd dataset to wiki_questions (the style of questions asked)
MIN_WORDS = 20                     # texts shorter than this are dropped
RANDOM_STATE = 42                  # makes the train/test split repeatable
TARGET_COLUMNS = ["text", "is_ai_generated", "prompt_name", "source",]  # columns from base dataset that are kept


os.makedirs(OUT_DIR, exist_ok=True)
steps = []  # a running log of rows kept after each step


def log_step(description, df):
    steps.append({"step": description, "rows": len(df)})


# Loading JSON
def load_json_any(path):
    """Reads either a normal JSON array or 'JSON Lines' (one object per line)."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return pd.DataFrame(json.load(f))
    return pd.read_json(path, lines=True)

# defining the types of files for each dataset
base = pd.read_csv(BASE_PATH)
ds2 = pd.read_excel(DS2_PATH)
ds3 = load_json_any(DS3_PATH)

# prints the amount of rows and columns for each dataset and total of missing values for each column
for name, d in [("base", base), ("dataset 2", ds2), ("dataset 3", ds3)]:
    print(f"\n{name}: {d.shape[0]} rows x {d.shape[1]} columns")
    print(d.isna().sum().to_string())

# load base dataset but rename label column to make content clearer
base = base.rename(columns={"label": "is_ai_generated"})
base = base[TARGET_COLUMNS].copy()

# loading dataset 2 and flipping label column values to match the base dataset (0 = human, 1 = AI)
print("\nDataset 2 label check (before flipping):")
print(pd.crosstab(ds2["label_name"], ds2["label_id"]))
assert set(ds2["label_id"].dropna().unique()) <= {0, 1}, "label_id should only be 0 or 1"
ds2["is_ai_generated"] = 1 - ds2["label_id"]
ds2["prompt_name"] = "no_prompt"   # these texts weren't written from a known essay prompt
ds2 = ds2[TARGET_COLUMNS]        # drops sr.no (just a row counter) and label_name/label_id (redundant now)

# load dataset 3 and explode the lists of answers into one row per answer, then combine them into a single long dataset
def explode_answers(df, answer_col, label):
    out = df[["question", answer_col]].explode(answer_col)  # 1 row per list item
    out = out.rename(columns={answer_col: "text", "question": "prompt_name"})
    out["is_ai_generated"] = label
    return out

# defines text with "human_answers" as label 0 and opposite for "chatgpt_answers"
ds3_long = pd.concat(
    [explode_answers(ds3, "human_answers", 0), explode_answers(ds3, "chatgpt_answers", 1)],
    ignore_index=True,
)
ds3_long["source"] = DS3_SOURCE_NAME
ds3_long = ds3_long[TARGET_COLUMNS]

# combines all three datasets into one, ignoring the original row indices
combined = pd.concat([base, ds2, ds3_long], ignore_index=True)
log_step("combined (raw)", combined)

# Drop rows with no text or no label because the model has no use for them
combined = combined.dropna(subset=["text", "is_ai_generated"])
log_step("dropped missing text/label", combined)


# clean dataset by fixing formatting issues (encodings, odd spaces, extra blank lines) but keeping punctuation and capitalisation for AI detection
def clean_text(t):
    t = unicodedata.normalize("NFKC", str(t))            # unify odd unicode forms (ligatures, full-width chars)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)     # invisible zero-width characters
    t = t.replace("\u00a0", " ")                         # non-breaking space -> normal space
    t = t.replace("\r\n", "\n").replace("\r", "\n")      # unify line endings
    t = t.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}))  # curly -> straight quotes
    t = re.sub(r"[ \t]+", " ", t)                        # runs of spaces/tabs -> one space
    t = re.sub(r" *\n *", "\n", t)                       # trailing/leading spaces around line breaks
    t = re.sub(r"\n{3,}", "\n\n", t)                     # 3+ blank lines -> one blank line
    return t.strip()


combined["text"] = combined["text"].map(clean_text)
combined = combined[combined["text"] != ""]
log_step("removed empty texts after cleaning", combined)

# removes any rows that has duplicate text but both a 0 and 1 under labels (contradictory)
labels_per_text = combined.groupby("text")["is_ai_generated"].nunique()
contradictory = labels_per_text[labels_per_text > 1].index
combined = combined[~combined["text"].isin(contradictory)]
log_step("removed texts with conflicting labels", combined)

# removes any rows that are exact duplicates of another row (same text, same label)
combined = combined.drop_duplicates(subset="text", keep="first")
log_step("removed exact duplicates", combined)

# removes any rows that are too short to be useful for AI detection (less than MIN_WORDS words)
combined["word_count"] = combined["text"].str.split().str.len()
combined = combined[combined["word_count"] >= MIN_WORDS]
log_step(f"removed texts under {MIN_WORDS} words", combined)

combined["is_ai_generated"] = combined["is_ai_generated"].astype(int)
combined = combined.reset_index(drop=True)

print("\nRows kept after each step:")
print(pd.DataFrame(steps).to_string(index=False))

# -----------------
# -----------------
# Graphs created by AI, remove later and recreate our own if needed
# -----------------
# -----------------
print("\nSource x label counts (0 = human, 1 = AI):")
print(pd.crosstab(combined["source"], combined["is_ai_generated"]))

sns.set_theme()

# Class balance per source. If a source has ONE class only, the model can cheat by
# learning the source's style instead of "AI vs human".
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=combined, y="source", hue="is_ai_generated", ax=ax)
ax.set_title("Samples per source and label (0 = human, 1 = AI)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_balance_by_source.png", dpi=150)
plt.close()

# Length by label. A big gap means the model could just learn "long = human".
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=combined, x="word_count", hue="is_ai_generated", log_scale=True,
             bins=50, element="step", ax=ax)
ax.set_title("Word count by label (log scale)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/word_count_by_label.png", dpi=150)
plt.close()

print("\nMedian words by label:")
print(combined.groupby("is_ai_generated")["word_count"].median())

combined = combined[TARGET_COLUMNS]
combined.to_csv(f"{OUT_DIR}/combined_clean.csv", index=False)

# Partitioning data into training and testing sets. 80/20 split.
strata = combined["source"] + "_" + combined["is_ai_generated"].astype(str)
train, test = train_test_split(combined, test_size=0.2, stratify=strata,
                               random_state=RANDOM_STATE)
train.to_csv(f"{OUT_DIR}/train.csv", index=False)
test.to_csv(f"{OUT_DIR}/test.csv", index=False)
print(f"\nTrain: {len(train)} rows | Test: {len(test)} rows")

# ------------------
# ------------------
# Basic AI Model made by AI, remove later and replace with our own
# ------------------
# ------------------

# TF-IDF turns each text into numbers (how important each word/word-pair is).
# Logistic regression then learns which words push towards "AI" or "human".
# It is fitted on TRAIN ONLY, then applied to TEST, so nothing leaks between them.
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

model = LogisticRegression(max_iter=1000, class_weight="balanced")  # balanced = fair to the smaller class
model.fit(X_train, train["is_ai_generated"])
pred = model.predict(X_test)

print("\nBaseline results on the test set:")
print(classification_report(test["is_ai_generated"], pred, target_names=["human (0)", "AI (1)"]))

ConfusionMatrixDisplay.from_predictions(test["is_ai_generated"], pred,
                                        display_labels=["human", "AI"])
plt.title("Baseline confusion matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/baseline_confusion_matrix.png", dpi=150)
plt.close()

# Accuracy per source: if one source is far better than the others, the model may be
# recognising the source rather than AI-ness.
print("Accuracy per source:")
print((test["is_ai_generated"] == pred).groupby(test["source"]).mean().round(3))