import json
import os
import re
import unicodedata
from config import BASE_PATH, SECONDARY_PATH, COMPARISON_PATH, COMBINED_CLEAN_PATH, TRAIN_PATH, TEST_PATH
import pandas as pd
from sklearn.model_selection import train_test_split
# dont forget to pip install pandas matplotlib seaborn scikit-learn openpyxl


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# CONFIGURATION - edit file paths as needed - maybe use a local version so we dont keep pushing/pulling different paths to git
# need to make config file for paths and other constants so that they can be easily changed without editing the code
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
DS3_SOURCE_NAME = "wiki_questions"
MIN_WORDS = 20
RANDOM_STATE = 42
TARGET_COLUMNS = ["text", "is_ai_generated", "prompt_name", "source"]
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
ds2 = pd.read_excel(SECONDARY_PATH)
ds3 = load_json_any(COMPARISON_PATH)

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


# clean dataset by fixing formatting issues (odd spaces, extra blank lines)
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



print("\nMedian words by label:")
print(combined.groupby("is_ai_generated")["word_count"].median())

combined = combined[TARGET_COLUMNS]
combined.to_csv({COMBINED_CLEAN_PATH}, index=False)

# Partitioning data into training and testing sets. 80/20 split.
strata = combined["source"] + "_" + combined["is_ai_generated"].astype(str)
train, test = train_test_split(combined, test_size=0.2, stratify=strata,
                               random_state=RANDOM_STATE)
train.to_csv(f"{TRAIN_PATH}", index=False)
test.to_csv(f"{TEST_PATH}", index=False)
print(f"\nTrain: {len(train)} rows | Test: {len(test)} rows")

