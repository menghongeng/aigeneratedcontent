import numpy as np
import pandas as pd

# features.py - turns raw text into a table of numeric features.
# kept in its own file so the same code can be used for training AND for predicting on new text later.
# every column starts with "feature_" so model_selection.py picks them up automatically.


def compute_features(text, ttr_window=200):
    """Basic stylometric features for ONE text.

    - text length: raw character count
    - avg sentence length: words per sentence, averaged
    - avg word length: characters per word, averaged
    - type-token ratio (TTR): unique words / total words over a FIXED window
    - sentence length variation: standard deviation of sentence lengths
    - punctuation density: punctuation relative to text length
    - vocabulary repetition: proportion of repeated words
    """
    text = str(text)
    words = text.split()

    # split on . ! ? to get rough sentences
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    word_lengths = [len(w) for w in words] if words else [0]

    ttr_words = words[:ttr_window]
    ttr = len(set(ttr_words)) / len(ttr_words) if ttr_words else 0.0

    sentence_length_std = np.std(sentence_lengths)

    punctuation_count = sum(1 for char in text if char in ".,;:!?")
    punctuation_density = punctuation_count / len(text) if text else 0.0

    vocabulary_repetition = (
        1 - len(set(words)) / len(words)
        if words else 0.0
    )

    return pd.Series({
        "feature_text_length": len(text),
        "feature_avg_sentence_len": np.mean(sentence_lengths),
        "feature_avg_word_len": np.mean(word_lengths),
        "feature_type_token_ratio": ttr,
        "feature_sentence_length_std": sentence_length_std,
        "feature_punctuation_density": punctuation_density,
        "feature_vocabulary_repetition": vocabulary_repetition,
    })


def build_feature_table(df, text_col="text"):
    """Adds the feature_* columns to a dataframe that has a raw text column."""
    features = df[text_col].apply(compute_features)
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)