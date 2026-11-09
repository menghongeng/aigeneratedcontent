import hashlib

import joblib

# Run the following command to get hash
# python -c "import hashlib;print(hashlib.sha256(open('ai_detector.joblib','rb').read()).hexdigest())"
EXPECTED_HASH = "PASTE_YOUR_HASH_HERE"


def _verify_hash(path):
    # joblib files are pickles, so loading a tampered file can run attacker code - check the hash first
    with open(path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != EXPECTED_HASH:
        raise RuntimeError("Model file hash mismatch - refusing to load")


class AIDetector:
    def __init__(self, path):
        _verify_hash(path)
        bundle = joblib.load(path)  # loaded ONCE when the server starts, never per request
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        self.features = bundle["features"]

    def predict(self, feature_row):
        # feature_row[self.features] keeps the column order identical to training
        proba = float(self.pipeline.predict_proba(feature_row[self.features])[0][1])
        label = "ai" if proba >= self.threshold else "human"
        return label, proba