"""Shared helpers: dataset loading, train/test split, preprocessing pipeline.

Dataset: UCI Phishing Websites (Mohammad, Thabtah & McCluskey, 2015),
downloaded from OpenML (data id 4534). Every feature is an encoded URL / page /
domain indicator with values in {-1, 0, 1}. In the original coding the target
`Result` is -1 for phishing and 1 for legitimate.
"""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "PhishingWebsites.arff"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RESULTS_FILE = OUT / "results.json"

RANDOM_STATE = 42
TARGET = "phishing"  # 1 = suspicious (phishing), 0 = normal (legitimate)


def load_data(dedupe: bool = True) -> pd.DataFrame:
    """Read the ARFF file into a DataFrame and add a 0/1 `phishing` label.

    The raw file has 5,206 exact duplicate rows (47%). Left in, the same website
    can land in both train and test sets and inflate every score, so they are
    dropped by default."""
    names = [
        re.match(r"@attribute\s+(\S+)", line, re.I).group(1)
        for line in DATA_FILE.read_text().splitlines()
        if line.lower().startswith("@attribute")
    ]
    # Data rows contain no '@', so treating '@' as a comment char skips the header.
    df = pd.read_csv(DATA_FILE, comment="@", header=None, names=names, skipinitialspace=True)
    df[TARGET] = (df.pop("Result") == -1).astype(int)
    return df.drop_duplicates().reset_index(drop=True) if dedupe else df


def split(df: pd.DataFrame):
    """Stratified 80/20 split so both sets keep the phishing ratio."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Binary {-1,1} indicators are treated as numeric and standardised.
    Three-level indicators {-1,0,1} (legit / suspicious / phishing) are
    categorical and one-hot encoded. Imputers guard against missing values
    (the shipped dataset has none, but new data might)."""
    ternary = [c for c in X.columns if X[c].nunique() > 2]
    binary = [c for c in X.columns if c not in ternary]
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, binary), ("cat", cat, ternary)])


def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Fit `model`, time training and prediction, return metrics and save a confusion matrix plot."""
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    predict_ms = (time.perf_counter() - t0) * 1000

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Phishing"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()
    fig.savefig(OUT / f"cm_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    res = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "train_seconds": train_s,
        "predict_ms": predict_ms,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(f"{name:<20} acc={res['accuracy']:.4f} prec={res['precision']:.4f} rec={res['recall']:.4f} "
          f"f1={res['f1']:.4f} train={train_s:.3f}s  FP={fp} FN={fn}")
    return res


def save_result(key: str, value) -> None:
    """Merge one entry into outputs/results.json (read by build_report.py)."""
    data = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    data[key] = value
    RESULTS_FILE.write_text(json.dumps(data, indent=2))
