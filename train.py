"""Train the fake news models and save them as one bundle.

Usage:
    python train.py                                  # uses data/Fake.csv and data/True.csv
    python train.py --fake data/Fake.csv --true data/True.csv --out models/detector.joblib
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

from detector.preprocessing import clean_text

HERE = Path(__file__).resolve().parent


def load_dataset(fake_csv: Path, true_csv: Path) -> pd.DataFrame:
    frames = []
    for path, label in ((fake_csv, 0), (true_csv, 1)):
        df = pd.read_csv(path, usecols=["title", "text"])
        df["content"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean_text)
        df["label"] = label
        frames.append(df[["content", "label"]])
        print(f"  {path.name}: {len(df):,} rows")

    data = pd.concat(frames, ignore_index=True)
    data = data[data["content"].str.split().str.len() >= 15]
    # The dataset has many re-posted stories; keep one copy so the
    # test split doesn't contain articles the model has already seen.
    data = data.drop_duplicates("content")
    print(f"  after cleaning + de-duplication: {len(data):,} rows")
    return data


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", type=Path, default=HERE / "data" / "Fake.csv")
    parser.add_argument("--true", type=Path, default=HERE / "data" / "True.csv")
    parser.add_argument("--out", type=Path, default=HERE / "models" / "detector.joblib")
    parser.add_argument("--max-features", type=int, default=50_000)
    args = parser.parse_args()

    print("Loading data")
    data = load_dataset(args.fake, args.true)

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.2, random_state=7, stratify=data["label"]
    )

    print("Vectorising")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.5,
        max_features=args.max_features,
        sublinear_tf=True,
        stop_words="english",
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    models = {
        "logreg": LogisticRegression(C=4.0, max_iter=2000),
        "nb": ComplementNB(alpha=0.3),
    }
    metrics = {}
    for key, model in models.items():
        print(f"Training {key}")
        model.fit(Xtr, y_train)
        metrics[key] = evaluate(model, Xte, y_test)
        print(f"  {json.dumps(metrics[key])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "models": models,
            "metrics": metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "samples": int(len(data)),
        },
        args.out,
        compress=3,
    )
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
