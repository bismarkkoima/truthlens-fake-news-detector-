import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import create_app  # noqa: E402
from detector.preprocessing import clean_text  # noqa: E402

FAKE = [
    "SHOCKING you wont believe what this crooked liar just did watch the video before they delete it",
    "BREAKING the media is hiding the truth share this before it gets banned wake up people",
    "Unbelievable meltdown as the disgusting elites get exposed in a shocking video watch now",
    "They dont want you to know this one secret the corrupt establishment is terrified share it",
]
REAL = [
    "The finance ministry said on Tuesday that economic growth slowed in the third quarter officials said",
    "Lawmakers voted on Wednesday to approve the budget bill according to a statement from the committee",
    "The central bank kept interest rates unchanged on Thursday citing stable inflation a spokesman said",
    "Officials said the talks would resume next week after negotiators met on Monday in the capital",
]


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    texts = [clean_text(t) for t in FAKE + REAL]
    labels = [0] * len(FAKE) + [1] * len(REAL)
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    models = {"logreg": LogisticRegression().fit(X, labels), "nb": ComplementNB().fit(X, labels)}
    path = tmp_path_factory.mktemp("models") / "detector.joblib"
    joblib.dump(
        {
            "vectorizer": vec,
            "models": models,
            "metrics": {k: {"accuracy": 1.0, "f1": 1.0, "confusion": {"tn": 4, "fp": 0, "fn": 0, "tp": 4}} for k in models},
            "trained_at": "2026-01-01T00:00:00+00:00",
            "samples": 8,
        },
        path,
    )
    return path


@pytest.fixture
def app(model_path, tmp_path):
    return create_app({"MODEL_PATH": model_path, "DATABASE": tmp_path / "test.sqlite3", "TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def fake_article():
    return " ".join(FAKE[:2])


@pytest.fixture
def real_article():
    return " ".join(REAL[:2])
