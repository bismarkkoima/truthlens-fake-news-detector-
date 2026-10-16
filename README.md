# TruthLens: Fake News Detector

TruthLens is a small Flask web app that reads a news article and tells you whether its writing looks more like **real reporting** or **fake news**. It also shows *which words* pushed it to that verdict.

It started as a rethink of my earlier fake news detector project. The goal is the same, but it's lighter, more honest about its accuracy, and easier to run.

| | Original app | TruthLens |
|---|---|---|
| Models | TF-IDF + LogReg, Keras LSTM | TF-IDF + Logistic Regression, Complement Naive Bayes |
| Explanations | LIME (random sampling, slow) | Exact per-word contributions (linear model weights) |
| Data leakage | Reuters datelines and image credits left in | Stripped, plus de-duplication before the split |
| Storage | SQLAlchemy ORM | Plain `sqlite3` |
| Charts | matplotlib/seaborn/plotly on the server | Pure HTML/CSS, works offline |
| Extras | | JSON API, history filters, dark mode, mobile layout, tests |
| Dependencies | ~20 incl. TensorFlow | 4 (Flask, scikit-learn, pandas, joblib) |

## Quick start

```bash
cd truthlens-fake-news-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py            # open http://localhost:5000
```

A trained model is included at `models/detector.joblib` (about 3 MB), so the app works straight away.

## Retraining

Training uses the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) (`Fake.csv` and `True.csv`). Download them into a `data/` folder, then:

```bash
python train.py --fake data/Fake.csv --true data/True.csv
# or
python train.py --fake path/to/Fake.csv --true path/to/True.csv --out models/detector.joblib
```

Results on a held-out 20% split (38.5k de-duplicated articles):

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 98.5% | 98.7% |
| Complement Naive Bayes | 95.2% | 95.5% |

### Why the cleaning matters

The ISOT dataset has "giveaways" that let a model cheat:

- every real article starts with a dateline like `WASHINGTON (Reuters) -`
- many fake articles end with `Featured image via Getty Images`
- fake articles had their apostrophes stripped (`don t`), real ones kept them (`don’t`)

`detector/preprocessing.py` removes all three, so the model has to learn from vocabulary and tone instead.

## JSON API

```bash
curl -X POST http://localhost:5000/api/check \
  -H "Content-Type: application/json" \
  -d '{"text": "…at least 20 words…", "model": "logreg", "save": false}'
```

```json
{
  "label": "FAKE",
  "confidence": 97.3,
  "real_probability": 0.027,
  "model": "logreg",
  "evidence": [{"term": "video", "weight": -1.57}, ...]
}
```

`model` is `logreg` (default) or `nb`. `save: false` keeps the check out of the history.

## Project layout

```
truthlens-fake-news-detector/
├── run.py                 # start the web server
├── train.py               # train and save models/detector.joblib
├── detector/
│   ├── __init__.py        # create_app() factory
│   ├── preprocessing.py   # text cleaning (shared by training and inference)
│   ├── model.py           # Detector: predict + explain
│   ├── storage.py         # SQLite history
│   ├── routes.py          # pages + JSON API
│   ├── templates/
│   └── static/style.css
├── models/detector.joblib
└── tests/                 # pytest, uses a tiny in-memory model
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploying

```bash
gunicorn "detector:create_app()" --bind 0.0.0.0:$PORT
```

Set `DATABASE=/path/to/history.sqlite3` to keep the history somewhere persistent.

## Limitations

TruthLens judges **writing style, not truth**. It was trained on US political news from 2015 to 2018, so it is less reliable on other topics, regions and eras. Use it as a hint, never as a verdict.

---

Built by [@bismarkkoima](https://github.com/bismarkkoima).
