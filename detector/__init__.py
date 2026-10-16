"""TruthLens — a small Flask app that flags likely fake news articles."""

import os
from pathlib import Path

from flask import Flask, render_template

from .model import Detector
from .storage import History

ROOT = Path(__file__).resolve().parent.parent


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MODEL_PATH=os.environ.get("MODEL_PATH", ROOT / "models" / "detector.joblib"),
        DATABASE=os.environ.get("DATABASE", ROOT / "instance" / "history.sqlite3"),
        MIN_WORDS=20,
        MAX_CHARS=20_000,
    )
    if config:
        app.config.update(config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    app.extensions["history"] = History(str(app.config["DATABASE"]))

    model_path = Path(app.config["MODEL_PATH"])
    app.extensions["detector"] = Detector.load(model_path) if model_path.exists() else None

    from .routes import bp

    app.register_blueprint(bp)

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="That page doesn't exist."), 404

    return app
