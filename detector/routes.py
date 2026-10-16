from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from .model import MODEL_NAMES
from .preprocessing import word_count

bp = Blueprint("main", __name__)


def _detector():
    return current_app.extensions["detector"]


def _history():
    return current_app.extensions["history"]


def _validate(text: str, model_key: str) -> str | None:
    """Return an error message, or None if the input can be checked."""
    if _detector() is None:
        return "No trained model found. Run `python train.py` first."
    if model_key not in _detector().available_models:
        return f"Unknown model '{model_key}'."
    if len(text) > current_app.config["MAX_CHARS"]:
        return f"Please keep it under {current_app.config['MAX_CHARS']:,} characters."
    if word_count(text) < current_app.config["MIN_WORDS"]:
        return (
            f"Paste at least {current_app.config['MIN_WORDS']} words — a full paragraph "
            "or article works best. Short claims don't carry enough writing style to judge."
        )
    return None


@bp.get("/")
def index():
    return render_template("index.html", models=MODEL_NAMES)


@bp.post("/check")
def check():
    text = request.form.get("text", "").strip()
    model_key = request.form.get("model", "logreg")
    error = _validate(text, model_key)
    if error:
        return render_template("index.html", models=MODEL_NAMES, error=error, text=text,
                               selected=model_key), 400

    verdict = _detector().predict(text, model_key)
    _history().add(verdict, text)
    max_weight = max((abs(w) for _, w in verdict.evidence), default=1) or 1
    return render_template("result.html", verdict=verdict, text=text, max_weight=max_weight)


@bp.post("/api/check")
def api_check():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    model_key = payload.get("model", "logreg")
    error = _validate(text, model_key)
    if error:
        return jsonify(error=error), 400

    verdict = _detector().predict(text, model_key)
    if payload.get("save", True):
        _history().add(verdict, text)
    return jsonify(
        label=verdict.label,
        confidence=verdict.confidence,
        real_probability=round(verdict.real_probability, 4),
        model=verdict.model_key,
        evidence=[{"term": t, "weight": round(w, 4)} for t, w in verdict.evidence],
    )


@bp.get("/history")
def history():
    label = request.args.get("label")
    return render_template("history.html", rows=_history().recent(100, label), label=label,
                           models=MODEL_NAMES)


@bp.post("/history/clear")
def clear_history():
    _history().clear()
    return redirect(url_for("main.history"))


@bp.get("/stats")
def stats():
    return render_template("stats.html", summary=_history().summary(), models=MODEL_NAMES)


@bp.get("/about")
def about():
    d = _detector()
    return render_template(
        "about.html",
        models=MODEL_NAMES,
        metrics=d.metrics if d else {},
        trained_at=d.trained_at if d else None,
        samples=d.samples if d else None,
    )
