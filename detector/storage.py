"""Tiny SQLite store for past checks (stdlib only, no ORM)."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    words       INTEGER NOT NULL,
    snippet     TEXT    NOT NULL
);
"""


class History:
    def __init__(self, path: str):
        self.path = path
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def add(self, verdict, text: str) -> int:
        snippet = " ".join(text.split())[:280]
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "INSERT INTO checks (created_at, model, label, confidence, words, snippet)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    verdict.model_key,
                    verdict.label,
                    verdict.confidence,
                    len(text.split()),
                    snippet,
                ),
            )
            return cur.lastrowid

    def recent(self, limit: int = 50, label: str | None = None):
        query = "SELECT * FROM checks"
        params = []
        if label in ("FAKE", "REAL"):
            query += " WHERE label = ?"
            params.append(label)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as db:
            return [dict(r) for r in db.execute(query, params)]

    def clear(self):
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM checks")

    def summary(self) -> dict:
        with closing(self._connect()) as db:
            totals = db.execute(
                "SELECT COUNT(*) AS total,"
                " SUM(label = 'FAKE') AS fake,"
                " SUM(label = 'REAL') AS real,"
                " AVG(confidence) AS avg_conf"
                " FROM checks"
            ).fetchone()
            per_day = db.execute(
                "SELECT substr(created_at, 1, 10) AS day,"
                " SUM(label = 'FAKE') AS fake, SUM(label = 'REAL') AS real, COUNT(*) AS total"
                " FROM checks GROUP BY day ORDER BY day DESC LIMIT 30"
            ).fetchall()
            per_model = db.execute(
                "SELECT model, COUNT(*) AS n, AVG(confidence) AS avg_conf"
                " FROM checks GROUP BY model"
            ).fetchall()
            buckets = db.execute(
                "SELECT MIN(CAST(confidence / 10 AS INTEGER), 9) AS bucket, COUNT(*) AS n"
                " FROM checks GROUP BY bucket"
            ).fetchall()

        conf_hist = [0] * 10
        for b in buckets:
            conf_hist[b["bucket"]] = b["n"]
        return {
            "total": totals["total"] or 0,
            "fake": totals["fake"] or 0,
            "real": totals["real"] or 0,
            "avg_conf": round(totals["avg_conf"] or 0, 1),
            "per_day": [dict(r) for r in reversed(per_day)],
            "per_model": [dict(r) for r in per_model],
            "conf_hist": conf_hist,
        }
