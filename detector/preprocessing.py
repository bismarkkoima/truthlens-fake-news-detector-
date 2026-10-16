"""Text cleaning shared by training and inference.

The ISOT Fake/True news dataset carries a few "giveaway" artifacts that let a
model score ~99% without reading the article at all: every real story starts
with a Reuters dateline, and many fake ones end with image credits. We strip
those so the classifier has to learn from the writing itself.
"""

import re

_DATELINE = re.compile(r"^[^-–]{0,120}?\((?:reuters|ap)\)\s*[-–]\s*", re.IGNORECASE)
_IMAGE_CREDIT = re.compile(
    r"(featured image|photo|image|screenshot|screen capture)s?\s+(via|by|credit)\b.*$",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|org|net)\S*", re.IGNORECASE)
_HANDLE = re.compile(r"[@#]\w+")
_APOSTROPHE = re.compile(r"[’'`]")
# Fake stories in the dataset lost their apostrophes ("don t", "it s"), real
# ones kept them ("don’t"). Re-join both forms so neither leaks the label.
_BROKEN_CONTRACTION = re.compile(r"\b([a-z]+) (t|s|ve|ll|re|m|d)\b")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")

# Words that only exist because of the source, not the content.
_SOURCE_WORDS = re.compile(r"\b(reuters|getty|images|wire|21st century)\b")


def clean_text(text: str) -> str:
    """Normalise a raw article (title and/or body) into model-ready text."""
    if not text:
        return ""
    text = _DATELINE.sub("", text.strip())
    text = _IMAGE_CREDIT.sub("", text)
    text = _URL.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    text = _APOSTROPHE.sub("", text.lower())
    text = _BROKEN_CONTRACTION.sub(r"\1\2", text)
    text = _NON_LETTER.sub(" ", text)
    text = _SOURCE_WORDS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())
