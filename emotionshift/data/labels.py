"""Shared emotion label space (proposal: "Data" section).

MELD and DailyDialog already use the same seven labels once DailyDialog's
``happiness`` is read as ``joy`` and ``no_emotion`` as ``neutral``, so no lossy
mapping is needed between the two datasets.
"""

# Fixed order: every model in the project should use these integer ids.
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}

# Raw dataset label -> shared label.
MELD_MAP = {e: e for e in EMOTIONS}
DAILYDIALOG_MAP = {
    "no_emotion": "neutral",
    "happiness": "joy",
    "surprise": "surprise",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    "fear": "fear",
}

# Valence used for shift typing (surprise is ambiguous, neutral has none).
POSITIVE = {"joy"}
NEGATIVE = {"anger", "sadness", "disgust", "fear"}
