"""Shared emotion label space (proposal: "Data" section).

MELD and DailyDialog already use the same seven labels once DailyDialog's
``happiness`` is read as ``joy`` and ``no_emotion`` as ``neutral``, so no lossy
mapping is needed between the two datasets.

EmoWOZ (Feng et al., 2022) keeps its own seven labels for task-oriented
dialogue: mapping "dissatisfied" or "apologetic" onto Ekman emotions would be
lossy. Only its user turns are labelled; system turns are unlabelled context.
Use ``emotions(dataset)`` for the label list of a dataset.
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

# EmoWOZ raw label ids 0..6, in its own order; -1 marks unlabelled system turns.
EMOWOZ_EMOTIONS = [
    "neutral",
    "fearful",  # fearful / sad / disappointed
    "dissatisfied",  # dissatisfied / disliking
    "apologetic",
    "abusive",
    "excited",  # excited / happy / anticipating
    "satisfied",  # satisfied / liking
]
EMOWOZ_MAP = dict(enumerate(EMOWOZ_EMOTIONS))

LABEL_SPACES = {"meld": EMOTIONS, "dailydialog": EMOTIONS, "emowoz": EMOWOZ_EMOTIONS}
DATASETS = tuple(LABEL_SPACES)


def emotions(dataset: str) -> list[str]:
    """Label list of ``dataset``; position is the integer class id."""
    if dataset not in LABEL_SPACES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASETS}")
    return LABEL_SPACES[dataset]


# Valence used for shift typing (surprise is ambiguous, neutral has none).
# EmoWOZ valences follow its labelling scheme; label names do not overlap
# across label spaces, so one table serves every dataset.
POSITIVE = {"joy", "excited", "satisfied"}
NEGATIVE = {
    "anger",
    "sadness",
    "disgust",
    "fear",
    "fearful",
    "dissatisfied",
    "apologetic",
    "abusive",
}

SPLITS = ("train", "dev", "test")
VALENCE = {
    e: (
        "positive"
        if e in POSITIVE
        else "negative"
        if e in NEGATIVE
        else "ambiguous"
        if e == "surprise"
        else "neutral"
    )
    for e in dict.fromkeys([*EMOTIONS, *EMOWOZ_EMOTIONS])
}
