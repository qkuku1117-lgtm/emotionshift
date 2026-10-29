"""Compatibility imports for the unified data loader; implementation is in loaders.py."""

from .labels import EMOTIONS, SPLITS, VALENCE
from .loaders import (
    DATA_ROOT,
    clean_text,
    duplicate_dialogues,
    load,
    load_dailydialog,
    load_meld,
    load_meld_fr,
)

__all__ = [
    "DATA_ROOT",
    "EMOTIONS",
    "SPLITS",
    "VALENCE",
    "clean_text",
    "duplicate_dialogues",
    "load",
    "load_dailydialog",
    "load_meld",
    "load_meld_fr",
]
