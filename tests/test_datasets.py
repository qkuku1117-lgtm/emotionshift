"""Checks against the real data; skipped until `bash scripts/fetch_data.sh` has run."""

import pytest

from emotionshift.data import EMOTIONS, add_shift_columns, duplicate_dialogues, load, load_meld_fr
from emotionshift.data.datasets import DATA_ROOT

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "meld").exists(), reason="data not downloaded")


@pytest.fixture(scope="module")
def meld():
    return add_shift_columns(load("meld"))


def test_meld_split_sizes(meld):
    assert meld.groupby("split").size().to_dict() == {"train": 9989, "dev": 1109, "test": 2610}


def test_labels_are_the_shared_ekman_set(meld):
    assert set(meld["emotion"]) == set(EMOTIONS)
    assert set(load("dailydialog")["emotion"]) == set(EMOTIONS)


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_shift_counts_match_meld_fr_flips(meld, split):
    """Our shift definition reproduces the MELD-FR flips (the released files miss at most one)."""
    ours = (meld.loc[meld["split"] == split, "stratum"] == "shift").sum()
    flips = load_meld_fr(split)["instance_id"].nunique()
    assert 0 <= ours - flips <= 1


def test_dailydialog_test_leakage_is_detected():
    dd = load("dailydialog", splits=("train", "test"))
    leaked = duplicate_dialogues(dd[dd["split"] == "train"], dd[dd["split"] == "test"])
    assert 100 <= len(leaked) <= 120
