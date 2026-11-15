import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_b4_all as MB  # noqa: E402


def test_merge_adds_block_and_is_idempotent(tmp_path):
    for tag, k in [("b4", "10"), ("b4all", "all")]:
        d = tmp_path / "reports" / tag / "meld"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"chosen_k": k, "test": {"test": {"n": 3}}, "x": tag}))
    (tmp_path / "reports/b4all/meld/controls.csv").write_text("input,score\nreal,1\n")
    assert MB.merge(tmp_path, "b4", "b4all", ["meld", "emowoz"], "A100") == ["meld"]
    first = (tmp_path / "reports/b4/meld/summary.json").read_text()
    s = json.loads(first)
    assert s["chosen_k"] == "10" and s["x"] == "b4"                      # untouched
    assert s["k_all"]["chosen_k"] == "all" and s["k_all"]["note"] == "A100"
    assert (tmp_path / "reports/b4/meld/controls_k_all.csv").exists()
    MB.merge(tmp_path, "b4", "b4all", ["meld"], "A100")
    assert (tmp_path / "reports/b4/meld/summary.json").read_text() == first
