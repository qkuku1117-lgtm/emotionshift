# Add EmoWOZ to the shared data pipeline

`load("emowoz")` now returns the shared ten-column target schema plus native
emotion IDs/names and source provenance. The loader consumes the official
Zenodo release and split manifest, excludes unlabelled system turns, preserves
original user-turn positions, and reuses the existing same-speaker shift rules.

The seven-class mapping is explicitly lossy and documented in `EMOWOZ.md`.
Merged native classes can produce stable mapped transitions. No changes are
made to existing MELD/DailyDialog loader logic, fixed label IDs or shift rules.

Includes a separate download command, reproducible statistics CSV/JSON with raw
hashes, offline fixture tests, official-release checks, and a VS Code workspace
Python default. Raw data and the pre-existing untracked `uv.lock` are excluded
from this change set.

Validation: **20 passed, 2 skipped** (optional model dependencies absent).
Existing MELD/DailyDialog regression counts and pilot results pass unchanged.
`git diff --check` passes. Full-release user targets: train 66,474 / dev 8,509 /
test 8,634, spanning 11,433 unique dialogues. Mapped shift rates excluding first
turns: train 35.89% / dev 35.40% / test 34.71%.

No training runs or native-taxonomy benchmark claims are included.
