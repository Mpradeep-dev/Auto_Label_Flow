"""Best-effort parser for Ultralytics' per-epoch training console output.

Why this exists: LOCAL training gets live per-epoch rows "for free" via
Ultralytics' own `on_fit_epoch_end` callback (see workers/tasks/training.py) —
structured data straight from the trainer object, running in this same
process. A Kaggle job trains on Kaggle's own remote kernel; there's no
callback hook this app can attach to over there. The kernel's *console log*
is the only channel back, and Ultralytics prints one progress line per epoch
plus (when validation runs) a summary row — this module regexes those back
into the same shape `on_fit_epoch_end` produces, so a Kaggle job can populate
`training_job_epochs` the same way and the existing per-epoch chart UI (built
for LOCAL, entirely provider-agnostic — it just queries by training_job_id)
picks it up with no frontend change.

Caveat, stated plainly: this is regex-matched against Ultralytics' current
stdout format (confirmed against the installed `ultralytics` version's
actual training loop print layout — trainer.py's `pbar.set_description` and
val.py's summary-row `pf` format), not a stable/versioned API — a future
Ultralytics release could reformat that output and silently break this.
This module expects plain decoded text: the input Kaggle kernel logs
actually arrive as (confirmed live, against a real completed kernel run —
see `kaggle_provider.py`'s `_decode_kaggle_log`) is a JSON array of
`{"stream_name", "time", "data"}` records, with a whole epoch's worth of
`\r`-redrawn progress bar updates packed into one record's `data` field —
`get_logs()` decodes that into plain text before this module ever sees it,
which is also why `_EPOCH_LINE` below is deliberately not `^`-anchored.
Every call site wraps this in a try/except and treats "found nothing new"
as a no-op, never a failure — a parse miss degrades to "no epoch history
yet," the same before-this-existed behavior, never a broken job.

Expected input shape, one block per epoch, e.g.:

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/50      2.87G      1.478      2.997      1.379         26        640: 100%|##########| 8/8 ...
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|##########| 2/2 ...
                   all         64        128      0.823      0.756      0.812      0.543
"""
from __future__ import annotations

import re

# Ultralytics formats both rows below with Python's `%g`, which switches to
# scientific notation outside a certain magnitude (confirmed against the
# installed ultralytics version's own trainer.py/val.py format strings:
# "%11.4g" for the loss row, "%11.3g" for the P/R/mAP row) — a loss or
# metric very close to 0 can print as e.g. "1.234e-05", not just "0.00001".
_FLOAT = r"[\d.]+(?:[eE][+-]?\d+)?"

# "   3/50      2.87G      1.478      2.997      1.379         26        640: 100%|"
#
# Deliberately NOT `^`-anchored (confirmed against a real captured Kaggle
# kernel log): tqdm redraws this line once per batch within an epoch via
# `\r` + an ANSI clear-line code, not a real `\n` — a whole epoch's worth of
# redraws (0%, 5%, ..., 100%) live inside ONE log record, so only the very
# first redraw of an epoch would ever follow a true line start. Matching
# unanchored means every redraw of a given epoch matches, each overwriting
# `by_epoch[epoch_num]` in `parse_ultralytics_epochs` below — the LAST match
# (the 100%-complete redraw) is what survives, which is exactly the epoch's
# final accumulated loss, same value LOCAL training's own callback reports.
_EPOCH_LINE = re.compile(
    rf"(?P<epoch>\d+)/(?P<total>\d+)\s+\S+\s+(?P<box>{_FLOAT})\s+(?P<cls>{_FLOAT})\s+(?P<dfl>{_FLOAT})\s+\d+\s+\d+:"
)
# "       all         64        128      0.823      0.756      0.812      0.543"
_VAL_SUMMARY_LINE = re.compile(
    rf"^\s*all\s+\d+\s+\d+\s+(?P<precision>{_FLOAT})\s+(?P<recall>{_FLOAT})\s+(?P<map50>{_FLOAT})\s+(?P<map50_95>{_FLOAT})\s*$",
    re.MULTILINE,
)


def parse_ultralytics_epochs(log_text: str) -> list[dict]:
    """Returns one dict per epoch found, in ascending epoch order:
    {"epoch": int, "box_loss": float, "cls_loss": float, "dfl_loss": float,
     "precision": float | None, "recall": float | None,
     "map50": float | None, "map50_95": float | None}

    A validation summary ("all ...") row is paired with whichever epoch's
    loss line most recently preceded it in the log — matching how
    Ultralytics actually interleaves them (train epoch, then that epoch's
    val pass immediately after). If a run doesn't validate every epoch
    (e.g. a `val_period` > 1), the skipped epochs simply keep
    precision/recall/map50/map50_95 as None, same nullability as LOCAL
    training's own `_extract_epoch_metrics`.
    """
    if not log_text:
        return []

    events: list[tuple[int, str, re.Match]] = []
    for m in _EPOCH_LINE.finditer(log_text):
        events.append((m.start(), "epoch", m))
    for m in _VAL_SUMMARY_LINE.finditer(log_text):
        events.append((m.start(), "val", m))
    events.sort(key=lambda e: e[0])

    by_epoch: dict[int, dict] = {}
    pending_epoch: int | None = None
    for _, kind, m in events:
        if kind == "epoch":
            epoch_num = int(m.group("epoch"))
            by_epoch[epoch_num] = {
                "epoch": epoch_num,
                "box_loss": float(m.group("box")),
                "cls_loss": float(m.group("cls")),
                "dfl_loss": float(m.group("dfl")),
                "precision": None,
                "recall": None,
                "map50": None,
                "map50_95": None,
            }
            pending_epoch = epoch_num
        elif kind == "val" and pending_epoch is not None:
            row = by_epoch[pending_epoch]
            row["precision"] = float(m.group("precision"))
            row["recall"] = float(m.group("recall"))
            row["map50"] = float(m.group("map50"))
            row["map50_95"] = float(m.group("map50_95"))
            # Only the first "all" row after a given epoch line belongs to
            # it — later ones (e.g. a final re-validation summary at the
            # very end of training) shouldn't overwrite an earlier epoch.
            pending_epoch = None

    return [by_epoch[e] for e in sorted(by_epoch)]
