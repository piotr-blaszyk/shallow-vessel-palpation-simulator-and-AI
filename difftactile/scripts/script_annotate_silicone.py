"""Annotate (and review) the silicone dataset by clicking on vessel points.

    DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_annotate_silicone

Existing annotations are loaded and redrawn, so the same window reviews the
shipped annotations and creates new ones.

Editing is LOCKED until you press `g`, so browsing cannot modify annotations by
accident; the setting lasts for the whole session. `z` undoes the last point
added in this session and stops once they are used up, so it can never eat into
annotations loaded from disk.

Keys: g toggle editing, left click add point (max 4/frame), z undo last point
added this session, d clear frame, m / n next / previous video, k / j next /
previous frame, p save, q save and quit, x quit WITHOUT saving (press twice).

Nothing is written until p or q. `x` and killing the process both discard
everything changed since the last save.
"""

from difftactile.data_analysis.experiment.preprocess_silicone_data import annotate

if __name__ == "__main__":
    annotate()
