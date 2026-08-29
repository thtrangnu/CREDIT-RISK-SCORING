"""Plain re-export of ml.src.reason_codes. See the layer-boundary note in scorer.py.

This file exists (rather than importing ml.src.reason_codes directly at each call site)
to match the target layout in docs/NOTES.md and to make the dependency explicit: this IS
the backend's declared dependency on ml/, not a scattering of imports that are hard to
track.
"""
from ml.src.reason_codes import build_reason_codes, direction_phrase, humanize

__all__ = ["build_reason_codes", "direction_phrase", "humanize"]
