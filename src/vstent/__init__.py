"""vstent: virtual stenting planner (research prototype, not a medical device)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vstent")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

DISCLAIMER = "Research use only — not for clinical decisions."
