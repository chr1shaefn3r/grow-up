"""What `pip install` puts in an environment.

Read from `pyproject.toml` itself, so a dependency changed there without its
reason is caught here rather than in somebody's broken install.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def dependencies() -> dict[str, str]:
    """Distribution name -> its full requirement string."""
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    return {re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0].lower(): req for req in declared}


class TestThereIsOneOpenCV:
    """Every OpenCV distribution writes the same `cv2` folder. mediapipe depends
    on opencv-contrib-python, and declaring opencv-python as well put two of them
    in every install: it worked only while pip chose the same version for both,
    and uninstalling either deleted files the other still needed."""

    def test_it_is_the_one_mediapipe_brings(self):
        assert "opencv-contrib-python" in dependencies()

    def test_no_other_opencv_is_declared_beside_it(self):
        others = [name for name in dependencies()
                  if name.startswith("opencv") and name != "opencv-contrib-python"]
        assert others == []


class TestMediapipeStaysBelowOne:

    def test_the_pin_is_still_there(self):
        """1.0.1 aborts building the face landmarker on macOS, so every analyze
        worker died at startup. Lift it only after checking a newer 1.x there."""
        assert "<1.0" in dependencies()["mediapipe"].replace(" ", "")
