from __future__ import annotations

from pathlib import Path

import cv2


class FrameSource:
    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        raise NotImplementedError

    def iter_frames(self, step: int = 1):
        raise NotImplementedError


class VideoFrameSource(FrameSource):
    def __init__(self, path: str):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self._fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def iter_frames(self, step: int = 1):
        step = max(1, int(step))
        cap = cv2.VideoCapture(str(self.path))
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    yield frame
                i += 1
        finally:
            cap.release()
