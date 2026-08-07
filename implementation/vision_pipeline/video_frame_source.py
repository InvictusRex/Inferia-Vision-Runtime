from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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


class DatasetVideoSource(FrameSource):
    """Multi-video frame source with chunked-episode sampling.

    On each `random_episode` call a random video is selected and a random
    contiguous chunk within it (of `min_frames`..`max_frames` yielded frames)
    is chosen. Subsequent `iter_frames` yields exactly that chunk. Seeded for
    reproducible episode sequences.
    """

    def __init__(self, paths: list[str], seed: int = 0):
        if not paths:
            raise ValueError("DatasetVideoSource requires at least one video path")
        self.paths = [Path(p) for p in paths]
        self._rng = np.random.default_rng(int(seed))
        self._current_index = 0
        self._start_frame = 0
        self._max_frames: int | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._width = 0
        self._height = 0
        self._set_current(0)

    def _set_current(self, index: int):
        self._current_index = int(index) % len(self.paths)
        probe = VideoFrameSource(str(self.paths[self._current_index]))
        self._fps = probe.fps
        self._frame_count = probe.frame_count
        self._width = probe.width
        self._height = probe.height

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

    @property
    def current_path(self) -> Path:
        return self.paths[self._current_index]

    @property
    def n_videos(self) -> int:
        return len(self.paths)

    def random_episode(self, min_frames: int = 150, max_frames: int = 300, step: int = 1):
        """Pick a random video + random contiguous chunk.

        Returns ``(start_frame, length)`` of the chosen chunk. ``length`` is in
        *yielded* frames (accounting for ``step``).
        """
        step = max(1, int(step))
        min_frames = max(1, int(min_frames))
        index = int(self._rng.integers(0, len(self.paths)))
        self._set_current(index)
        n = self._frame_count
        if n <= 0:
            self._start_frame = 0
            self._max_frames = None
            return 0, 0
        hi = min(int(max_frames), n // step)
        if hi < min_frames:
            length = hi
        else:
            length = int(self._rng.integers(min_frames, hi + 1))
        if length <= 0:
            self._start_frame = 0
            self._max_frames = None
            return 0, 0
        max_start = max(0, n - length * step)
        start = int(self._rng.integers(0, max_start + 1))
        self._start_frame = start
        self._max_frames = length
        return start, length

    def iter_frames(self, step: int = 1):
        step = max(1, int(step))
        cap = cv2.VideoCapture(str(self.paths[self._current_index]))
        try:
            if self._start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
            i = 0
            yielded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    yield frame
                    yielded += 1
                    if self._max_frames is not None and yielded >= self._max_frames:
                        break
                i += 1
        finally:
            cap.release()
