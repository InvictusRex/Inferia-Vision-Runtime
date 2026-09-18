from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from .video_frame_source import FrameSource


def load_detrac_annotations(xml_path: str) -> dict[int, np.ndarray]:
    """Parse a DETRAC-Train/Test-Annotations-XML file into {frame_num: (N,4) xyxy boxes}.

    Class-agnostic (car/bus/van/others all count as "vehicle"), ignored_region
    and occlusion/truncation metadata are not applied -- a fast, honest
    simplification, not a full benchmark-grade evaluation protocol.
    """
    root = ET.parse(xml_path).getroot()
    frames: dict[int, np.ndarray] = {}
    for frame_el in root.findall("frame"):
        num = int(frame_el.get("num"))
        boxes = []
        for box_el in frame_el.findall("./target_list/target/box"):
            left = float(box_el.get("left"))
            top = float(box_el.get("top"))
            w = float(box_el.get("width"))
            h = float(box_el.get("height"))
            boxes.append((left, top, left + w, top + h))
        frames[num] = (
            np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        )
    return frames


class DetracFrameSource(FrameSource):
    """Multi-sequence frame source over UA-DETRAC's per-sequence JPEG folders.

    Mirrors DatasetVideoSource's chunked-episode sampling (random sequence +
    random contiguous frame range) so it's a drop-in alternative frame source
    for VisionRuntimeEnv. Ground-truth boxes for the currently-yielded frame
    are available via `gt_for_current_frame()`.
    """

    FPS = 25.0
    MAX_SAMPLE_ATTEMPTS = 20

    def __init__(self, sequence_dirs: list[str], xml_dir: str, seed: int = 0):
        if not sequence_dirs:
            raise ValueError("DetracFrameSource requires at least one sequence directory")
        self.sequences = [Path(p) for p in sequence_dirs]
        self.xml_dir = Path(xml_dir)
        self._rng = np.random.default_rng(int(seed))
        self._annotations: dict[str, dict[int, np.ndarray]] = {}
        self._current_index = 0
        self._start_frame = 0
        self._max_frames: int | None = None
        self._current_abs_frame = 0
        self._frame_count = 0
        self._set_current(0)

    def _annotations_for(self, seq_dir: Path) -> dict[int, np.ndarray]:
        name = seq_dir.name
        if name not in self._annotations:
            xml_path = self.xml_dir / f"{name}.xml"
            self._annotations[name] = (
                load_detrac_annotations(str(xml_path)) if xml_path.exists() else {}
            )
        return self._annotations[name]

    def _set_current(self, index: int) -> None:
        self._current_index = int(index) % len(self.sequences)
        seq_dir = self.sequences[self._current_index]
        self._frame_count = len(list(seq_dir.glob("img*.jpg")))

    @property
    def fps(self) -> float:
        return self.FPS

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_path(self) -> Path:
        return self.sequences[self._current_index]

    @property
    def n_videos(self) -> int:
        return len(self.sequences)

    def random_episode(self, min_frames: int = 150, max_frames: int = 300, step: int = 1):
        step = max(1, int(step))
        min_frames = max(1, int(min_frames))
        for _ in range(self.MAX_SAMPLE_ATTEMPTS):
            index = int(self._rng.integers(0, len(self.sequences)))
            self._set_current(index)
            n = self._frame_count
            if n > 0:
                break
        else:
            raise FileNotFoundError("DetracFrameSource: no non-empty sequence found")
        hi = min(int(max_frames), n // step)
        length = hi if hi < min_frames else int(self._rng.integers(min_frames, hi + 1))
        if length <= 0:
            self._start_frame, self._max_frames = 0, None
            return 0, 0
        max_start = max(0, n - length * step)
        start = int(self._rng.integers(0, max_start + 1))
        self._start_frame = start
        self._max_frames = length
        return start, length

    def iter_frames(self, step: int = 1):
        step = max(1, int(step))
        seq_dir = self.sequences[self._current_index]
        i, yielded = 0, 0
        frame_num = self._start_frame + 1  # DETRAC frames are 1-indexed
        while True:
            img_path = seq_dir / f"img{frame_num:05d}.jpg"
            if not img_path.exists():
                break
            if i % step == 0:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    break
                self._current_abs_frame = frame_num
                yield frame
                yielded += 1
                if self._max_frames is not None and yielded >= self._max_frames:
                    break
            i += 1
            frame_num += 1

    def gt_for_current_frame(self) -> np.ndarray:
        anns = self._annotations_for(self.sequences[self._current_index])
        return anns.get(self._current_abs_frame, np.zeros((0, 4), dtype=np.float32))
