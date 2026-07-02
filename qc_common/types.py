"""Data contracts shared by precheck and annotation verification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


LazyLoader = Callable[[], Any]


@dataclass
class CheckResult:
    """Uniform per-frame output emitted by every QC check."""

    check: str
    episode_idx: int
    frame_idx: int
    metrics: dict[str, float] = field(default_factory=dict)
    flag: bool | None = None
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        """Convert to a flat record suitable for pandas storage."""
        return {
            "check": self.check,
            "episode_idx": self.episode_idx,
            "frame_idx": self.frame_idx,
            "metrics": self.metrics,
            "flag": self.flag,
            "reason": self.reason,
        }


class ClipInputs:
    """
    Optional, lazily-loaded inputs for one clip.

    Keypoints are 3D joint positions in meters, keyed by joint name with shape
    (num_frames, 3). Optional rotations are the corresponding 3x3 rotation
    blocks, keyed by joint name with shape (num_frames, 3, 3). Supplier-specific
    transforms should be converted by an injected adapter before checks consume
    them.
    """

    def __init__(
        self,
        episode_idx: int,
        frame_indices: list[int] | None = None,
        frames: list[np.ndarray] | np.ndarray | None = None,
        keypoints: dict[str, np.ndarray] | None = None,
        rotations: dict[str, np.ndarray] | None = None,
        confidences: dict[str, np.ndarray] | None = None,
        quality_hand: np.ndarray | None = None,
        masks: dict[int, np.ndarray] | list[np.ndarray] | np.ndarray | None = None,
        instruction: str | None = None,
        text_label: dict[str, Any] | None = None,
        text_label_raw: str | None = None,
        text_label_parse_error: str | None = None,
        intrinsics: np.ndarray | None = None,
        fps: float | None = None,
        frames_loader: LazyLoader | None = None,
        keypoints_loader: LazyLoader | None = None,
        rotations_loader: LazyLoader | None = None,
        confidences_loader: LazyLoader | None = None,
        quality_hand_loader: LazyLoader | None = None,
        masks_loader: LazyLoader | None = None,
        instruction_loader: LazyLoader | None = None,
        text_label_loader: LazyLoader | None = None,
        text_label_raw_loader: LazyLoader | None = None,
        text_label_parse_error_loader: LazyLoader | None = None,
        intrinsics_loader: LazyLoader | None = None,
    ) -> None:
        self.episode_idx = episode_idx
        self.frame_indices = frame_indices
        self.fps = fps
        self._frames = frames
        self._keypoints = keypoints
        self._rotations = rotations
        self._confidences = confidences
        self._quality_hand = quality_hand
        self._masks = masks
        self._instruction = instruction
        self._text_label = text_label
        self._text_label_raw = text_label_raw
        self._text_label_parse_error = text_label_parse_error
        self._intrinsics = intrinsics
        self._frames_loader = frames_loader
        self._keypoints_loader = keypoints_loader
        self._rotations_loader = rotations_loader
        self._confidences_loader = confidences_loader
        self._quality_hand_loader = quality_hand_loader
        self._masks_loader = masks_loader
        self._instruction_loader = instruction_loader
        self._text_label_loader = text_label_loader
        self._text_label_raw_loader = text_label_raw_loader
        self._text_label_parse_error_loader = text_label_parse_error_loader
        self._intrinsics_loader = intrinsics_loader

    @property
    def frames(self) -> list[np.ndarray] | np.ndarray | None:
        if self._frames is None and self._frames_loader is not None:
            self._frames = self._frames_loader()
        return self._frames

    @property
    def keypoints(self) -> dict[str, np.ndarray] | None:
        if self._keypoints is None and self._keypoints_loader is not None:
            self._keypoints = self._keypoints_loader()
        return self._keypoints

    @property
    def rotations(self) -> dict[str, np.ndarray] | None:
        if self._rotations is None and self._rotations_loader is not None:
            self._rotations = self._rotations_loader()
        return self._rotations

    @property
    def confidences(self) -> dict[str, np.ndarray] | None:
        if self._confidences is None and self._confidences_loader is not None:
            self._confidences = self._confidences_loader()
        return self._confidences

    @property
    def quality_hand(self) -> np.ndarray | None:
        if self._quality_hand is None and self._quality_hand_loader is not None:
            self._quality_hand = self._quality_hand_loader()
        return self._quality_hand

    @property
    def masks(self) -> dict[int, np.ndarray] | list[np.ndarray] | np.ndarray | None:
        if self._masks is None and self._masks_loader is not None:
            self._masks = self._masks_loader()
        return self._masks

    @property
    def instruction(self) -> str | None:
        if self._instruction is None and self._instruction_loader is not None:
            self._instruction = self._instruction_loader()
        return self._instruction

    @property
    def text_label(self) -> dict[str, Any] | None:
        if self._text_label is None and self._text_label_loader is not None:
            self._text_label = self._text_label_loader()
        return self._text_label

    @property
    def text_label_raw(self) -> str | None:
        if self._text_label_raw is None and self._text_label_raw_loader is not None:
            self._text_label_raw = self._text_label_raw_loader()
        return self._text_label_raw

    @property
    def text_label_parse_error(self) -> str | None:
        if (
            self._text_label_parse_error is None
            and self._text_label_parse_error_loader is not None
        ):
            self._text_label_parse_error = self._text_label_parse_error_loader()
        return self._text_label_parse_error

    @property
    def intrinsics(self) -> np.ndarray | None:
        if self._intrinsics is None and self._intrinsics_loader is not None:
            self._intrinsics = self._intrinsics_loader()
        return self._intrinsics

    @property
    def num_frames(self) -> int:
        if self.frame_indices is not None:
            return len(self.frame_indices)
        frames = self.frames
        if frames is not None:
            return int(len(frames))
        keypoints = self.keypoints
        if keypoints:
            first = next(iter(keypoints.values()))
            return int(first.shape[0])
        rotations = self.rotations
        if rotations:
            first = next(iter(rotations.values()))
            return int(first.shape[0])
        quality_hand = self.quality_hand
        if quality_hand is not None:
            return int(quality_hand.shape[0])
        return 0

    def frame_idx_at(self, offset: int) -> int:
        if self.frame_indices is not None:
            return int(self.frame_indices[offset])
        return int(offset)
