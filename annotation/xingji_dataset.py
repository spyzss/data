"""Interstellar Silicon Path (星际硅途/星际归途) dataset adapter.

Bridges the vendor's hdf5 + mp4 deliverable into the marmalade annotation
pipeline. The pipeline only needs RGB frames (for SAM3 + DA3) and a task
instruction (for discovery/Qwen). Everything else in the hdf5 — hand/body
pose transforms, confidences, quality_hand — is IGNORED on purpose.

Granularity (confirmed from a real sample 408817_hdf5.hdf5):
- ONE hdf5 file == ONE episode == N frames (sample had N=899), no batch axis.
- Files are id-named, so the dataset is enumerated by listing the hdf5 dir.

Per-hdf5 schema we rely on:
- label/text_label : scalar object (a JSON string). Use the English field
  `text_en` as the instruction. JSON keys: {scene, task, text_cn, text_en, actions}.
- camera/intrinsic : (3,3) fx,fy,cx,cy. Exposed via get_intrinsic().
- transforms/*, confidences/*, label/quality_hand : NOT used here.

Interface matches LeRobotV3Dataset: same __init__ kwargs, __len__,
get_episode(idx), episode_index_at(idx).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class XingjiDataset:
    """Adapter for the vendor hdf5 + mp4 human-manipulation deliverable."""

    def __init__(
        self,
        dataset_path: Path,
        camera_names: list[str],
        instruction_config: dict,
        episode_indices: list[int] | None = None,
        frame_indices: list[int] | None = None,
        sample_frames_per_episode: int | None = None,
        sampling_config: dict | None = None,
        max_episodes: int | None = None,
        max_frames_per_episode: int | None = None,
        load_frames: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.camera_names = camera_names
        self.instruction_config = instruction_config or {}
        self.episode_indices = episode_indices
        self.frame_indices = frame_indices
        self.sample_frames_per_episode = sample_frames_per_episode
        self.sampling_config = sampling_config or {}
        self.max_episodes = max_episodes
        self.max_frames_per_episode = max_frames_per_episode
        self.load_frames = load_frames

        self.hdf5_subdir = self.instruction_config.get("hdf5_subdir", "hdf5")
        self.video_subdir = self.instruction_config.get("video_subdir", "video")
        self.instruction_json_field = self.instruction_config.get(
            "instruction_json_field", "text_en"
        )

        self._clip_ids = self._discover_clips()
        logger.info(
            "XingjiDataset: found %d clips under %s",
            len(self._clip_ids),
            self.dataset_path / self.hdf5_subdir,
        )

    def _discover_clips(self) -> list[str]:
        hdf5_dir = self.dataset_path / self.hdf5_subdir
        if not hdf5_dir.is_dir():
            raise FileNotFoundError(f"hdf5 dir not found: {hdf5_dir}")
        files = sorted(
            [p for p in hdf5_dir.iterdir() if p.suffix in (".h5", ".hdf5")]
        )
        clip_ids = [self._clip_id_from_hdf5(p) for p in files]
        if self.episode_indices is not None:
            clip_ids = [
                clip_ids[i] for i in self.episode_indices if 0 <= i < len(clip_ids)
            ]
        if self.max_episodes is not None:
            clip_ids = clip_ids[: self.max_episodes]
        return clip_ids

    @staticmethod
    def _clip_id_from_hdf5(path: Path) -> str:
        stem = path.stem
        stem = re.sub(r"_hdf5$", "", stem)
        return stem

    def _hdf5_path(self, clip_id: str) -> Path:
        d = self.dataset_path / self.hdf5_subdir
        for cand in (d / f"{clip_id}.hdf5", d / f"{clip_id}.h5",
                     d / f"{clip_id}_hdf5.hdf5"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"no hdf5 for clip {clip_id} under {d}")

    def _video_path(self, clip_id: str) -> Path:
        d = self.dataset_path / self.video_subdir
        for cand in (d / f"{clip_id}.mp4", d / f"{clip_id}_video.mp4"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"no mp4 for clip {clip_id} under {d}")

    def __len__(self) -> int:
        return len(self._clip_ids)

    def episode_index_at(self, idx: int) -> int:
        clip_id = self._clip_ids[idx]
        m = re.search(r"\d+", clip_id)
        return int(m.group()) if m else idx

    def get_episode(self, idx: int) -> dict[str, Any]:
        clip_id = self._clip_ids[idx]
        hdf5_path = self._hdf5_path(clip_id)
        instruction, num_frames_hint = self._read_instruction_and_len(hdf5_path)
        frame_indices, num_frames = self._resolve_frame_indices(
            clip_id, num_frames_hint
        )
        frames: dict[str, dict[int, np.ndarray]] = {}
        if self.load_frames:
            video_path = self._video_path(clip_id)
            decoded = self._decode_frames(video_path, frame_indices)
            for cam in self.camera_names:
                frames[cam] = decoded
        frame_metadata = self._build_frame_metadata(frame_indices, num_frames)
        return {
            "episode_index": self.episode_index_at(idx),
            "instruction": instruction,
            "num_frames": num_frames,
            "frame_indices": frame_indices,
            "frames": frames,
            "frame_metadata": frame_metadata,
        }

    def _read_instruction_and_len(self, hdf5_path: Path) -> tuple[str, int]:
        import h5py

        with h5py.File(hdf5_path, "r") as f:
            raw = f["label/text_label"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            instruction = self._parse_instruction(raw)
            n = 0
            if "transforms/camera" in f:
                n = int(f["transforms/camera"].shape[0])
        return instruction, n

    def _parse_instruction(self, raw: str) -> str:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return str(raw).strip()
        if isinstance(obj, dict):
            val = obj.get(self.instruction_json_field)
            if isinstance(val, str) and val.strip():
                return val.strip()
            for k in ("text_en", "task", "text_cn"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    logger.warning(
                        "XingjiDataset: field %r empty, fell back to %r",
                        self.instruction_json_field, k,
                    )
                    return v.strip()
        return str(raw).strip()

    def _resolve_frame_indices(
        self, clip_id: str, num_frames_hint: int
    ) -> tuple[list[int], int]:
        num_frames = num_frames_hint
        if num_frames <= 0:
            num_frames = self._probe_video_len(clip_id)
        if self.frame_indices is not None:
            picked = [i for i in self.frame_indices if 0 <= i < num_frames]
            return picked, num_frames
        mode = (self.sampling_config or {}).get("mode", "uniform")
        if mode == "subtask_aware":
            logger.warning(
                "XingjiDataset: subtask_aware unsupported (no per-frame "
                "subtask); falling back to uniform."
            )
        stride = (self.sampling_config or {}).get("stride_frames")
        if stride:
            picked = list(range(0, num_frames, int(stride)))
        elif self.sample_frames_per_episode:
            k = min(self.sample_frames_per_episode, num_frames)
            picked = (
                np.linspace(0, num_frames - 1, k).round().astype(int).tolist()
                if k > 0 else []
            )
        else:
            picked = list(range(num_frames))
        if self.max_frames_per_episode:
            picked = picked[: self.max_frames_per_episode]
        return picked, num_frames

    def _build_frame_metadata(
        self, frame_indices: list[int], num_frames: int
    ) -> dict[int, dict[str, Any]]:
        fps = float((self.sampling_config or {}).get("fps", 29.97))
        meta: dict[int, dict[str, Any]] = {}
        for fi in frame_indices:
            meta[fi] = {
                "subtask_index": None,
                "atomic_skill": "",
                "subtask": "",
                "timestamp": (fi / fps) if fps else None,
            }
        return meta

    def _decode_frames(
        self, video_path: Path, frame_indices: list[int]
    ) -> dict[int, np.ndarray]:
        if not frame_indices:
            return {}
        wanted = set(frame_indices)
        out: dict[int, np.ndarray] = {}
        try:
            import av

            container = av.open(str(video_path))
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    out[i] = frame.to_ndarray(format="rgb24")
                    if len(out) == len(wanted):
                        break
            container.close()
        except Exception as exc:
            logger.error("XingjiDataset: PyAV decode failed for %s: %s",
                         video_path, exc)
            raise
        missing = wanted - set(out.keys())
        if missing:
            logger.warning(
                "XingjiDataset: %d requested frames not decoded from %s "
                "(video shorter than hdf5?): %s",
                len(missing), video_path.name, sorted(missing)[:10],
            )
        return out

    def _probe_video_len(self, clip_id: str) -> int:
        try:
            import av

            container = av.open(str(self._video_path(clip_id)))
            stream = container.streams.video[0]
            n = stream.frames or 0
            container.close()
            return int(n)
        except Exception:
            return 0

    def get_intrinsic(self, idx: int = 0) -> np.ndarray:
        import h5py

        with h5py.File(self._hdf5_path(self._clip_ids[idx]), "r") as f:
            return np.array(f["camera/intrinsic"][:], dtype=float)