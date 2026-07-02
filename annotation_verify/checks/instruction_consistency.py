"""Semantic instruction/video consistency check stub."""

from __future__ import annotations

from annotation_verify.base import BaseCheck
from annotation_verify.registry import register
from qc_common.types import CheckResult, ClipInputs


@register
class InstructionConsistencyCheck(BaseCheck):
    """VLM-backed description consistency check contract."""

    name = "instruction_consistency"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.sample_frames = int(config.get("sample_frames", 8))
        self.model_name = config.get("model_name")

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        frames = clip.frames
        instruction = clip.instruction
        num_frames = clip.num_frames
        if num_frames == 0:
            return []

        metrics = {
            "available_frames": float(len(frames) if frames is not None else 0),
            "instruction_chars": float(len(instruction or "")),
            "sample_frames_requested": float(self.sample_frames),
        }
        # TODO: call a configured VLM to judge text_en against sampled frames.
        reason = "VLM judgment not implemented; contract and I/O plumbing only"

        return [
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=clip.frame_idx_at(offset),
                metrics=metrics,
                flag=None,
                reason=reason,
            )
            for offset in range(num_frames)
        ]
