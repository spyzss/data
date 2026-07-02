"""Clip-level supplier text_label integrity check."""

from __future__ import annotations

import json
from typing import Any

from precheck.base import BaseCheck
from precheck.registry import register
from qc_common.hdf5_loader import read_scalar_json
from qc_common.types import CheckResult, ClipInputs

from .quality_score import SUMMARY_FRAME_IDX


@register
class TextIntegrityCheck(BaseCheck):
    """Validate required supplier text_label fields."""

    name = "text_integrity"
    granularity = "clip"

    def __init__(self, config: dict) -> None:
        self.required_fields = list(
            config.get("required_fields", ["scene", "task", "text_en"])
        )

    def run(self, clip: ClipInputs) -> list[CheckResult]:
        text_label, parse_error, absent = self._text_label(clip)
        metrics = self._empty_metrics()
        if absent:
            return [
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=SUMMARY_FRAME_IDX,
                    metrics=metrics,
                    flag=True,
                    reason="no text_label",
                )
            ]
        if parse_error is not None or text_label is None:
            return [
                CheckResult(
                    check=self.name,
                    episode_idx=clip.episode_idx,
                    frame_idx=SUMMARY_FRAME_IDX,
                    metrics=metrics,
                    flag=True,
                    reason="text_label not valid JSON",
                )
            ]

        missing_fields: list[str] = []
        empty_fields: list[str] = []
        metrics = {}
        for field in self.required_fields:
            present = field in text_label
            nonempty = present and self._nonempty(text_label[field])
            metrics[f"field_present_{field}"] = float(present)
            metrics[f"field_nonempty_{field}"] = float(nonempty)
            if not present:
                missing_fields.append(field)
            elif not nonempty:
                empty_fields.append(field)

        metrics["missing_field_count"] = float(len(missing_fields))
        metrics["empty_field_count"] = float(len(empty_fields))
        flag = bool(missing_fields or empty_fields)
        return [
            CheckResult(
                check=self.name,
                episode_idx=clip.episode_idx,
                frame_idx=SUMMARY_FRAME_IDX,
                metrics=metrics,
                flag=True if flag else None,
                reason=self._reason(missing_fields, empty_fields),
            )
        ]

    def _text_label(
        self,
        clip: ClipInputs,
    ) -> tuple[dict[str, Any] | None, str | None, bool]:
        if clip.text_label is not None:
            return clip.text_label, None, False
        if clip.text_label_parse_error is not None:
            return None, clip.text_label_parse_error, False
        if clip.text_label_raw is not None:
            parsed, _raw, error = read_scalar_json(clip.text_label_raw)
            return parsed, error, False
        return None, None, True

    def _empty_metrics(self) -> dict[str, float]:
        metrics = {}
        for field in self.required_fields:
            metrics[f"field_present_{field}"] = 0.0
            metrics[f"field_nonempty_{field}"] = 0.0
        metrics["missing_field_count"] = float(len(self.required_fields))
        metrics["empty_field_count"] = 0.0
        return metrics

    def _nonempty(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, bytes):
            return bool(value.decode("utf-8").strip())
        if isinstance(value, (list, tuple, dict, set)):
            return bool(value)
        return bool(str(value).strip())

    def _reason(self, missing_fields: list[str], empty_fields: list[str]) -> str:
        if not missing_fields and not empty_fields:
            return "required text_label fields present and nonempty"
        reason = {
            "missing_fields": missing_fields,
            "empty_fields": empty_fields,
        }
        return json.dumps(reason, sort_keys=True)
