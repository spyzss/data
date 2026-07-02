from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestAsset:
    asset_id: str
    scene: str
    task: str


@dataclass(frozen=True)
class IdIssue:
    issue_type: str
    asset_id: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    valid_ids: set[str]
    issues: list[IdIssue]

    def issue_pairs(self) -> set[tuple[str, str]]:
        return {(issue.issue_type, issue.asset_id) for issue in self.issues}


@dataclass(frozen=True)
class SampleRow:
    asset_id: str
    scene: str
    task: str
    reason: str


@dataclass(frozen=True)
class PullResult:
    asset_id: str
    file_type: str
    source: str
    target: Path
    status: str
    error: str = ""
