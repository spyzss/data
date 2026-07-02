from __future__ import annotations

import math
import random
from collections import defaultdict

from acceptance_pull.models import ManifestAsset, SampleRow


DEFAULT_SAMPLE_RATIO = 0.01


def minimum_sample_count(valid_count: int, sample_ratio: float = DEFAULT_SAMPLE_RATIO) -> int:
    if valid_count <= 0:
        return 0
    return math.ceil(valid_count * sample_ratio)


def sample_assets(
    manifest: dict[str, ManifestAsset],
    valid_ids: set[str],
    seed: int,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
) -> list[SampleRow]:
    if not valid_ids:
        raise ValueError("no valid IDs available for sampling")

    rng = random.Random(seed)
    by_scene: dict[str, list[ManifestAsset]] = defaultdict(list)
    for asset_id in sorted(valid_ids):
        by_scene[manifest[asset_id].scene].append(manifest[asset_id])

    selected: dict[str, SampleRow] = {}
    covered_tasks: set[str] = set()
    for scene in sorted(by_scene):
        candidates = by_scene[scene][:]
        rng.shuffle(candidates)
        chosen = candidates[0]
        selected[chosen.asset_id] = SampleRow(chosen.asset_id, chosen.scene, chosen.task, "scene_coverage")
        covered_tasks.add(chosen.task)

    target = max(minimum_sample_count(len(valid_ids), sample_ratio), len(by_scene))
    remaining = [manifest[asset_id] for asset_id in sorted(valid_ids) if asset_id not in selected]
    rng.shuffle(remaining)
    while len(selected) < target and remaining:
        remaining.sort(key=lambda item: (item.task in covered_tasks, item.scene, item.task, item.asset_id))
        chosen = remaining.pop(0)
        selected[chosen.asset_id] = SampleRow(chosen.asset_id, chosen.scene, chosen.task, "task_diversity")
        covered_tasks.add(chosen.task)

    return sorted(selected.values(), key=lambda row: row.asset_id)
