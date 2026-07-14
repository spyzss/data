from acceptance_pull.models import ManifestAsset
from acceptance_pull.sampling import minimum_sample_count, sample_assets


def asset(asset_id: str, scene: str, task: str) -> ManifestAsset:
    return ManifestAsset(asset_id=asset_id, scene=scene, task=task)


def test_minimum_sample_count_uses_ceiling() -> None:
    assert minimum_sample_count(1) == 1
    assert minimum_sample_count(100) == 1
    assert minimum_sample_count(101) == 2


def test_minimum_sample_count_uses_configurable_ratio() -> None:
    assert minimum_sample_count(200, sample_ratio=0.02) == 4
    assert minimum_sample_count(201, sample_ratio=0.005) == 2


def test_sampling_covers_all_scenes_even_when_over_one_percent() -> None:
    manifest = {
        "1": asset("1", "办公室", "阅读文件"),
        "2": asset("2", "厨房", "准备食材"),
        "3": asset("3", "工厂", "生产操作"),
    }

    rows = sample_assets(manifest, set(manifest), seed=7)

    assert {row.scene for row in rows} == {"办公室", "厨房", "工厂"}
    assert len(rows) == 3
    assert {row.reason for row in rows} == {"scene_coverage"}


def test_sampling_adds_task_diversity_until_minimum_met() -> None:
    manifest = {}
    for index in range(250):
        asset_id = str(index)
        scene = "办公室" if index < 200 else "厨房"
        task = f"task-{index % 5}"
        manifest[asset_id] = asset(asset_id, scene, task)

    rows = sample_assets(manifest, set(manifest), seed=11)

    assert len(rows) == 3
    assert {row.scene for row in rows} == {"办公室", "厨房"}
    assert len({row.task for row in rows}) >= 2


def test_sampling_uses_configurable_ratio_for_target_size() -> None:
    manifest = {
        str(index): asset(str(index), "办公室", f"task-{index}")
        for index in range(50)
    }

    rows = sample_assets(manifest, set(manifest), seed=17, sample_ratio=0.1)

    assert len(rows) == 5
