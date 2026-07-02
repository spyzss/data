from pathlib import Path

from acceptance_pull.inventory import discover_hdf5, discover_video
from acceptance_pull.manifest import read_manifest
from acceptance_pull.validation import validate_ids
from tests.fixtures import write_manifest


def test_manifest_inventory_and_valid_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "XJGT_20260616.xlsx"
    write_manifest(
        manifest_path,
        [
            ("408817", "办公室", "阅读文件"),
            ("408820", "办公室", "整理文件"),
            ("408825", "厨房", "准备食材"),
        ],
    )
    hdf5_dir = tmp_path / "hdf5"
    video_dir = tmp_path / "video"
    hdf5_dir.mkdir()
    video_dir.mkdir()
    for asset_id in ["408817", "408820", "999999"]:
        (hdf5_dir / f"{asset_id}_hdf5.hdf5").write_bytes(b"h")
    for asset_id in ["408817", "408825"]:
        (video_dir / f"{asset_id}_video.mp4").write_bytes(b"v")

    manifest = read_manifest(manifest_path)
    result = validate_ids(manifest, discover_hdf5(hdf5_dir), discover_video(video_dir))

    assert sorted(result.valid_ids) == ["408817"]
    assert ("missing_hdf5", "408825") in result.issue_pairs()
    assert ("hdf5_only", "999999") in result.issue_pairs()
    assert ("missing_video", "408820") in result.issue_pairs()
