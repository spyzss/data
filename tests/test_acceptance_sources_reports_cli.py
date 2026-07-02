import json
from datetime import date
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from acceptance_pull.config import PullConfig, SourceConfig, endpoint_from_region, parse_oss_uri
from acceptance_pull.cli import main
from acceptance_pull.models import IdIssue, ManifestAsset, PullResult, SampleRow, ValidationResult
from acceptance_pull.reports import write_reports
from acceptance_pull.sources import (
    LocalSource,
    OssBrowserSource,
    download_oss_batch_inputs,
    load_oss_browser2_current_session,
    pull_pairs,
)
from tests.fixtures import write_manifest


def test_load_local_config_defaults_workers_and_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "pull.yaml"
    config_path.write_text(
        """
manifest: manifest.xlsx
readme: README.txt
output: sampled
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
""",
        encoding="utf-8",
    )

    config = PullConfig.from_file(config_path)

    assert config.manifest == tmp_path / "manifest.xlsx"
    assert config.readme == tmp_path / "README.txt"
    assert config.output == tmp_path / "sampled"
    assert config.hdf5.kind == "local"
    assert config.hdf5.root == tmp_path / "hdf5"
    assert config.video.root == tmp_path / "video"
    assert config.workers == 8
    assert config.seed == int(date.today().strftime("%Y%m%d"))
    assert config.sample_ratio == 0.01


def test_endpoint_from_region_accepts_domestic_short_name() -> None:
    assert endpoint_from_region("beijing") == "https://oss-cn-beijing.aliyuncs.com"
    assert endpoint_from_region("cn-beijing") == "https://oss-cn-beijing.aliyuncs.com"
    assert endpoint_from_region("oss-cn-beijing") == "https://oss-cn-beijing.aliyuncs.com"


def test_parse_oss_uri_splits_bucket_and_prefix() -> None:
    bucket, prefix = parse_oss_uri("oss://xingjiguitu/egodata/XJGT_20260616/")

    assert bucket == "xingjiguitu"
    assert prefix == "egodata/XJGT_20260616"


def test_load_oss_batch_config_from_uri_and_region(tmp_path: Path) -> None:
    config_path = tmp_path / "pull.yaml"
    config_path.write_text(
        """
batch_uri: oss://xingjiguitu/egodata/XJGT_20260616
region: beijing
output: sampled
sample_ratio: 0.02
workers: 4
""",
        encoding="utf-8",
    )

    config = PullConfig.from_file(config_path)

    assert config.manifest is None
    assert config.readme is None
    assert config.batch_source is not None
    assert config.batch_source.endpoint == "https://oss-cn-beijing.aliyuncs.com"
    assert config.batch_source.bucket == "xingjiguitu"
    assert config.batch_source.prefix == "egodata/XJGT_20260616"
    assert config.hdf5.kind == "oss"
    assert config.hdf5.bucket == "xingjiguitu"
    assert config.hdf5.prefix == "egodata/XJGT_20260616/hdf5"
    assert config.video.prefix == "egodata/XJGT_20260616/video"
    assert config.seed == int(date.today().strftime("%Y%m%d"))
    assert config.sample_ratio == 0.02
    assert config.workers == 4


def test_load_config_accepts_sample_ratio(tmp_path: Path) -> None:
    config_path = tmp_path / "pull.yaml"
    config_path.write_text(
        """
manifest: manifest.xlsx
readme: README.txt
output: sampled
sample_ratio: 0.02
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
""",
        encoding="utf-8",
    )

    config = PullConfig.from_file(config_path)

    assert config.sample_ratio == 0.02


def test_config_rejects_invalid_sample_ratio(tmp_path: Path) -> None:
    config_path = tmp_path / "pull.yaml"
    config_path.write_text(
        """
manifest: manifest.xlsx
output: sampled
sample_ratio: 0
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_ratio must be > 0 and <= 1"):
        PullConfig.from_file(config_path)


def test_config_rejects_workers_below_one(tmp_path: Path) -> None:
    config_path = tmp_path / "pull.yaml"
    config_path.write_text(
        """
manifest: manifest.xlsx
output: sampled
workers: 0
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workers must be >= 1"):
        PullConfig.from_file(config_path)


def test_local_source_concurrent_pull(tmp_path: Path) -> None:
    hdf5_root = tmp_path / "hdf5-source"
    video_root = tmp_path / "video-source"
    output = tmp_path / "out"
    hdf5_root.mkdir()
    video_root.mkdir()
    (hdf5_root / "408817_hdf5.hdf5").write_bytes(b"hdf5")
    (video_root / "408817_video.mp4").write_bytes(b"video")

    rows = [SampleRow("408817", "办公室", "阅读文件", "scene_coverage")]
    results = pull_pairs(
        rows,
        LocalSource(hdf5_root, "hdf5"),
        LocalSource(video_root, "video"),
        output,
        workers=2,
    )

    assert (output / "hdf5" / "408817_hdf5.hdf5").read_bytes() == b"hdf5"
    assert (output / "video" / "408817_video.mp4").read_bytes() == b"video"
    assert {(result.file_type, result.status) for result in results} == {("hdf5", "success"), ("video", "success")}


def test_oss_browser_source_reports_sanitized_missing_login(tmp_path: Path) -> None:
    config = SourceConfig(
        kind="oss-browser2",
        endpoint="https://oss-cn.example.aliyuncs.com",
        bucket="bucket",
        prefix="data",
    )
    source = OssBrowserSource(config, "hdf5", app_support_dir=tmp_path)

    try:
        source.resolve("408817")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing login error")

    assert "oss-browser2 login context is unavailable" in message
    assert "currentSession" not in message


def test_oss_browser_session_loader_accepts_plain_json_session(tmp_path: Path) -> None:
    support_dir = tmp_path / "oss-browser2"
    support_dir.mkdir()
    (support_dir / "config.json").write_text(
        '{"currentSession": {"loginSession": {"options": {"accessKeyId": "id", "accessKeySecret": "secret", "stsToken": "token"}}}}',
        encoding="utf-8",
    )

    session = load_oss_browser2_current_session(support_dir)

    assert session["loginSession"]["options"]["accessKeyId"] == "id"


def test_oss_browser_session_loader_decrypts_hex_session(tmp_path: Path) -> None:
    support_dir = tmp_path / "oss-browser2"
    support_dir.mkdir()
    payload = '{"loginSession": {"options": {"accessKeyId": "id", "accessKeySecret": "secret", "stsToken": "token"}}}'
    key = b"NsGVeVW7BM7BBPqizlJv8fPz0hQFU4Dn"
    iv = bytes(16)
    pad_len = 16 - (len(payload.encode("utf-8")) % 16)
    padded = payload.encode("utf-8") + bytes([pad_len]) * pad_len
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    (support_dir / "config.json").write_text(
        json.dumps({"currentSession": encrypted.hex()}),
        encoding="utf-8",
    )

    session = load_oss_browser2_current_session(support_dir)

    assert session["loginSession"]["options"]["accessKeySecret"] == "secret"


def test_oss_browser_source_resolves_bucket_prefix_without_secret(tmp_path: Path) -> None:
    support_dir = tmp_path / "oss-browser2"
    support_dir.mkdir()
    (support_dir / "config.json").write_text(
        json.dumps(
            {
                "currentSession": {
                    "loginSession": {
                        "options": {
                            "accessKeyId": "id",
                            "accessKeySecret": "very-secret",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = SourceConfig(
        kind="oss-browser2",
        endpoint="https://oss-cn.example.aliyuncs.com",
        bucket="video-bucket",
        prefix="supplier/batch/video",
    )

    source = OssBrowserSource(config, "video", app_support_dir=support_dir)

    resolved = source.resolve("408817")
    assert resolved == "oss://video-bucket/supplier/batch/video/408817_video.mp4"
    assert "very-secret" not in resolved


def test_download_oss_batch_inputs_discovers_readme_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOssSource:
        def __init__(self, config: SourceConfig, file_type: str) -> None:
            self.config = config
            self.file_type = file_type

        def list_direct_object_keys(self, prefix: str) -> list[str]:
            assert prefix == "egodata/XJGT_20260616"
            return [
                "egodata/XJGT_20260616/README.txt",
                "egodata/XJGT_20260616/XJGT_20260616.xlsx",
            ]

        def download_key(self, key: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(key, encoding="utf-8")

    monkeypatch.setattr("acceptance_pull.sources.OssBrowserSource", FakeOssSource)
    config = SourceConfig(
        kind="oss",
        endpoint="https://oss-cn-beijing.aliyuncs.com",
        bucket="xingjiguitu",
        prefix="egodata/XJGT_20260616",
    )

    files = download_oss_batch_inputs(config, tmp_path)

    assert files.readme == tmp_path / "README.txt"
    assert files.manifest == tmp_path / "XJGT_20260616.xlsx"
    assert files.readme.read_text(encoding="utf-8").endswith("README.txt")
    assert files.manifest.read_text(encoding="utf-8").endswith("XJGT_20260616.xlsx")


def test_write_reports(tmp_path: Path) -> None:
    manifest = {"408817": ManifestAsset("408817", "办公室", "阅读文件")}
    validation = ValidationResult(
        valid_ids={"408817"},
        issues=[IdIssue("missing_video", "408820", "manifest and hdf5 exist")],
    )
    samples = [SampleRow("408817", "办公室", "阅读文件", "scene_coverage")]
    pulls = [
        PullResult(
            "408817",
            "hdf5",
            "/src/408817_hdf5.hdf5",
            tmp_path / "hdf5" / "408817_hdf5.hdf5",
            "success",
        )
    ]

    write_reports(
        output=tmp_path,
        manifest=manifest,
        validation=validation,
        samples=samples,
        pulls=pulls,
        minimum_count=1,
        seed=42,
        workers=8,
        sample_ratio=0.02,
    )

    assert (tmp_path / "reports" / "id_consistency.csv").read_text(encoding="utf-8").startswith(
        "issue_type,asset_id,detail"
    )
    sample_manifest = (tmp_path / "reports" / "sample_manifest.csv").read_text(encoding="utf-8")
    assert "hdf5_source" in sample_manifest
    assert "408817,办公室,阅读文件,scene_coverage,/src/408817_hdf5.hdf5" in sample_manifest
    summary = json.loads((tmp_path / "reports" / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid_id_count"] == 1
    assert summary["actual_sample_count"] == 1
    assert summary["scene_count"] == 1
    assert summary["workers"] == 8
    assert summary["sample_ratio"] == 0.02


def test_cli_local_smoke(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.xlsx"
    write_manifest(manifest, [("408817", "办公室", "阅读文件")])
    (tmp_path / "README.txt").write_text("total duration: 1h", encoding="utf-8")
    (tmp_path / "hdf5").mkdir()
    (tmp_path / "video").mkdir()
    (tmp_path / "hdf5" / "408817_hdf5.hdf5").write_bytes(b"h")
    (tmp_path / "video" / "408817_video.mp4").write_bytes(b"v")
    config = tmp_path / "pull.yaml"
    config.write_text(
        """
manifest: manifest.xlsx
readme: README.txt
output: out
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
""",
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config)])

    assert exit_code == 0
    assert (tmp_path / "out" / "hdf5" / "408817_hdf5.hdf5").exists()
    assert (tmp_path / "out" / "video" / "408817_video.mp4").exists()
    assert (tmp_path / "out" / "reports" / "summary.json").exists()


def test_cli_rejects_missing_readme(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.xlsx"
    write_manifest(manifest, [("408817", "办公室", "阅读文件")])
    (tmp_path / "hdf5").mkdir()
    (tmp_path / "video").mkdir()
    (tmp_path / "hdf5" / "408817_hdf5.hdf5").write_bytes(b"h")
    (tmp_path / "video" / "408817_video.mp4").write_bytes(b"v")
    config = tmp_path / "pull.yaml"
    config.write_text(
        """
manifest: manifest.xlsx
readme: README.txt
output: out
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="README"):
        main(["--config", str(config)])
