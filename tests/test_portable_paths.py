from __future__ import annotations

import re
from pathlib import Path

from project_paths import PROJECT_ROOT, configured_path


def test_relative_environment_override_is_repository_anchored() -> None:
    result = configured_path(
        "UAV_TEST_PATH",
        PROJECT_ROOT / "default",
        {"UAV_TEST_PATH": "external/resources"},
    )

    assert result == (PROJECT_ROOT / "external" / "resources").resolve(strict=False)


def test_default_is_independent_of_current_working_directory(tmp_path: Path) -> None:
    default = PROJECT_ROOT / "models" / "model.pt"
    result = configured_path("UAV_TEST_PATH", default, {})

    assert result == default.resolve(strict=False)
    assert tmp_path not in result.parents


def test_active_code_and_configs_have_no_username_specific_absolute_paths() -> None:
    scopes = [
        PROJECT_ROOT / "01_WINDOWS_AI",
        PROJECT_ROOT / "02_ROS2_WINDOWS_MIRROR",
        PROJECT_ROOT / "04_DATASET_ENGINEERING",
        PROJECT_ROOT / "05_TRAINING" / "scripts",
        PROJECT_ROOT / "05_TRAINING" / "configs",
        PROJECT_ROOT / "06_AGENTIC_AUTONOMY",
        PROJECT_ROOT / "08_MODEL_UNCERTAINTY" / "src",
    ]
    suffixes = {".py", ".ps1", ".bat", ".cmd", ".sh", ".yaml", ".yml", ".json", ".toml"}
    unix_home = "/" + "home" + r"/[^/\s]+/"
    hosted_notebook = "/" + "content" + "/"
    forbidden = re.compile(
        rf"(?i)([A-Z]:[\\/](?:Users|home)[\\/]|{unix_home}|{hosted_notebook}|OneDrive[\\/])"
    )
    offenders = []

    for scope in scopes:
        for path in scope.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if "_BASELINE" in path.name:
                continue
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            if forbidden.search(text):
                offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == []
