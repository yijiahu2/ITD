from __future__ import annotations

from pathlib import Path

import pytest

from ITD_agent.orchestration.runtime_support import remove_path, remove_vector_dataset


def test_remove_path_rejects_targets_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    blocked_root = tmp_path / "blocked"
    allowed_root.mkdir()
    blocked_root.mkdir()
    blocked_file = blocked_root / "artifact.txt"
    blocked_file.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside allowed roots"):
        remove_path(blocked_file, allowed_roots=[allowed_root])

    assert blocked_file.exists()


def test_remove_vector_dataset_removes_shapefile_sidecars_inside_allowed_roots(tmp_path: Path) -> None:
    dataset = tmp_path / "outputs" / "result.shp"
    dataset.parent.mkdir(parents=True)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        dataset.with_suffix(ext).write_text(ext, encoding="utf-8")

    removed = remove_vector_dataset(dataset, allowed_roots=[dataset.parent])

    assert sorted(removed) == sorted(str(dataset.with_suffix(ext)) for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"])
    assert not any(dataset.with_suffix(ext).exists() for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"])


def test_remove_vector_dataset_removes_single_non_shapefile_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "outputs" / "result.gpkg"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("gpkg", encoding="utf-8")

    removed = remove_vector_dataset(dataset, allowed_roots=[dataset.parent])

    assert removed == [str(dataset)]
    assert not dataset.exists()


def test_resolve_gateway_config_reads_bashrc_when_env_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    from ITD_agent.llm_gateway import gateway as gateway_mod

    home = tmp_path / "fake-home"
    home.mkdir()
    (home / ".bashrc").write_text(
        'export ARK_API_KEY="bashrc-key"\nexport ARK_MODEL="bashrc-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("ARK_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    gateway_mod._load_bashrc_exports.cache_clear()

    cfg = gateway_mod.resolve_gateway_config(runtime_cfg={"ITD_agent": {"llm_gateway": {}}})

    assert cfg.api_key == "bashrc-key"
    assert cfg.model == "bashrc-model"


def test_resolve_gateway_config_prefers_runtime_cfg_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    from ITD_agent.llm_gateway import gateway as gateway_mod

    monkeypatch.setenv("ARK_API_KEY", "env-key")
    monkeypatch.setenv("ARK_MODEL", "env-model")
    gateway_mod._load_bashrc_exports.cache_clear()

    cfg = gateway_mod.resolve_gateway_config(
        runtime_cfg={
            "ITD_agent": {
                "llm_gateway": {
                    "provider": "openai",
                    "api_key": "cfg-key",
                    "model": "cfg-model",
                    "base_url": "https://example.invalid/v1",
                }
            }
        }
    )

    assert cfg.provider == "openai"
    assert cfg.api_key == "cfg-key"
    assert cfg.model == "cfg-model"
    assert cfg.base_url == "https://example.invalid/v1"
