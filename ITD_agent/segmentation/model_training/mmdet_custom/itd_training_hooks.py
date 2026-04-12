from __future__ import annotations

import json
from pathlib import Path

from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ITDTrainingTraceHook(Hook):
    def __init__(self, injection_manifest_path: str, summary_interval: int = 200) -> None:
        self.injection_manifest_path = str(injection_manifest_path)
        self.summary_interval = int(summary_interval)
        self._payload = {}

    def before_train(self, runner) -> None:
        path = Path(self.injection_manifest_path)
        if path.exists():
            try:
                self._payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._payload = {}
        runner.message_hub.update_info("itd_expert_injection", self._payload)
        trace_path = Path(runner.work_dir) / "itd_expert_injection_runtime.json"
        trace_path.write_text(json.dumps(self._payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if self.summary_interval <= 0:
            return
        if (batch_idx + 1) % self.summary_interval == 0:
            runner.logger.info(
                "ITD expert injection | family=%s | wrapper=%s | curriculum=%s",
                self._payload.get("target_expert_family"),
                (self._payload.get("dataset_wrapper") or {}).get("type"),
                self._payload.get("curriculum_mode"),
            )
