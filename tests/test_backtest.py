import json
from pathlib import Path

import pytest

from ai_investment_lab import backtest


def test_runner_passes_selected_parameters_and_saves_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_run_backtest(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return None, object()

    def fake_export_performance(_strategy, artifact_dir: Path, initial_budget: float):
        metrics = {
            "initial_equity": initial_budget,
            "final_equity": initial_budget,
            "total_return": 0.0,
            "benchmark_total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "trading_days": 1,
        }
        (artifact_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return metrics

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(backtest.AIInvestmentCommitteeStrategy, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(backtest, "export_performance", fake_export_performance)
    config = backtest.ExperimentConfig(lookback_days=73, max_position_weight=0.05)

    artifact_dir = backtest.run_experiment(config, project_root=tmp_path)

    parameters = captured["kwargs"]["parameters"]
    assert parameters["lookback_days"] == 73
    assert parameters["max_position_weight"] == 0.05
    assert parameters["max_gross_exposure"] == 1.0
    summary = json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"]["lookback_days"] == 73
    assert summary["experiment"]["max_position_weight"] == 0.05
    assert summary["risk_limits"]["max_position_weight"] == 0.05


def test_matching_completed_config_is_reused_without_running_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = backtest.ExperimentConfig(lookback_days=10, max_position_weight=0.50)
    artifact_dir = tmp_path / "artifacts" / "runs" / "20260101T000000000000Z"
    artifact_dir.mkdir(parents=True)
    summary = {
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "risk_limits": {
            "max_position_weight": config.max_position_weight,
            "max_gross_exposure": config.max_gross_exposure,
        },
        "experiment": config.artifact_config(),
    }
    for filename in backtest.COMPLETION_FILES:
        contents = json.dumps(summary) if filename == "run_summary.json" else ""
        (artifact_dir / filename).write_text(contents, encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        pytest.fail("a matching completed run should not invoke the model-backed runner")

    monkeypatch.setattr(backtest, "run_experiment", fail_if_called)

    result = backtest.run_or_reuse_experiment(config, project_root=tmp_path)

    assert result.reused is True
    assert result.artifact_dir == artifact_dir.resolve()


def test_legacy_default_run_is_reused_despite_first_day_equity_move(tmp_path: Path) -> None:
    config = backtest.ExperimentConfig()
    artifact_dir = tmp_path / "artifacts" / "runs" / "20250916T000000Z"
    artifact_dir.mkdir(parents=True)
    legacy_summary = {
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "risk_limits": {
            "max_position_weight": config.max_position_weight,
            "max_gross_exposure": config.max_gross_exposure,
        },
        "metrics": {"initial_equity": 100_347.49},
    }
    for filename in backtest.COMPLETION_FILES:
        contents = json.dumps(legacy_summary) if filename == "run_summary.json" else ""
        (artifact_dir / filename).write_text(contents, encoding="utf-8")

    assert backtest.find_completed_experiment(config, project_root=tmp_path) == artifact_dir.resolve()
