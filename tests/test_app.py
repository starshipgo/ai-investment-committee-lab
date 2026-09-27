import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SPEC = importlib.util.spec_from_file_location("ai_investment_lab_app", APP_PATH)
if APP_SPEC is None or APP_SPEC.loader is None:
    raise RuntimeError("Could not load the Streamlit app module for testing")
APP = importlib.util.module_from_spec(APP_SPEC)
sys.modules[APP_SPEC.name] = APP
APP_SPEC.loader.exec_module(APP)

ExperimentSettings = APP.ExperimentSettings
TodaySettings = APP.TodaySettings
run_experiment_if_requested = APP.run_experiment_if_requested
run_today_if_requested = APP.run_today_if_requested
create_forward_test_if_requested = APP.create_forward_test_if_requested
refresh_forward_test_if_requested = APP.refresh_forward_test_if_requested
safe_experiment_failure = APP.safe_experiment_failure


def test_page_load_gate_does_not_call_the_experiment_runner() -> None:
    calls: list[ExperimentSettings] = []
    settings = ExperimentSettings(lookback_days=20, max_position_weight=0.25)

    def runner(received_settings: ExperimentSettings) -> str:
        calls.append(received_settings)
        return "unexpected"

    result = run_experiment_if_requested(False, settings, runner=runner)

    assert result is None
    assert calls == []


def test_explicit_run_request_passes_both_interactive_settings() -> None:
    settings = ExperimentSettings(lookback_days=120, max_position_weight=0.05)

    result = run_experiment_if_requested(
        True,
        settings,
        runner=lambda received_settings: received_settings,
    )

    assert result == settings


def test_today_page_load_gate_does_not_call_the_analysis_runner() -> None:
    calls: list[TodaySettings] = []
    settings = TodaySettings(lookback_days=20, max_position_weight=0.25)

    def runner(received_settings: TodaySettings) -> str:
        calls.append(received_settings)
        return "unexpected"

    result = run_today_if_requested(False, settings, runner=runner)

    assert result is None
    assert calls == []


def test_explicit_today_request_passes_both_interactive_settings() -> None:
    settings = TodaySettings(lookback_days=10, max_position_weight=0.05)

    result = run_today_if_requested(
        True,
        settings,
        runner=lambda received_settings: received_settings,
    )

    assert result == settings


def test_today_page_load_gate_does_not_create_a_forward_test() -> None:
    calls: list[Path] = []
    today_dir = Path("/saved/today")

    result = create_forward_test_if_requested(
        False,
        today_dir,
        runner=lambda received_dir: calls.append(received_dir),
    )

    assert result is None
    assert calls == []


def test_explicit_forward_creation_passes_only_the_selected_today_artifact() -> None:
    today_dir = Path("/saved/today")

    result = create_forward_test_if_requested(
        True,
        today_dir,
        runner=lambda received_dir: received_dir,
    )

    assert result == today_dir


def test_forward_page_load_gate_does_not_refresh_market_data() -> None:
    calls: list[Path] = []
    forward_dir = Path("/saved/forward")

    result = refresh_forward_test_if_requested(
        False,
        forward_dir,
        runner=lambda received_dir: calls.append(received_dir),
    )

    assert result is None
    assert calls == []


def test_explicit_forward_refresh_passes_only_the_selected_forward_artifact() -> None:
    forward_dir = Path("/saved/forward")

    result = refresh_forward_test_if_requested(
        True,
        forward_dir,
        runner=lambda received_dir: received_dir,
    )

    assert result == forward_dir


def test_lookback_display_prefers_saved_experiment_configuration() -> None:
    rendered = APP.find_lookback_window(
        {"experiment": {"lookback_days": 10}},
        {"Researcher": {"summary": "The legacy text mentions a 20-bar lookback."}},
    )

    assert rendered == "10 daily bars"


def test_lookback_display_falls_back_for_legacy_artifacts() -> None:
    rendered = APP.find_lookback_window(
        {"experiment": None},
        {"Researcher": {"summary": "Evidence uses 20 completed daily bars."}},
    )

    assert rendered == "20 daily bars"


def test_proposal_recovery_failure_has_a_safe_specific_ui_message() -> None:
    class ProposalFailure(RuntimeError):
        safe_category = "portfolio_proposal"

    category, message = safe_experiment_failure(ProposalFailure("internal details"))

    assert category == "Portfolio proposal rejected"
    assert "valid recorded target-weight proposal" in message
    assert "internal details" not in message


def test_unknown_failure_never_exposes_raw_error_text() -> None:
    fake_secret = "do-not-display-this-api-key"

    category, message = safe_experiment_failure(RuntimeError(fake_secret))

    assert category == "Experiment did not complete"
    assert fake_secret not in message


def test_today_market_data_failure_has_a_safe_specific_ui_message() -> None:
    class MarketDataFailure(RuntimeError):
        safe_category = "market_data"

    category, message = safe_experiment_failure(MarketDataFailure("provider details"))

    assert category == "Current market data unavailable"
    assert "provider details" not in message


def test_today_presenter_reads_only_today_artifacts_without_historical_metrics(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "artifacts" / "today" / "20260923T010203000000Z"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "today_summary.json").write_text(
        json.dumps(
            {
                "artifact_kind": "today_analysis",
                "data_source": "YahooData",
                "analysis_timestamp": "2026-09-23T01:02:03+00:00",
                "evidence_as_of": "2026-09-22T00:00:00+00:00",
                "lookback_days": 10,
                "risk_limits": {
                    "max_position_weight": 0.25,
                    "max_gross_exposure": 1.0,
                },
                "analysis_only": True,
                "order_submitted": False,
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "evidence.json").write_text(
        json.dumps({"lookback_days": 10}), encoding="utf-8"
    )
    (analysis_dir / "agent_decisions.jsonl").write_text(
        json.dumps({"role": "Researcher", "summary": "Saved evidence."}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "proposed_weight": 0.2,
                "executed_weight": 0.2,
                "overridden": False,
                "reason": "approved",
            },
            {
                "symbol": "CASH",
                "proposed_weight": 0.0,
                "executed_weight": 0.8,
                "overridden": True,
                "reason": "residual_after_risk_gate",
            },
        ]
    ).to_csv(analysis_dir / "risk_decisions.csv", index=False)

    analysis = APP.load_today_analysis(str(analysis_dir))
    view = APP.build_today_presenter_data(analysis_dir, analysis)

    assert view.lookback_window == "10 completed daily bars"
    assert view.cash_residual == pytest.approx(0.8)
    assert not hasattr(view, "total_return")
    assert not hasattr(view, "equity_chart")


def test_historical_latest_finder_never_selects_a_today_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    historical = artifacts_root / "runs" / "20260922T010203000000Z"
    historical.mkdir(parents=True)
    (historical / "run_summary.json").write_text("{}", encoding="utf-8")
    today = artifacts_root / "today" / "20260923T010203000000Z"
    today.mkdir(parents=True)
    (today / "today_summary.json").write_text(
        json.dumps({"artifact_kind": "today_analysis", "order_submitted": False}),
        encoding="utf-8",
    )
    (artifacts_root / "latest_today.json").write_text(
        json.dumps({"artifact_dir": str(today)}), encoding="utf-8"
    )
    monkeypatch.setattr(APP, "ARTIFACTS_ROOT", artifacts_root)

    assert APP._safe_run_dir() == historical


def test_forward_presenter_reads_only_frozen_forward_artifact(tmp_path: Path) -> None:
    forward_dir = tmp_path / "artifacts" / "forward" / "20260924T010203000000Z"
    forward_dir.mkdir(parents=True)
    (forward_dir / "forward_summary.json").write_text(
        json.dumps(
            {
                "artifact_kind": "forward_test",
                "analysis_only": True,
                "order_submitted": False,
                "data_source": "YahooData",
            }
        ),
        encoding="utf-8",
    )
    (forward_dir / "frozen_decision.json").write_text(
        json.dumps(
            {
                "artifact_kind": "forward_test_frozen_decision",
                "data_source": "YahooData",
                "original_analysis_timestamp": "2026-09-23T01:02:03+00:00",
                "evidence_as_of": "2026-09-22T00:00:00+00:00",
                "lookback_days": 10,
                "risk_limits": {"max_position_weight": 0.25, "max_gross_exposure": 1.0},
                "allowed_allocation": {"AAPL": 0.25, "MSFT": 0.1, "NVDA": 0.1, "CASH": 0.55},
                "risk_decisions": [
                    {
                        "symbol": "AAPL",
                        "proposed_weight": 0.7,
                        "executed_weight": 0.25,
                        "overridden": True,
                        "reason": "max_position_25pct",
                    },
                    {
                        "symbol": "CASH",
                        "proposed_weight": 0.1,
                        "executed_weight": 0.55,
                        "overridden": True,
                        "reason": "residual_after_risk_gate",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (forward_dir / "forward_performance.json").write_text(
        json.dumps(
            {
                "observed_trading_days": 2,
                "current": {
                    "date": "2026-09-24",
                    "portfolio_cumulative_return": 0.02,
                    "spy_cumulative_return": 0.01,
                    "relative_return": 0.01,
                },
                "horizons": {
                    "1": {
                        "status": "observed",
                        "trading_day": 1,
                        "date": "2026-09-23",
                        "portfolio_cumulative_return": 0.01,
                        "spy_cumulative_return": 0.0,
                        "relative_return": 0.01,
                    },
                    "5": {"status": "pending", "trading_day": 5},
                    "20": {"status": "pending", "trading_day": 20},
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = APP.load_forward_test(str(forward_dir), (1, 1, 1))
    view = APP.build_forward_presenter_data(forward_dir, loaded)

    assert view.lookback_window == "10 completed daily bars"
    assert view.cash_residual == pytest.approx(0.55)
    assert view.observed_trading_days == 2
    assert view.current_relative_return == pytest.approx(0.01)
    assert view.horizons[0].status == "observed"
    assert view.horizons[1].status == "pending"
    assert not hasattr(view, "total_return")
    assert not hasattr(view, "agent_cards")


def test_forward_finder_rejects_today_and_historical_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    today = artifacts_root / "today" / "20260923T010203000000Z"
    historical = artifacts_root / "runs" / "20260922T010203000000Z"
    today.mkdir(parents=True)
    historical.mkdir(parents=True)
    (today / "forward_summary.json").write_text(
        json.dumps({"artifact_kind": "forward_test", "analysis_only": True, "order_submitted": False}),
        encoding="utf-8",
    )
    (today / "frozen_decision.json").write_text(
        json.dumps({"artifact_kind": "forward_test_frozen_decision"}), encoding="utf-8"
    )
    (historical / "forward_summary.json").write_text(
        json.dumps({"artifact_kind": "forward_test", "analysis_only": True, "order_submitted": False}),
        encoding="utf-8",
    )
    (historical / "frozen_decision.json").write_text(
        json.dumps({"artifact_kind": "forward_test_frozen_decision"}), encoding="utf-8"
    )
    monkeypatch.setattr(APP, "ARTIFACTS_ROOT", artifacts_root)

    assert not APP._is_forward_child(today)
    assert not APP._is_forward_child(historical)
    with pytest.raises(FileNotFoundError):
        APP._safe_forward_dir()
