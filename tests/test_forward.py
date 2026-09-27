from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ai_investment_lab import forward


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_today_artifact(
    tmp_path: Path,
    *,
    artifact_id: str = "20260923T010203000000Z",
    proposal: dict[str, float] | None = None,
    allowed: dict[str, float] | None = None,
) -> Path:
    """Make a complete, independently valid Today artifact without an agent run."""

    source = tmp_path / "artifacts" / "today" / artifact_id
    source.mkdir(parents=True)
    proposal = proposal or {"AAPL": 0.70, "MSFT": 0.10, "NVDA": 0.10, "CASH": 0.10}
    allowed = allowed or {"AAPL": 0.25, "MSFT": 0.10, "NVDA": 0.10, "CASH": 0.55}
    analysis_timestamp = "2026-09-23T06:07:00+00:00"
    evidence_cutoff = "2026-09-21T16:00:00-04:00"
    fingerprint = "today-evidence-fingerprint"
    summary = {
        "artifact_kind": "today_analysis",
        "analysis_only": True,
        "order_submitted": False,
        "analysis_timestamp": analysis_timestamp,
        "evidence_as_of": evidence_cutoff,
        "evidence_fingerprint": fingerprint,
        "data_source": "YahooData",
        "lookback_days": 10,
        "universe": ["AAPL", "MSFT", "NVDA"],
        "benchmark": "SPY",
        "risk_limits": {"max_position_weight": 0.25, "max_gross_exposure": 1.0},
    }
    evidence = {
        "analysis_timestamp": analysis_timestamp,
        "evidence_as_of": evidence_cutoff,
        "evidence_fingerprint": fingerprint,
        "lookback_days": 10,
        "snapshot": {
            symbol: {"last_close": 100.0}
            for symbol in ("AAPL", "MSFT", "NVDA", "SPY")
        },
        "bar_timestamps": {
            symbol: evidence_cutoff for symbol in ("AAPL", "MSFT", "NVDA", "SPY")
        },
    }
    (source / "today_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    (source / "evidence.json").write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    (source / "portfolio_proposals.jsonl").write_text(
        json.dumps(
            {
                "as_of": analysis_timestamp,
                "aapl_weight": proposal["AAPL"],
                "msft_weight": proposal["MSFT"],
                "nvda_weight": proposal["NVDA"],
                "cash_weight": proposal["CASH"],
                "rationale": "A saved structured proposal for a forward-test fixture.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for symbol in ("AAPL", "MSFT", "NVDA", "CASH"):
        rows.append(
            {
                "as_of": analysis_timestamp,
                "symbol": symbol,
                "proposed_weight": proposal[symbol],
                "executed_weight": allowed[symbol],
                "overridden": proposal[symbol] != allowed[symbol],
                "reason": (
                    "max_position_25pct"
                    if symbol == "AAPL"
                    else "residual_after_risk_gate"
                    if symbol == "CASH"
                    else "approved"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(source / "risk_decisions.csv", index=False)
    # The regular Today completion marker includes saved agent summaries; Forward Test does not use them.
    (source / "agent_decisions.jsonl").write_text("", encoding="utf-8")
    return source


def _post_cutoff_prices(days: int = 2) -> dict[str, pd.Series]:
    cutoff = pd.Timestamp("2026-09-21T16:00:00-04:00")
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=days, freq="B")
    return {
        "AAPL": pd.Series([110.0 + index for index in range(days)], index=dates),
        "MSFT": pd.Series([90.0 + index for index in range(days)], index=dates),
        "NVDA": pd.Series([120.0 + index for index in range(days)], index=dates),
        "SPY": pd.Series([105.0 + index for index in range(days)], index=dates),
    }


def test_create_forward_test_freezes_today_decision_without_model_or_market_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_today_artifact(tmp_path)
    source_hashes = {
        path.name: _hash(path)
        for path in source.iterdir()
        if path.is_file()
    }
    calls: list[str] = []

    class UnexpectedYahoo:
        def __init__(self, **_kwargs: Any) -> None:
            calls.append("Yahoo")
            raise AssertionError("creation must not fetch market data")

    monkeypatch.setattr(forward, "YahooData", UnexpectedYahoo)

    created = forward.create_forward_test_from_today(source, project_root=tmp_path)

    assert created.reused is False
    assert created.artifact_dir.parent.name == "forward"
    assert calls == []
    assert forward._is_completed_forward_test(created.artifact_dir)
    assert source_hashes == {
        path.name: _hash(path)
        for path in source.iterdir()
        if path.is_file()
    }
    frozen = json.loads((created.artifact_dir / forward.FROZEN_DECISION_FILE).read_text())
    assert frozen["source_today_artifact_id"] == source.name
    assert frozen["proposal"] == {
        "AAPL": 0.7,
        "CASH": 0.1,
        "MSFT": 0.1,
        "NVDA": 0.1,
        "rationale": "A saved structured proposal for a forward-test fixture.",
    }
    assert frozen["allowed_allocation"] == {"AAPL": 0.25, "CASH": 0.55, "MSFT": 0.1, "NVDA": 0.1}
    assert frozen["cash_weight"] == pytest.approx(0.55)
    assert json.loads((created.artifact_dir / forward.FORWARD_PERFORMANCE_FILE).read_text())["status"] == (
        "waiting_for_future_data"
    )


def test_create_forward_test_reuses_matching_frozen_today_decision(tmp_path: Path) -> None:
    source = _write_today_artifact(tmp_path)

    first = forward.create_forward_test_from_today(source, project_root=tmp_path)
    second = forward.create_forward_test_from_today(source, project_root=tmp_path)

    assert first.reused is False
    assert second.reused is True
    assert second.artifact_dir == first.artifact_dir
    assert len(list((tmp_path / "artifacts" / "forward").iterdir())) == 1


def test_forward_creation_rejects_non_today_or_malformed_saved_decisions(tmp_path: Path) -> None:
    historical = tmp_path / "artifacts" / "runs" / "20260923T010203000000Z"
    historical.mkdir(parents=True)
    (historical / "run_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(forward.ForwardTestError, match="only be created from saved Today"):
        forward.create_forward_test_from_today(historical, project_root=tmp_path)

    source = _write_today_artifact(tmp_path)
    (source / "portfolio_proposals.jsonl").write_text(
        json.dumps(
            {
                "as_of": "2026-09-23T06:07:00+00:00",
                "aapl_weight": 0.8,
                "msft_weight": 0.2,
                "nvda_weight": 0.2,
                "cash_weight": 0.2,
                "rationale": "Weights do not sum to one.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(forward.ForwardTestError, match="invalid target weights"):
        forward.create_forward_test_from_today(source, project_root=tmp_path)

    altered_risk_source = _write_today_artifact(
        tmp_path,
        artifact_id="20260923T010204000000Z",
    )
    altered_risk = pd.read_csv(altered_risk_source / "risk_decisions.csv")
    altered_risk.loc[altered_risk["symbol"] == "AAPL", "executed_weight"] = 0.70
    altered_risk.to_csv(altered_risk_source / "risk_decisions.csv", index=False)
    with pytest.raises(forward.ForwardTestError, match="does not match the deterministic Risk Gate"):
        forward.create_forward_test_from_today(altered_risk_source, project_root=tmp_path)


def test_forward_refresh_uses_strict_evidence_cutoff_and_weighted_cash_portfolio(
    tmp_path: Path,
) -> None:
    source = _write_today_artifact(tmp_path)
    created = forward.create_forward_test_from_today(source, project_root=tmp_path)
    cutoff = pd.Timestamp("2026-09-21T16:00:00-04:00")
    # The first two timestamps identify the exact cutoff using different timezone notation.
    # Neither is eligible. Only the subsequent completed bar can be observed.
    prices = {
        "AAPL": pd.Series([99.0, 100.0, 110.0], index=[cutoff, cutoff.tz_convert("UTC"), cutoff + pd.Timedelta(days=1)]),
        "MSFT": pd.Series([99.0, 100.0, 90.0], index=[cutoff, cutoff.tz_convert("UTC"), cutoff + pd.Timedelta(days=1)]),
        "NVDA": pd.Series([99.0, 100.0, 120.0], index=[cutoff, cutoff.tz_convert("UTC"), cutoff + pd.Timedelta(days=1)]),
        "SPY": pd.Series([99.0, 100.0, 105.0], index=[cutoff, cutoff.tz_convert("UTC"), cutoff + pd.Timedelta(days=1)]),
    }

    forward.refresh_forward_test(
        created.artifact_dir,
        project_root=tmp_path,
        price_provider=lambda _frozen, **_kwargs: prices,
    )

    performance = json.loads((created.artifact_dir / forward.FORWARD_PERFORMANCE_FILE).read_text())
    assert performance["observed_trading_days"] == 1
    assert performance["observations"][0]["date"] == "2026-09-22"
    assert performance["current"]["portfolio_cumulative_return"] == pytest.approx(0.035)
    assert performance["current"]["spy_cumulative_return"] == pytest.approx(0.05)
    assert performance["current"]["relative_return"] == pytest.approx(-0.015)
    assert performance["horizons"]["1"]["status"] == "observed"
    assert performance["horizons"]["5"] == {"status": "pending", "trading_day": 5}
    assert performance["horizons"]["20"] == {"status": "pending", "trading_day": 20}


def test_forward_refresh_does_not_call_gemini_or_mutate_frozen_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_today_artifact(tmp_path)
    created = forward.create_forward_test_from_today(source, project_root=tmp_path)
    frozen_path = created.artifact_dir / forward.FROZEN_DECISION_FILE
    frozen_before = frozen_path.read_bytes()
    provider_calls: list[dict[str, Any]] = []

    def provider(frozen: dict[str, Any], **_kwargs: Any) -> dict[str, pd.Series]:
        provider_calls.append(frozen)
        return _post_cutoff_prices()

    # The module's native provider must not be touched when an explicit injected market source is used.
    monkeypatch.setattr(forward, "YahooData", lambda **_kwargs: pytest.fail("unexpected Yahoo call"))
    forward.refresh_forward_test(created.artifact_dir, project_root=tmp_path, price_provider=provider)

    assert len(provider_calls) == 1
    assert frozen_path.read_bytes() == frozen_before
    assert not any("GEMINI" in name for name in forward.__dict__)
    assert not (created.artifact_dir / "orders.csv").exists()
    assert not (created.artifact_dir / "trades.csv").exists()


def test_forward_refresh_is_deterministic_for_unchanged_market_data(tmp_path: Path) -> None:
    source = _write_today_artifact(tmp_path)
    created = forward.create_forward_test_from_today(source, project_root=tmp_path)
    prices = _post_cutoff_prices(days=2)
    provider = lambda _frozen, **_kwargs: prices

    forward.refresh_forward_test(created.artifact_dir, project_root=tmp_path, price_provider=provider)
    first = (created.artifact_dir / forward.FORWARD_PERFORMANCE_FILE).read_bytes()
    forward.refresh_forward_test(created.artifact_dir, project_root=tmp_path, price_provider=provider)
    second = (created.artifact_dir / forward.FORWARD_PERFORMANCE_FILE).read_bytes()

    assert first == second
    performance = json.loads(second)
    assert performance["observed_trading_days"] == 2
    assert performance["horizons"]["1"]["status"] == "observed"
    assert performance["horizons"]["5"]["status"] == "pending"


def test_empty_refresh_does_not_erase_previously_observed_outcomes(tmp_path: Path) -> None:
    source = _write_today_artifact(tmp_path)
    created = forward.create_forward_test_from_today(source, project_root=tmp_path)
    forward.refresh_forward_test(
        created.artifact_dir,
        project_root=tmp_path,
        price_provider=lambda _frozen, **_kwargs: _post_cutoff_prices(),
    )
    performance_path = created.artifact_dir / forward.FORWARD_PERFORMANCE_FILE
    observed = performance_path.read_bytes()
    cutoff_only = {
        symbol: pd.Series(
            [100.0],
            index=[pd.Timestamp("2026-09-21T16:00:00-04:00")],
        )
        for symbol in ("AAPL", "MSFT", "NVDA", "SPY")
    }

    forward.refresh_forward_test(
        created.artifact_dir,
        project_root=tmp_path,
        price_provider=lambda _frozen, **_kwargs: cutoff_only,
    )

    assert performance_path.read_bytes() == observed


def test_forward_artifacts_are_isolated_from_historical_and_today_pointers(tmp_path: Path) -> None:
    source = _write_today_artifact(tmp_path)
    historical = tmp_path / "artifacts" / "runs" / "20260922T010203000000Z"
    historical.mkdir(parents=True)
    (historical / "run_summary.json").write_text("{}", encoding="utf-8")
    historical_pointer = tmp_path / "artifacts" / "latest_run.json"
    today_pointer = tmp_path / "artifacts" / "latest_today.json"
    historical_pointer.write_text(json.dumps({"artifact_dir": str(historical)}), encoding="utf-8")
    today_pointer.write_text(json.dumps({"artifact_dir": str(source)}), encoding="utf-8")

    created = forward.create_forward_test_from_today(source, project_root=tmp_path)

    assert forward.find_latest_forward_test(project_root=tmp_path) == created.artifact_dir.resolve()
    assert json.loads(historical_pointer.read_text())["artifact_dir"] == str(historical)
    assert json.loads(today_pointer.read_text())["artifact_dir"] == str(source)
    summary = json.loads((created.artifact_dir / forward.FORWARD_SUMMARY_FILE).read_text())
    assert summary["source_today_artifact_id"] == source.name
    assert summary["order_submitted"] is False
    assert summary["rebalancing"] is False


def test_collect_forward_yahoo_prices_uses_completed_daily_path_and_excludes_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_today_artifact(tmp_path)
    created = forward.create_forward_test_from_today(source, project_root=tmp_path)
    frozen = json.loads((created.artifact_dir / forward.FROZEN_DECISION_FILE).read_text())
    created_with: list[dict[str, Any]] = []
    requests: list[tuple[str, int, str, Any]] = []
    cutoff = pd.Timestamp(frozen["evidence_as_of"])

    class FakeYahooData:
        def __init__(self, **kwargs: Any) -> None:
            created_with.append(kwargs)

        def get_historical_prices(
            self,
            symbol: str,
            length: int,
            timestep: str,
            *,
            timeshift: Any,
        ) -> SimpleNamespace:
            requests.append((symbol, length, timestep, timeshift))
            return SimpleNamespace(
                pandas_df=pd.DataFrame(
                    {"close": [100.0, 101.0]},
                    index=[cutoff, cutoff + pd.Timedelta(days=1)],
                )
            )

    monkeypatch.setattr(forward, "YahooData", FakeYahooData)
    now = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
    prices = forward.collect_forward_yahoo_prices(frozen, now=now)

    assert created_with == [
        {
            "auto_adjust": False,
            "datetime_start": now,
            "datetime_end": datetime(2026, 9, 27, 10, 1, tzinfo=UTC),
        }
    ]
    assert [request[0] for request in requests] == ["AAPL", "MSFT", "NVDA", "SPY"]
    assert all(request[2:] == ("day", None) for request in requests)
    assert all((series.index > cutoff.tz_convert("UTC")).all() for series in prices.values())
