from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from lumibot.components.agents import AgentRunResult, AgentTraceEvent

from ai_investment_lab import today
from ai_investment_lab.strategy import ProposalRecoveryError
from ai_investment_lab.today import TodayAnalysisRun, TodayCommittee, TodayConfig, TodayEvidence

PROPOSAL_PAYLOAD = {
    "aapl_weight": 0.2,
    "msft_weight": 0.0,
    "nvda_weight": 0.8,
    "cash_weight": 0.0,
    "rationale": "Frozen evidence favors NVDA, with AAPL as the secondary holding.",
}


def make_evidence(lookback_days: int = 10) -> TodayEvidence:
    timestamp = "2026-09-22T00:00:00+00:00"
    snapshot = {
        symbol: {
            "bars": lookback_days,
            "last_close": 100.0,
            "return_over_window": 0.1,
            "annualized_volatility": 0.2,
        }
        for symbol in ("AAPL", "MSFT", "NVDA", "SPY")
    }
    return TodayEvidence(
        analysis_timestamp="2026-09-23T01:02:03+00:00",
        evidence_as_of=timestamp,
        lookback_days=lookback_days,
        snapshot=snapshot,
        bar_timestamps={symbol: timestamp for symbol in snapshot},
        fingerprint="frozen-evidence-fingerprint",
    )


def make_agent_result(
    *,
    cache_hit: bool = False,
    portfolio_manager: bool = False,
    include_proposal: bool = True,
) -> AgentRunResult:
    events: list[AgentTraceEvent] = []
    if portfolio_manager and include_proposal:
        events = [
            AgentTraceEvent(
                kind="tool_call",
                tool_name="record_target_weight_proposal",
                call_id="cached-proposal-call",
                payload=PROPOSAL_PAYLOAD,
            ),
            AgentTraceEvent(
                kind="tool_result",
                tool_name="record_target_weight_proposal",
                call_id="cached-proposal-call",
                payload={
                    "status": "proposal_recorded_not_executed",
                    "proposal": PROPOSAL_PAYLOAD,
                },
            ),
        ]
    return AgentRunResult(
        summary="Saved committee summary.",
        model="test-gemini-model",
        events=events,
        cache_hit=cache_hit,
    )


def make_committee_results(
    *,
    cache_hit: bool = False,
    include_proposal: bool = True,
) -> dict[str, AgentRunResult]:
    return {
        "Researcher": make_agent_result(cache_hit=cache_hit),
        "Bull Analyst": make_agent_result(cache_hit=cache_hit),
        "Bear Analyst": make_agent_result(cache_hit=cache_hit),
        "Portfolio Manager": make_agent_result(
            cache_hit=cache_hit,
            portfolio_manager=True,
            include_proposal=include_proposal,
        ),
    }


class StaticCommittee:
    def __init__(self, results: dict[str, AgentRunResult]) -> None:
        self._results = results

    def run(self) -> dict[str, AgentRunResult]:
        return self._results


@pytest.mark.parametrize(
    ("max_position_weight", "expected_assets", "expected_cash"),
    [
        (0.05, {"AAPL": 0.05, "MSFT": 0.0, "NVDA": 0.05}, 0.90),
        (0.25, {"AAPL": 0.20, "MSFT": 0.0, "NVDA": 0.25}, 0.55),
    ],
)
def test_today_analysis_recovers_cached_proposal_then_applies_current_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_position_weight: float,
    expected_assets: dict[str, float],
    expected_cash: float,
) -> None:
    """A replayed PM result must take the same recovery-and-risk path as a cold result."""

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("AI_MODEL", "test-gemini-model")
    historical_pointer = tmp_path / "artifacts" / "latest_run.json"
    historical_pointer.parent.mkdir(parents=True)
    historical_pointer.write_text('{"artifact_dir": "/historical/run"}\n', encoding="utf-8")

    evidence_calls: list[tuple[int, datetime | None]] = []

    def evidence_provider(lookback_days: int, analysis_time: datetime | None) -> TodayEvidence:
        evidence_calls.append((lookback_days, analysis_time))
        return make_evidence(lookback_days)

    cached_results = make_committee_results(cache_hit=True)
    committee_arguments: list[dict[str, Any]] = []

    def committee_factory(**kwargs: Any) -> StaticCommittee:
        committee_arguments.append(kwargs)
        return StaticCommittee(cached_results)

    run = today.run_today_analysis(
        TodayConfig(lookback_days=10, max_position_weight=max_position_weight),
        project_root=tmp_path,
        evidence_provider=evidence_provider,
        committee_factory=committee_factory,
        now=datetime(2026, 9, 23, 1, 2, 3, tzinfo=UTC),
    )

    assert isinstance(run, TodayAnalysisRun)
    assert run.artifact_dir.parent.name == "today"
    assert evidence_calls == [(10, datetime(2026, 9, 23, 1, 2, 3, tzinfo=UTC))]
    assert committee_arguments[0]["evidence"].lookback_days == 10
    assert today._is_completed_today_analysis(run.artifact_dir)
    assert today.find_latest_today_analysis(project_root=tmp_path) == run.artifact_dir.resolve()
    assert historical_pointer.read_text(encoding="utf-8") == '{"artifact_dir": "/historical/run"}\n'

    summary = json.loads((run.artifact_dir / "today_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_kind"] == "today_analysis"
    assert summary["analysis_only"] is True
    assert summary["order_submitted"] is False
    assert summary["lookback_days"] == 10
    assert summary["risk_limits"]["max_position_weight"] == max_position_weight
    assert not (run.artifact_dir / "metrics.json").exists()
    assert not (run.artifact_dir / "equity_curve.csv").exists()
    assert not (run.artifact_dir / "trades.csv").exists()
    assert not (run.artifact_dir / "approved_orders.csv").exists()

    risk = pd.read_csv(run.artifact_dir / "risk_decisions.csv").set_index("symbol")
    for symbol, expected_weight in expected_assets.items():
        assert float(risk.loc[symbol, "executed_weight"]) == pytest.approx(expected_weight)
    assert float(risk.loc["CASH", "executed_weight"]) == pytest.approx(expected_cash)
    assert float(risk.loc["NVDA", "proposed_weight"]) == pytest.approx(0.8)
    assert float(risk.loc["AAPL", "proposed_weight"]) == pytest.approx(0.2)

    proposal = json.loads(
        (run.artifact_dir / "portfolio_proposals.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert proposal["nvda_weight"] == 0.8
    assert proposal["aapl_weight"] == 0.2
    decisions = [
        json.loads(line)
        for line in (run.artifact_dir / "agent_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    portfolio_manager = next(decision for decision in decisions if decision["role"] == "Portfolio Manager")
    assert portfolio_manager["cache_hit"] is True


def test_today_analysis_fails_closed_without_a_recorded_portfolio_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def evidence_provider(lookback_days: int, _analysis_time: datetime | None) -> TodayEvidence:
        return make_evidence(lookback_days)

    def committee_factory(**_kwargs: Any) -> StaticCommittee:
        return StaticCommittee(make_committee_results(cache_hit=True, include_proposal=False))

    with pytest.raises(ProposalRecoveryError, match="exactly one target-weight proposal"):
        today.run_today_analysis(
            TodayConfig(lookback_days=10, max_position_weight=0.25),
            project_root=tmp_path,
            evidence_provider=evidence_provider,
            committee_factory=committee_factory,
        )

    incomplete_dirs = list((tmp_path / "artifacts" / "today").iterdir())
    assert len(incomplete_dirs) == 1
    assert not today._is_completed_today_analysis(incomplete_dirs[0])
    assert not (tmp_path / "artifacts" / "latest_today.json").exists()


def _write_completed_today_artifact(path: Path, *, artifact_kind: str = "today_analysis") -> None:
    path.mkdir(parents=True)
    summary = {
        "artifact_kind": artifact_kind,
        "order_submitted": False,
    }
    for filename in today.TODAY_COMPLETION_FILES:
        contents = json.dumps(summary) if filename == today.TODAY_SUMMARY_FILE else ""
        (path / filename).write_text(contents, encoding="utf-8")


def test_today_artifact_finder_rejects_historical_or_incomplete_artifacts(tmp_path: Path) -> None:
    historical = tmp_path / "artifacts" / "runs" / "20260923T010101000000Z"
    historical.mkdir(parents=True)
    (historical / "run_summary.json").write_text("{}", encoding="utf-8")
    wrong_kind = tmp_path / "artifacts" / "today" / "20260923T010102000000Z"
    _write_completed_today_artifact(wrong_kind, artifact_kind="historical_backtest")
    valid = tmp_path / "artifacts" / "today" / "20260923T010103000000Z"
    _write_completed_today_artifact(valid)
    (tmp_path / "artifacts" / "latest_today.json").write_text(
        json.dumps({"artifact_dir": str(historical)}), encoding="utf-8"
    )

    assert not today._is_completed_today_analysis(historical)
    assert not today._is_completed_today_analysis(wrong_kind)
    assert today._is_completed_today_analysis(valid)
    assert today.find_latest_today_analysis(project_root=tmp_path) == valid.resolve()


def test_collect_latest_yahoo_evidence_uses_requested_completed_daily_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_with: list[dict[str, Any]] = []
    price_requests: list[tuple[str, int, str, Any]] = []

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
            price_requests.append((symbol, length, timestep, timeshift))
            frame = pd.DataFrame(
                {"close": range(100, 112)},
                index=pd.date_range("2026-09-07", periods=12, freq="B"),
            )
            return SimpleNamespace(pandas_df=frame)

    monkeypatch.setattr(today, "YahooData", FakeYahooData)
    analysis_time = datetime(2026, 9, 23, 10, 30, tzinfo=UTC)

    evidence = today.collect_latest_yahoo_evidence(10, analysis_time)

    assert created_with == [
        {
            "auto_adjust": False,
            "datetime_start": analysis_time,
            "datetime_end": analysis_time + timedelta(minutes=1),
        }
    ]
    assert price_requests == [
        (symbol, 10, "day", None) for symbol in ("AAPL", "MSFT", "NVDA", "SPY")
    ]
    assert evidence.analysis_timestamp == analysis_time.isoformat()
    assert evidence.lookback_days == 10
    assert all(item["bars"] == 10 for item in evidence.snapshot.values())


def test_collect_latest_yahoo_evidence_maps_provider_errors_to_safe_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingYahooData:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def get_historical_prices(
            self,
            _symbol: str,
            _length: int,
            _timestep: str,
            *,
            timeshift: Any,
        ) -> SimpleNamespace:
            del timeshift
            raise RuntimeError("provider detail that must stay out of the UI")

    monkeypatch.setattr(today, "YahooData", FailingYahooData)

    with pytest.raises(today.TodayEvidenceError) as error:
        today.collect_latest_yahoo_evidence(10, datetime(2026, 9, 23, tzinfo=UTC))

    assert error.value.safe_category == "market_data"
    assert isinstance(error.value.__cause__, RuntimeError)


class RecordingAgent:
    def __init__(self, result: AgentRunResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> AgentRunResult:
        self.calls.append(kwargs)
        return self.result


class RecordingAgentManager:
    def __init__(self, results: dict[str, AgentRunResult]) -> None:
        self.created: list[dict[str, Any]] = []
        self._agents = {
            "researcher": RecordingAgent(results["Researcher"]),
            "bull_analyst": RecordingAgent(results["Bull Analyst"]),
            "bear_analyst": RecordingAgent(results["Bear Analyst"]),
            "portfolio_manager": RecordingAgent(results["Portfolio Manager"]),
        }

    def create(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def __getitem__(self, name: str) -> RecordingAgent:
        return self._agents[name]


def test_today_committee_keeps_policy_out_of_agent_contexts_and_has_no_order_api(
    tmp_path: Path,
) -> None:
    evidence = make_evidence(10)
    manager = RecordingAgentManager(make_committee_results())
    committee = TodayCommittee(
        evidence=evidence,
        artifact_dir=tmp_path,
        model="test-gemini-model",
        agent_manager_factory=lambda _host: manager,
    )

    results = committee.run()

    assert tuple(results) == ("Researcher", "Bull Analyst", "Bear Analyst", "Portfolio Manager")
    assert [item["name"] for item in manager.created] == [
        "researcher",
        "bull_analyst",
        "bear_analyst",
        "portfolio_manager",
    ]
    assert all(item["allow_trading"] is False for item in manager.created)
    assert all(item["include_builtin_tools"] is False for item in manager.created)
    assert all(item["include_builtin_skills"] is False for item in manager.created)
    assert len(manager.created[-1]["tools"]) == 1
    assert "max_position_weight" not in committee.parameters
    assert not hasattr(committee, "create_order")
    assert not hasattr(committee, "submit_order")
    assert committee.get_current_universe_snapshot() == evidence.tool_payload()

    calls = [call for agent in manager._agents.values() for call in agent.calls]
    assert len(calls) == 4
    assert all(call["context"]["lookback_days"] == 10 for call in calls)
    assert all("max_position_weight" not in call["context"] for call in calls)


def test_today_committee_binds_only_its_intended_local_tools(tmp_path: Path) -> None:
    """Concrete AgentManager must not add built-ins to the Today tool surfaces."""

    committee = TodayCommittee(
        evidence=make_evidence(10),
        artifact_dir=tmp_path,
        model="test-gemini-model",
    )

    committee._create_agents()

    bound_tool_names = {
        name: [tool.name for tool in committee.agents[name]._ensure_bound_tools()]
        for name in ("researcher", "bull_analyst", "bear_analyst", "portfolio_manager")
    }
    assert bound_tool_names == {
        "researcher": ["get_current_universe_snapshot"],
        "bull_analyst": [],
        "bear_analyst": [],
        "portfolio_manager": ["record_target_weight_proposal"],
    }
