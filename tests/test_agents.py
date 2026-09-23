from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_investment_lab.models import TargetWeightProposal
from ai_investment_lab.strategy import AIInvestmentCommitteeStrategy


class RecordingAgentManager:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create(self, **kwargs):
        self.created.append(kwargs)


def test_exactly_four_agents_are_non_trading(tmp_path: Path) -> None:
    strategy = object.__new__(AIInvestmentCommitteeStrategy)
    strategy.parameters = {
        **AIInvestmentCommitteeStrategy.parameters,
        "artifact_dir": str(tmp_path),
    }
    strategy.agents = RecordingAgentManager()

    strategy.initialize()

    assert [item["name"] for item in strategy.agents.created] == [
        "researcher",
        "bull_analyst",
        "bear_analyst",
        "portfolio_manager",
    ]
    assert all(item["allow_trading"] is False for item in strategy.agents.created)
    portfolio_manager = strategy.agents.created[-1]
    assert len(portfolio_manager["tools"]) == 1
    assert portfolio_manager["include_builtin_tools"] is False


def test_configured_lookback_reaches_researcher_snapshot_tool() -> None:
    strategy = object.__new__(AIInvestmentCommitteeStrategy)
    strategy.parameters = {
        **AIInvestmentCommitteeStrategy.parameters,
        "lookback_days": 120,
    }
    calls: list[tuple[str, int, str, int]] = []

    def get_historical_prices(symbol: str, length: int, timestep: str, *, timeshift: int):
        calls.append((symbol, length, timestep, timeshift))
        return SimpleNamespace(pandas_df=pd.DataFrame({"close": range(100, 220)}))

    strategy.get_historical_prices = get_historical_prices
    strategy.get_datetime = lambda: datetime(2025, 1, 2, tzinfo=UTC)

    snapshot = strategy.get_universe_snapshot()

    assert [call[0] for call in calls] == ["AAPL", "MSFT", "NVDA", "SPY"]
    assert all(call[1:] == (120, "day", 1) for call in calls)
    assert snapshot["snapshot"]["AAPL"]["bars"] == 120


def test_configured_position_cap_reaches_deterministic_risk_gate() -> None:
    strategy = object.__new__(AIInvestmentCommitteeStrategy)
    strategy.parameters = {
        **AIInvestmentCommitteeStrategy.parameters,
        "max_position_weight": 0.05,
    }
    proposal = TargetWeightProposal(
        aapl_weight=1.0,
        msft_weight=0.0,
        nvda_weight=0.0,
        cash_weight=0.0,
        rationale="Test only the deterministic cap.",
    )

    result = strategy._apply_risk_gate(proposal)

    assert result.proposed_weights["AAPL"] == 1.0
    assert result.executed_weights["AAPL"] == 0.05
    assert result.max_position_weight == 0.05
    assert "max_position_5pct" in result.reasons["AAPL"]
