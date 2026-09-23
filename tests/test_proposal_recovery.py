import pytest
from lumibot.components.agents import AgentRunResult, AgentTraceEvent

from ai_investment_lab.models import TargetWeightProposal
from ai_investment_lab.strategy import (
    AIInvestmentCommitteeStrategy,
    ProposalRecoveryError,
    recover_recorded_proposal,
)

PROPOSAL_PAYLOAD = {
    "aapl_weight": 0.2,
    "msft_weight": 0.0,
    "nvda_weight": 0.8,
    "cash_weight": 0.0,
    "rationale": "NVDA leads the short lookback while AAPL adds resilience.",
}


def portfolio_manager_result(
    *,
    cache_hit: bool,
    call_payload: dict | None = None,
    result_payload: dict | None = None,
    include_call: bool = True,
    include_result: bool = True,
) -> AgentRunResult:
    """Build the same structural record LumiBot returns for a PM tool invocation."""

    call_id = "proposal-call-1"
    events = []
    if include_call:
        events.append(
            AgentTraceEvent(
                kind="tool_call",
                tool_name="record_target_weight_proposal",
                call_id=call_id,
                payload=call_payload or PROPOSAL_PAYLOAD,
            )
        )
    if include_result:
        events.append(
            AgentTraceEvent(
                kind="tool_result",
                tool_name="record_target_weight_proposal",
                call_id=call_id,
                payload=result_payload
                or {
                    "status": "proposal_recorded_not_executed",
                    "proposal": call_payload or PROPOSAL_PAYLOAD,
                },
            )
        )
    return AgentRunResult(
        summary="Recorded a target-weight proposal.",
        model="gemini-3.5-flash-lite",
        events=events,
        cache_hit=cache_hit,
    )


def test_cold_portfolio_manager_proposal_uses_recorded_agent_result() -> None:
    result = portfolio_manager_result(cache_hit=False)

    proposal = recover_recorded_proposal(result)

    assert proposal == TargetWeightProposal.model_validate(PROPOSAL_PAYLOAD)


def test_cached_portfolio_manager_proposal_needs_no_in_memory_tool_side_effect() -> None:
    result = portfolio_manager_result(cache_hit=True)

    proposal = recover_recorded_proposal(result)

    assert result.cache_hit is True
    assert proposal.nvda_weight == 0.8
    assert proposal.aapl_weight == 0.2


def test_missing_portfolio_manager_proposal_fails_closed() -> None:
    with pytest.raises(ProposalRecoveryError, match="exactly one target-weight proposal"):
        recover_recorded_proposal(portfolio_manager_result(cache_hit=True, include_call=False))


def test_malformed_portfolio_manager_proposal_fails_closed() -> None:
    malformed = {**PROPOSAL_PAYLOAD, "nvda_weight": 0.7}

    with pytest.raises(ProposalRecoveryError, match="audit record is invalid"):
        recover_recorded_proposal(
            portfolio_manager_result(
                cache_hit=True,
                call_payload=malformed,
                result_payload={
                    "status": "proposal_recorded_not_executed",
                    "proposal": malformed,
                },
            )
        )


@pytest.mark.parametrize(
    ("max_position_weight", "expected_assets", "expected_cash"),
    [
        (0.05, {"AAPL": 0.05, "MSFT": 0.0, "NVDA": 0.05}, 0.9),
        (0.25, {"AAPL": 0.2, "MSFT": 0.0, "NVDA": 0.25}, 0.55),
    ],
)
def test_cached_proposal_is_re_evaluated_by_each_deterministic_policy(
    max_position_weight: float,
    expected_assets: dict[str, float],
    expected_cash: float,
) -> None:
    strategy = object.__new__(AIInvestmentCommitteeStrategy)
    strategy.parameters = {
        **AIInvestmentCommitteeStrategy.parameters,
        "max_position_weight": max_position_weight,
    }
    proposal = recover_recorded_proposal(portfolio_manager_result(cache_hit=True))

    risk_result = strategy._apply_risk_gate(proposal)

    assert risk_result.executed_weights == expected_assets
    assert 1.0 - risk_result.executed_gross_exposure == pytest.approx(expected_cash)
