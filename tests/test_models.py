import pytest
from pydantic import ValidationError

from ai_investment_lab.models import TargetWeightProposal


def test_target_weight_proposal_requires_weights_to_sum_to_one() -> None:
    proposal = TargetWeightProposal(
        aapl_weight=0.4,
        msft_weight=0.3,
        nvda_weight=0.2,
        cash_weight=0.1,
        rationale="Test",
    )
    assert proposal.asset_weights() == {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.2}


def test_target_weight_proposal_rejects_bad_total() -> None:
    with pytest.raises(ValidationError):
        TargetWeightProposal(
            aapl_weight=0.4,
            msft_weight=0.4,
            nvda_weight=0.4,
            cash_weight=0.0,
            rationale="Invalid",
        )
