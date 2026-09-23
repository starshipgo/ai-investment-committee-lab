import math

import pytest

from ai_investment_lab.risk import apply_risk_gate


def test_position_cap_preserves_proposed_and_records_override() -> None:
    result = apply_risk_gate(
        {"AAPL": 0.6, "MSFT": 0.25, "NVDA": 0.15},
        ["AAPL", "MSFT", "NVDA"],
    )
    assert result.proposed_weights["AAPL"] == 0.6
    assert result.executed_weights["AAPL"] == 0.25
    assert "max_position_25pct" in result.reasons["AAPL"]
    assert result.executed_gross_exposure <= 1.0


def test_custom_position_cap_is_reflected_in_execution_and_reason() -> None:
    result = apply_risk_gate(
        {"AAPL": 1.0, "MSFT": 0.0, "NVDA": 0.0},
        ["AAPL", "MSFT", "NVDA"],
        max_position_weight=0.05,
    )
    assert result.proposed_weights["AAPL"] == 1.0
    assert result.executed_weights["AAPL"] == 0.05
    assert "max_position_5pct" in result.reasons["AAPL"]


def test_gross_cap_scales_positions_pro_rata() -> None:
    result = apply_risk_gate(
        {"AAPL": 0.6, "MSFT": 0.6, "NVDA": 0.6},
        ["AAPL", "MSFT", "NVDA"],
        max_position_weight=0.6,
        max_gross_exposure=1.0,
    )
    assert math.isclose(result.executed_gross_exposure, 1.0)
    assert all(math.isclose(weight, 1.0 / 3.0, abs_tol=1e-9) for weight in result.executed_weights.values())
    assert all("max_gross_100pct_pro_rata" in reasons for reasons in result.reasons.values())


def test_unknown_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the universe"):
        apply_risk_gate({"TSLA": 0.25}, ["AAPL", "MSFT", "NVDA"])
