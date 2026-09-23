"""Deterministic, model-independent target-weight risk controls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskGateResult:
    proposed_weights: dict[str, float]
    executed_weights: dict[str, float]
    reasons: dict[str, list[str]]
    max_position_weight: float
    max_gross_exposure: float

    @property
    def proposed_gross_exposure(self) -> float:
        return sum(abs(weight) for weight in self.proposed_weights.values())

    @property
    def executed_gross_exposure(self) -> float:
        return sum(abs(weight) for weight in self.executed_weights.values())


def apply_risk_gate(
    proposed_weights: Mapping[str, float],
    universe: Sequence[str],
    *,
    max_position_weight: float = 0.25,
    max_gross_exposure: float = 1.0,
) -> RiskGateResult:
    """Apply long-only position and portfolio caps in a deterministic order.

    Invalid and negative weights become zero, each position is clipped, and the
    resulting portfolio is scaled pro rata if gross exposure still exceeds the
    portfolio limit. Unknown symbols are rejected instead of silently traded.
    """

    if not 0.0 < max_position_weight <= 1.0:
        raise ValueError("max_position_weight must be in (0, 1]")
    if not 0.0 < max_gross_exposure <= 1.0:
        raise ValueError("max_gross_exposure must be in (0, 1]")

    allowed = tuple(symbol.upper() for symbol in universe)
    unknown = {symbol.upper() for symbol in proposed_weights} - set(allowed)
    if unknown:
        raise ValueError(f"proposal contains symbols outside the universe: {sorted(unknown)}")

    proposed: dict[str, float] = {}
    executed: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    for symbol in allowed:
        raw = proposed_weights.get(symbol, 0.0)
        symbol_reasons: list[str] = []
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            weight = 0.0
            symbol_reasons.append("invalid_weight_zeroed")
        if not math.isfinite(weight):
            weight = 0.0
            symbol_reasons.append("non_finite_weight_zeroed")

        proposed[symbol] = weight
        if weight < 0.0:
            approved = 0.0
            symbol_reasons.append("long_only_floor")
        else:
            approved = min(weight, max_position_weight)
            if approved < weight:
                symbol_reasons.append(f"max_position_{max_position_weight:.0%}".replace("%", "pct"))

        executed[symbol] = approved
        reasons[symbol] = symbol_reasons

    gross = sum(abs(weight) for weight in executed.values())
    if gross > max_gross_exposure:
        scale = max_gross_exposure / gross
        for symbol in allowed:
            executed[symbol] *= scale
            reasons[symbol].append(
                f"max_gross_{max_gross_exposure:.0%}_pro_rata".replace("%", "pct")
            )

    executed = {symbol: round(weight, 10) for symbol, weight in executed.items()}
    return RiskGateResult(
        proposed_weights=proposed,
        executed_weights=executed,
        reasons=reasons,
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
    )
