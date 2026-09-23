"""Small, explicit data contracts used at the AI/Python boundary."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

UNIVERSE = ("AAPL", "MSFT", "NVDA")
BENCHMARK = "SPY"


class TargetWeightProposal(BaseModel):
    """The only output accepted from the Portfolio Manager agent."""

    aapl_weight: float = Field(ge=0.0, le=1.0)
    msft_weight: float = Field(ge=0.0, le=1.0)
    nvda_weight: float = Field(ge=0.0, le=1.0)
    cash_weight: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_total_weight(self) -> TargetWeightProposal:
        values = (
            self.aapl_weight,
            self.msft_weight,
            self.nvda_weight,
            self.cash_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all target weights must be finite")
        total = sum(values)
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"target weights must sum to 1.0 (received {total:.6f})")
        return self

    def asset_weights(self) -> dict[str, float]:
        return {
            "AAPL": self.aapl_weight,
            "MSFT": self.msft_weight,
            "NVDA": self.nvda_weight,
        }
