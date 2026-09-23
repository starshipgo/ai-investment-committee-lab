"""The four-role Lumibot strategy and its deterministic execution boundary."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from lumibot.components.agents import AgentRunResult, agent_tool
from lumibot.strategies import Strategy
from pydantic import ValidationError

from .artifacts import append_csv, append_jsonl, save_agent_decision, save_risk_decisions
from .models import BENCHMARK, UNIVERSE, TargetWeightProposal
from .risk import apply_risk_gate

PROPOSAL_TOOL_NAME = "record_target_weight_proposal"
_PROPOSAL_FIELDS = frozenset(
    {"aapl_weight", "msft_weight", "nvda_weight", "cash_weight", "rationale"}
)


class ProposalRecoveryError(RuntimeError):
    """A recorded Portfolio Manager proposal cannot safely be used."""

    safe_category = "portfolio_proposal"


def _validate_recorded_proposal(payload: Any) -> TargetWeightProposal:
    """Turn one recorded tool payload into the project's typed proposal contract."""

    if not isinstance(payload, dict) or set(payload) != _PROPOSAL_FIELDS:
        raise ProposalRecoveryError(
            "Portfolio Manager proposal audit record is malformed; no orders were allowed"
        )
    try:
        return TargetWeightProposal.model_validate(payload)
    except ValidationError as exc:
        raise ProposalRecoveryError(
            "Portfolio Manager proposal audit record is invalid; no orders were allowed"
        ) from exc


def recover_recorded_proposal(result: AgentRunResult) -> TargetWeightProposal:
    """Recover the one validated Portfolio Manager proposal from its run audit record.

    LumiBot serializes tool call/result events into its replay cache but does not replay a
    custom tool's in-memory mutation by default. Both cold and cache-hit runs therefore use
    these recorded events as the only decision input after the agent returns.
    """

    proposal_calls = [
        event for event in result.tool_calls if event.tool_name == PROPOSAL_TOOL_NAME
    ]
    if len(proposal_calls) != 1:
        raise ProposalRecoveryError(
            "Portfolio Manager must record exactly one target-weight proposal; no orders were allowed"
        )

    proposal_call = proposal_calls[0]
    matching_results = [
        event
        for event in result.tool_results
        if event.tool_name == PROPOSAL_TOOL_NAME and event.call_id == proposal_call.call_id
    ]
    if len(matching_results) != 1:
        raise ProposalRecoveryError(
            "Portfolio Manager proposal has no matching recorded tool result; no orders were allowed"
        )

    tool_result = matching_results[0]
    result_payload = tool_result.payload
    if not isinstance(result_payload, dict) or result_payload.get("tool_error") is True:
        raise ProposalRecoveryError(
            "Portfolio Manager proposal tool did not complete successfully; no orders were allowed"
        )
    if result_payload.get("status") != "proposal_recorded_not_executed":
        raise ProposalRecoveryError(
            "Portfolio Manager proposal tool returned an unexpected result; no orders were allowed"
        )

    proposal_from_call = _validate_recorded_proposal(proposal_call.payload)
    proposal_from_result = _validate_recorded_proposal(result_payload.get("proposal"))
    if proposal_from_call != proposal_from_result:
        raise ProposalRecoveryError(
            "Portfolio Manager proposal call and result do not match; no orders were allowed"
        )
    return proposal_from_result


class AIInvestmentCommitteeStrategy(Strategy):
    """One decision, three read-only analysts, and one non-trading allocator."""

    parameters: ClassVar[dict[str, Any]] = {
        "universe": list(UNIVERSE),
        "benchmark": BENCHMARK,
        "lookback_days": 20,
        "max_position_weight": 0.25,
        "max_gross_exposure": 1.0,
        "artifact_dir": "artifacts/latest",
        "agent_max_model_calls": 4,
    }

    @agent_tool(
        name="get_universe_snapshot",
        description=(
            "Return point-in-time daily price features for AAPL, MSFT, NVDA, and SPY. "
            "It uses the experiment's configured completed-bar lookback. This tool is read-only "
            "and safe for historical backtests."
        ),
    )
    def get_universe_snapshot(self) -> dict[str, Any]:
        """Build a compact point-in-time snapshot from Lumibot/Yahoo bars.

        Returns:
            A symbol-keyed dictionary of prices, returns, and annualized volatility.
        """

        length = self._configured_lookback_days()
        snapshot: dict[str, Any] = {}
        for symbol in (*self.parameters["universe"], self.parameters["benchmark"]):
            # Explicitly shift back one bar so a daily backtest never exposes the
            # simulated session's not-yet-completed close to the Researcher.
            bars = self.get_historical_prices(symbol, length, "day", timeshift=1)
            if bars is None or bars.pandas_df is None or bars.pandas_df.empty:
                snapshot[symbol] = {"error": "no historical bars available"}
                continue
            frame = bars.pandas_df
            close_column = "close" if "close" in frame.columns else "Close"
            closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
            if closes.empty:
                snapshot[symbol] = {"error": "no valid closing prices"}
                continue
            returns = closes.pct_change(fill_method=None).dropna()
            snapshot[symbol] = {
                "bars": len(closes),
                "last_close": round(float(closes.iloc[-1]), 4),
                "return_over_window": round(float(closes.iloc[-1] / closes.iloc[0] - 1.0), 6),
                "annualized_volatility": round(
                    float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0,
                    6,
                ),
            }
        return {"as_of": self.get_datetime().isoformat(), "snapshot": snapshot}

    def _configured_lookback_days(self) -> int:
        """Return the validated lookback owned by the experiment configuration."""

        try:
            length = int(self.parameters["lookback_days"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("lookback_days must be an integer between 10 and 120") from exc
        if not 10 <= length <= 120:
            raise ValueError("lookback_days must be between 10 and 120")
        return length

    @agent_tool(
        name="record_target_weight_proposal",
        description=(
            "Record, but do not execute, a structured long-only target-weight proposal. "
            "All four weights must sum to 1.0. This tool cannot place orders."
        ),
    )
    def record_target_weight_proposal(
        self,
        aapl_weight: float,
        msft_weight: float,
        nvda_weight: float,
        cash_weight: float,
        rationale: str,
    ) -> dict[str, Any]:
        """Validate and record the Portfolio Manager's sole permitted output.

        Args:
            aapl_weight: Raw target fraction for AAPL before Python risk controls.
            msft_weight: Raw target fraction for MSFT before Python risk controls.
            nvda_weight: Raw target fraction for NVDA before Python risk controls.
            cash_weight: Raw target fraction to leave as cash.
            rationale: Concise synthesis of the Researcher, Bull, and Bear handoffs.

        Returns:
            The validated proposal and a reminder that no order was submitted.
        """

        proposal = TargetWeightProposal(
            aapl_weight=aapl_weight,
            msft_weight=msft_weight,
            nvda_weight=nvda_weight,
            cash_weight=cash_weight,
            rationale=rationale,
        )
        self._pending_proposal = proposal
        return {
            "status": "proposal_recorded_not_executed",
            "proposal": proposal.model_dump(),
        }

    def initialize(self) -> None:
        self.sleeptime = "1D"
        self._decision_complete = False
        self._artifact_dir = Path(self.parameters["artifact_dir"]).resolve()
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        model = os.environ.get("AI_MODEL", "gemini-3.5-flash-lite")
        self.agents.create(
            name="researcher",
            model=model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            tools=[self.get_universe_snapshot],
            system_prompt=(
                "You are the Researcher. Use only point-in-time price evidence to compare "
                "AAPL, MSFT, and NVDA against SPY; report facts and uncertainty, never orders."
            ),
        )
        self.agents.create(
            name="bull_analyst",
            model=model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            system_prompt=(
                "You are the Bull Analyst. Build the strongest evidence-based long case from "
                "the Researcher handoff, while remaining read-only and never placing orders."
            ),
        )
        self.agents.create(
            name="bear_analyst",
            model=model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            system_prompt=(
                "You are the Bear Analyst. Challenge the evidence, identify drawdown and "
                "concentration risks, and remain read-only; never place orders."
            ),
        )
        self.agents.create(
            name="portfolio_manager",
            model=model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            tools=[self.record_target_weight_proposal],
            system_prompt=(
                "You are the Portfolio Manager. Synthesize the three handoffs into raw long-only "
                "target weights; you may only record a proposal and can never place an order."
            ),
        )

    @staticmethod
    def _decision_text(result: Any) -> str:
        return (result.summary or result.text or "").strip()

    def _apply_risk_gate(self, proposal: TargetWeightProposal):
        """Keep policy application explicit and independent of every agent."""

        return apply_risk_gate(
            proposal.asset_weights(),
            self.parameters["universe"],
            max_position_weight=float(self.parameters["max_position_weight"]),
            max_gross_exposure=float(self.parameters["max_gross_exposure"]),
        )

    def on_trading_iteration(self) -> None:
        if self._decision_complete:
            return

        as_of = self.get_datetime().isoformat()
        lookback_days = self._configured_lookback_days()
        context = {
            "as_of": as_of,
            "universe": self.parameters["universe"],
            "benchmark": self.parameters["benchmark"],
            "lookback_days": lookback_days,
            "educational_only": True,
        }
        researcher = self.agents["researcher"].run(
            task_prompt=(
                "Call get_universe_snapshot once. It uses the configured "
                f"{lookback_days} completed daily bars. Then produce a compact evidence table and "
                "rank the three stocks. Do not recommend an order."
            ),
            context=context,
        )
        save_agent_decision(
            self._artifact_dir / "agent_decisions.jsonl",
            role="Researcher",
            as_of=as_of,
            result=researcher,
        )

        bull = self.agents["bull_analyst"].run(
            task_prompt="Make the strongest concise long case for the best candidate.",
            context={**context, "researcher_handoff": self._decision_text(researcher)},
        )
        save_agent_decision(
            self._artifact_dir / "agent_decisions.jsonl",
            role="Bull Analyst",
            as_of=as_of,
            result=bull,
        )

        bear = self.agents["bear_analyst"].run(
            task_prompt="Stress-test the evidence and the Bull Analyst's preferred candidate.",
            context={
                **context,
                "researcher_handoff": self._decision_text(researcher),
                "bull_handoff": self._decision_text(bull),
            },
        )
        save_agent_decision(
            self._artifact_dir / "agent_decisions.jsonl",
            role="Bear Analyst",
            as_of=as_of,
            result=bear,
        )

        portfolio_manager = self.agents["portfolio_manager"].run(
            task_prompt=(
                "Call record_target_weight_proposal exactly once. The four weights must sum to "
                "1.0. State raw conviction weights without applying any position cap—the "
                "deterministic Python risk gate owns all limits. Do not place or request orders."
            ),
            context={
                **context,
                "researcher_handoff": self._decision_text(researcher),
                "bull_handoff": self._decision_text(bull),
                "bear_handoff": self._decision_text(bear),
            },
        )
        save_agent_decision(
            self._artifact_dir / "agent_decisions.jsonl",
            role="Portfolio Manager",
            as_of=as_of,
            result=portfolio_manager,
        )
        # Never depend on the tool's local mutation: cache replay returns its audit events,
        # not necessarily the in-memory effect of the original tool invocation.
        proposal = recover_recorded_proposal(portfolio_manager)
        append_jsonl(
            self._artifact_dir / "portfolio_proposals.jsonl",
            {"as_of": as_of, **proposal.model_dump()},
        )
        risk_result = self._apply_risk_gate(proposal)
        save_risk_decisions(
            self._artifact_dir / "risk_decisions.csv",
            as_of=as_of,
            result=risk_result,
            proposed_cash_weight=proposal.cash_weight,
        )
        self._execute_approved_targets(risk_result.executed_weights, as_of)
        self._decision_complete = True

    def _execute_approved_targets(self, weights: dict[str, float], as_of: str) -> None:
        portfolio_value = float(self.get_portfolio_value())
        sell_orders = []
        buy_orders = []
        for symbol in self.parameters["universe"]:
            price = float(self.get_last_price(symbol))
            position = self.get_position(symbol)
            current_quantity = int(float(position.quantity)) if position is not None else 0
            target_quantity = math.floor(portfolio_value * weights[symbol] / price)
            delta = target_quantity - current_quantity
            append_csv(
                self._artifact_dir / "approved_orders.csv",
                {
                    "as_of": as_of,
                    "symbol": symbol,
                    "executed_weight": weights[symbol],
                    "reference_price": price,
                    "current_quantity": current_quantity,
                    "target_quantity": target_quantity,
                    "delta_quantity": delta,
                },
                (
                    "as_of",
                    "symbol",
                    "executed_weight",
                    "reference_price",
                    "current_quantity",
                    "target_quantity",
                    "delta_quantity",
                ),
            )
            if delta < 0:
                sell_orders.append(self.create_order(symbol, abs(delta), "sell"))
            elif delta > 0:
                buy_orders.append(self.create_order(symbol, delta, "buy"))

        for order in (*sell_orders, *buy_orders):
            self.submit_order(order)

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        append_csv(
            self._artifact_dir / "trades.csv",
            {
                "time": self.get_datetime().isoformat(),
                "symbol": order.asset.symbol,
                "side": str(order.side),
                "quantity": float(quantity),
                "price": float(price),
                "notional": float(price) * float(quantity) * float(multiplier),
            },
            ("time", "symbol", "side", "quantity", "price", "notional"),
        )
