"""Artifact writers for the backtest and future read-only UI."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk import RiskGateResult


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def append_csv(path: Path, record: dict[str, Any], fieldnames: Iterable[str]) -> None:
    fields = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def save_agent_decision(path: Path, *, role: str, as_of: str, result: Any) -> None:
    payload = result.payload if isinstance(result.payload, dict) else {}
    append_jsonl(
        path,
        {
            "as_of": as_of,
            "role": role,
            "model": result.model,
            "summary": result.summary,
            "text": result.text,
            "cache_hit": result.cache_hit,
            "tool_calls": [event.tool_name for event in result.tool_calls],
            "warnings": result.warning_messages,
            "trace_path": payload.get("trace_path"),
            "latency_ms": result.latency_ms,
        },
    )


def save_risk_decisions(
    path: Path,
    *,
    as_of: str,
    result: RiskGateResult,
    proposed_cash_weight: float,
) -> None:
    fields = (
        "as_of",
        "symbol",
        "proposed_weight",
        "executed_weight",
        "overridden",
        "reason",
    )
    for symbol, proposed_weight in result.proposed_weights.items():
        executed_weight = result.executed_weights[symbol]
        append_csv(
            path,
            {
                "as_of": as_of,
                "symbol": symbol,
                "proposed_weight": proposed_weight,
                "executed_weight": executed_weight,
                "overridden": not math.isclose(proposed_weight, executed_weight, abs_tol=1e-10),
                "reason": "|".join(result.reasons[symbol]) or "approved",
            },
            fields,
        )

    executed_cash = max(0.0, 1.0 - result.executed_gross_exposure)
    append_csv(
        path,
        {
            "as_of": as_of,
            "symbol": "CASH",
            "proposed_weight": proposed_cash_weight,
            "executed_weight": executed_cash,
            "overridden": not math.isclose(proposed_cash_weight, executed_cash, abs_tol=1e-10),
            "reason": "residual_after_risk_gate",
        },
        fields,
    )


def _daily_series(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").dropna().copy()
    index = pd.to_datetime(result.index, utc=True, errors="coerce")
    result.index = index.tz_localize(None).normalize()
    return result[~result.index.duplicated(keep="last")].sort_index()


def export_performance(strategy: Any, artifact_dir: Path, initial_budget: float) -> dict[str, Any]:
    """Export aligned portfolio/benchmark equity and independently computed metrics."""

    stats = getattr(strategy, "_strategy_returns_df", None)
    if not isinstance(stats, pd.DataFrame) or stats.empty or "portfolio_value" not in stats:
        stats = getattr(strategy, "_stats", None)
    if not isinstance(stats, pd.DataFrame) or stats.empty or "portfolio_value" not in stats:
        raise RuntimeError("Lumibot did not produce a portfolio equity series")

    portfolio = _daily_series(stats["portfolio_value"])
    benchmark_frame = getattr(strategy, "_benchmark_returns_df", None)
    if (
        not isinstance(benchmark_frame, pd.DataFrame)
        or benchmark_frame.empty
        or "symbol_cumprod" not in benchmark_frame
    ):
        raise RuntimeError("Lumibot/Yahoo did not produce the SPY benchmark series")
    benchmark = _daily_series(benchmark_frame["symbol_cumprod"]) * float(initial_budget)

    equity = pd.concat(
        [portfolio.rename("portfolio_equity"), benchmark.rename("benchmark_equity")],
        axis=1,
    ).sort_index()
    equity.index.name = "date"
    equity.to_csv(artifact_dir / "equity_curve.csv")

    returns = portfolio.pct_change(fill_method=None).dropna()
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(math.sqrt(252.0) * returns.mean() / volatility)
        if volatility > 0.0
        else 0.0
    )
    drawdown = portfolio / portfolio.cummax() - 1.0
    benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
    metrics = {
        "initial_equity": float(portfolio.iloc[0]),
        "final_equity": float(portfolio.iloc[-1]),
        "total_return": float(portfolio.iloc[-1] / portfolio.iloc[0] - 1.0),
        "benchmark_total_return": benchmark_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe,
        "risk_free_rate": 0.0,
        "trading_days": len(portfolio),
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
