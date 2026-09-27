"""Immutable, analysis-only forward observations for saved Today decisions.

This module deliberately has no committee, agent-manager, Gemini, broker, or order API
dependency.  A Forward Test copies a completed Today decision once, then explicit refreshes
can add only post-cutoff Yahoo market observations to that frozen decision.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from lumibot.data_sources import YahooData
from pydantic import ValidationError

from .models import BENCHMARK, UNIVERSE, TargetWeightProposal
from .risk import apply_risk_gate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORWARD_ARTIFACT_DIRECTORY = "forward"
FORWARD_SUMMARY_FILE = "forward_summary.json"
FROZEN_DECISION_FILE = "frozen_decision.json"
FORWARD_PERFORMANCE_FILE = "forward_performance.json"
FORWARD_COMPLETION_FILES = (FORWARD_SUMMARY_FILE, FROZEN_DECISION_FILE)
FORWARD_HORIZONS = (1, 5, 20)
_TODAY_SOURCE_FILES = (
    "today_summary.json",
    "evidence.json",
    "agent_decisions.jsonl",
    "portfolio_proposals.jsonl",
    "risk_decisions.csv",
)


class ForwardTestError(RuntimeError):
    """A saved decision cannot safely become, or remain, a Forward Test."""

    safe_category = "forward_test"


class ForwardMarketDataError(ForwardTestError):
    """Yahoo did not provide a usable completed-bar observation."""

    safe_category = "market_data"


@dataclass(frozen=True)
class ForwardTestRun:
    """The Forward Test directory and whether an existing frozen decision was reused."""

    artifact_dir: Path
    reused: bool


def _utc_now(value: datetime | None = None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Avoid exposing a partial mutable observation file to a local Streamlit reload."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    _write_json(temporary_path, payload)
    temporary_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForwardTestError("Saved Forward Test data is unreadable") from exc
    if not isinstance(result, dict):
        raise ForwardTestError("Saved Forward Test data has an invalid structure")
    return result


def _timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ForwardTestError(f"Saved {label} is invalid") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ForwardTestError(f"Saved {label} must include a timezone")
    return result.tz_convert("UTC")


def _finite_weight(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ForwardTestError(f"Saved {label} is not numeric") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ForwardTestError(f"Saved {label} is outside the allowed range")
    return result


def _finite_price(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ForwardTestError(f"Saved {label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ForwardTestError(f"Saved {label} must be a positive price")
    return result


def _today_root(project_root: Path) -> Path:
    return (project_root / "artifacts" / "today").resolve()


def _forward_root(project_root: Path) -> Path:
    return (project_root / "artifacts" / FORWARD_ARTIFACT_DIRECTORY).resolve()


def _is_within_root(path: Path, root: Path) -> bool:
    return path.parent == root


def _resolve_today_source(today_dir: Path, *, project_root: Path) -> Path:
    source = Path(today_dir).resolve()
    if not _is_within_root(source, _today_root(project_root)):
        raise ForwardTestError("Forward Tests can only be created from saved Today analyses")
    if not source.is_dir() or not all((source / name).is_file() for name in _TODAY_SOURCE_FILES):
        raise ForwardTestError("The selected Today analysis is incomplete")
    return source


def _resolve_forward_artifact(forward_dir: Path, *, project_root: Path) -> Path:
    artifact_dir = Path(forward_dir).resolve()
    if not _is_within_root(artifact_dir, _forward_root(project_root)):
        raise ForwardTestError("The selected artifact is not a Forward Test")
    if not artifact_dir.is_dir() or not all(
        (artifact_dir / name).is_file() for name in FORWARD_COMPLETION_FILES
    ):
        raise ForwardTestError("The selected Forward Test is incomplete")
    return artifact_dir


def _parse_proposal(path: Path, *, analysis_timestamp: pd.Timestamp) -> TargetWeightProposal:
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ForwardTestError("Saved Today proposal is unreadable") from exc
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ForwardTestError("Saved Today analysis must contain exactly one structured proposal")
    if _timestamp(records[0].get("as_of"), label="Portfolio Manager proposal timestamp") != analysis_timestamp:
        raise ForwardTestError("Saved Today proposal timestamp does not match the analysis")
    try:
        return TargetWeightProposal.model_validate(records[0])
    except ValidationError as exc:
        raise ForwardTestError("Saved Today proposal has invalid target weights") from exc


def _parse_risk_decisions(
    path: Path,
    *,
    proposal: TargetWeightProposal,
    analysis_timestamp: pd.Timestamp,
    risk_limits: Mapping[str, float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    except OSError as exc:
        raise ForwardTestError("Saved Today risk decisions are unreadable") from exc

    expected_symbols = (*UNIVERSE, "CASH")
    by_symbol: dict[str, dict[str, Any]] = {}
    for record in records:
        symbol = str(record.get("symbol", "")).upper()
        if symbol not in expected_symbols or symbol in by_symbol:
            raise ForwardTestError("Saved Today risk decisions have invalid symbols")
        by_symbol[symbol] = record
    if set(by_symbol) != set(expected_symbols):
        raise ForwardTestError("Saved Today risk decisions are incomplete")

    proposal_weights = {**proposal.asset_weights(), "CASH": proposal.cash_weight}
    expected_gate = apply_risk_gate(
        proposal.asset_weights(),
        UNIVERSE,
        max_position_weight=risk_limits["max_position_weight"],
        max_gross_exposure=risk_limits["max_gross_exposure"],
    )
    expected_allowed = {
        **expected_gate.executed_weights,
        "CASH": max(0.0, 1.0 - expected_gate.executed_gross_exposure),
    }
    allowed_weights: dict[str, float] = {}
    frozen_rows: list[dict[str, Any]] = []
    for symbol in expected_symbols:
        record = by_symbol[symbol]
        if _timestamp(record.get("as_of"), label=f"{symbol} Risk Gate timestamp") != analysis_timestamp:
            raise ForwardTestError("Saved Today Risk Gate timestamps do not match the analysis")
        proposed = _finite_weight(record.get("proposed_weight"), label=f"{symbol} proposed weight")
        executed = _finite_weight(record.get("executed_weight"), label=f"{symbol} allowed weight")
        if not math.isclose(proposed, proposal_weights[symbol], abs_tol=1e-9):
            raise ForwardTestError("Saved Today proposal and Risk Gate records do not match")
        if not math.isclose(executed, expected_allowed[symbol], abs_tol=1e-9):
            raise ForwardTestError("Saved Today allocation does not match the deterministic Risk Gate")
        raw_overridden = str(record.get("overridden", "")).strip().lower()
        if raw_overridden not in {"true", "false", "1", "0", "yes", "no"}:
            raise ForwardTestError("Saved Today Risk Gate override flag is invalid")
        overridden = raw_overridden in {"true", "1", "yes"}
        expected_overridden = not math.isclose(
            proposal_weights[symbol], expected_allowed[symbol], abs_tol=1e-10
        )
        if overridden is not expected_overridden:
            raise ForwardTestError("Saved Today Risk Gate override flags do not match the allocation")
        expected_reason = (
            "residual_after_risk_gate"
            if symbol == "CASH"
            else "|".join(expected_gate.reasons[symbol]) or "approved"
        )
        if str(record.get("reason", "")).strip() != expected_reason:
            raise ForwardTestError("Saved Today Risk Gate reasons do not match the allocation")
        allowed_weights[symbol] = executed
        frozen_rows.append(
            {
                "symbol": symbol,
                "proposed_weight": proposed,
                "executed_weight": executed,
                "overridden": overridden,
                "reason": str(record.get("reason", "approved")),
            }
        )

    if not math.isclose(sum(allowed_weights.values()), 1.0, abs_tol=1e-8):
        raise ForwardTestError("Saved Today allocation does not preserve all capital")
    return allowed_weights, frozen_rows


def _parse_risk_limits(summary: Mapping[str, Any]) -> dict[str, float]:
    raw_limits = summary.get("risk_limits")
    if not isinstance(raw_limits, Mapping):
        raise ForwardTestError("Saved Today risk limits are missing")
    max_position_weight = _finite_weight(
        raw_limits.get("max_position_weight"), label="maximum position limit"
    )
    max_gross_exposure = _finite_weight(
        raw_limits.get("max_gross_exposure"), label="maximum gross exposure"
    )
    if max_position_weight <= 0.0 or max_gross_exposure <= 0.0:
        raise ForwardTestError("Saved Today risk limits must be positive")
    return {
        "max_position_weight": max_position_weight,
        "max_gross_exposure": max_gross_exposure,
    }


def _freeze_today_decision(today_dir: Path, *, project_root: Path) -> dict[str, Any]:
    """Read and validate a completed Today artifact without importing any agent code."""

    source = _resolve_today_source(today_dir, project_root=project_root)
    summary = _read_json(source / "today_summary.json")
    evidence = _read_json(source / "evidence.json")
    if summary.get("artifact_kind") != "today_analysis" or summary.get("analysis_only") is not True:
        raise ForwardTestError("The selected artifact is not an analysis-only Today decision")
    if summary.get("order_submitted") is not False:
        raise ForwardTestError("A Forward Test requires a no-order Today decision")
    if tuple(summary.get("universe", ())) != UNIVERSE or summary.get("benchmark") != BENCHMARK:
        raise ForwardTestError("Saved Today universe or benchmark does not match this lab")

    original_analysis = _timestamp(summary.get("analysis_timestamp"), label="Today analysis timestamp")
    evidence_cutoff = _timestamp(summary.get("evidence_as_of"), label="Today evidence cutoff")
    if _timestamp(evidence.get("analysis_timestamp"), label="evidence analysis timestamp") != original_analysis:
        raise ForwardTestError("Saved Today analysis timestamps do not match")
    if _timestamp(evidence.get("evidence_as_of"), label="evidence cutoff") != evidence_cutoff:
        raise ForwardTestError("Saved Today evidence cutoff does not match")
    if evidence.get("evidence_fingerprint") != summary.get("evidence_fingerprint"):
        raise ForwardTestError("Saved Today evidence fingerprint does not match")

    try:
        lookback_days = int(summary["lookback_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ForwardTestError("Saved Today lookback window is invalid") from exc
    if lookback_days <= 0 or evidence.get("lookback_days") != lookback_days:
        raise ForwardTestError("Saved Today evidence window does not match")

    snapshot = evidence.get("snapshot")
    bar_timestamps = evidence.get("bar_timestamps")
    if not isinstance(snapshot, Mapping) or not isinstance(bar_timestamps, Mapping):
        raise ForwardTestError("Saved Today evidence is incomplete")
    entry_prices: dict[str, float] = {}
    entry_bar_timestamps: dict[str, str] = {}
    for symbol in (*UNIVERSE, BENCHMARK):
        item = snapshot.get(symbol)
        if not isinstance(item, Mapping):
            raise ForwardTestError("Saved Today evidence is missing a reference price")
        entry_prices[symbol] = _finite_price(item.get("last_close"), label=f"{symbol} reference price")
        bar_timestamp = _timestamp(bar_timestamps.get(symbol), label=f"{symbol} evidence bar")
        if bar_timestamp != evidence_cutoff:
            raise ForwardTestError("Saved Today evidence bars do not share the frozen cutoff")
        entry_bar_timestamps[symbol] = str(bar_timestamps[symbol])

    risk_limits = _parse_risk_limits(summary)
    proposal = _parse_proposal(source / "portfolio_proposals.jsonl", analysis_timestamp=original_analysis)
    allowed_weights, risk_rows = _parse_risk_decisions(
        source / "risk_decisions.csv",
        proposal=proposal,
        analysis_timestamp=original_analysis,
        risk_limits=risk_limits,
    )
    return {
        "artifact_kind": "forward_test_frozen_decision",
        "schema_version": 1,
        "source_today_artifact_id": source.name,
        "source_today_evidence_fingerprint": str(summary["evidence_fingerprint"]),
        "original_analysis_timestamp": str(summary["analysis_timestamp"]),
        "evidence_as_of": str(summary["evidence_as_of"]),
        "evidence_cutoff_utc": evidence_cutoff.isoformat(),
        "data_source": str(summary.get("data_source", "YahooData")),
        "lookback_days": lookback_days,
        "universe": list(UNIVERSE),
        "benchmark": BENCHMARK,
        "proposal": {
            "AAPL": proposal.aapl_weight,
            "MSFT": proposal.msft_weight,
            "NVDA": proposal.nvda_weight,
            "CASH": proposal.cash_weight,
            "rationale": proposal.rationale,
        },
        "allowed_allocation": allowed_weights,
        "cash_weight": allowed_weights["CASH"],
        "risk_decisions": risk_rows,
        "risk_limits": risk_limits,
        "entry_prices": entry_prices,
        "entry_bar_timestamps": entry_bar_timestamps,
    }


def _is_completed_forward_test(path: Path) -> bool:
    """Recognize a complete Forward Test without looking at Today or historical artifacts."""

    if not path.is_dir() or not all((path / name).is_file() for name in FORWARD_COMPLETION_FILES):
        return False
    try:
        summary = json.loads((path / FORWARD_SUMMARY_FILE).read_text(encoding="utf-8"))
        frozen = json.loads((path / FROZEN_DECISION_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(summary, dict)
        and isinstance(frozen, dict)
        and summary.get("artifact_kind") == "forward_test"
        and summary.get("analysis_only") is True
        and summary.get("order_submitted") is False
        and frozen.get("artifact_kind") == "forward_test_frozen_decision"
    )


def _find_forward_for_frozen_decision(
    frozen_decision: Mapping[str, Any], *, project_root: Path
) -> Path | None:
    forward_root = _forward_root(project_root)
    if not forward_root.is_dir():
        return None
    for candidate in sorted(forward_root.iterdir(), key=lambda path: path.name, reverse=True):
        if not _is_completed_forward_test(candidate):
            continue
        try:
            candidate_frozen = _read_json(candidate / FROZEN_DECISION_FILE)
        except ForwardTestError:
            continue
        # Source artifacts are intended to be immutable, but compare the complete frozen payload
        # before reuse so a locally altered risk allocation can never silently inherit an old test.
        if candidate_frozen == dict(frozen_decision):
            return candidate.resolve()
    return None


def _new_forward_artifact_dir(project_root: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = _forward_root(project_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _waiting_performance() -> dict[str, Any]:
    return {
        "status": "waiting_for_future_data",
        "observed_trading_days": 0,
        "observations": [],
        "current": None,
        "horizons": {
            str(days): {"status": "pending", "trading_day": days} for days in FORWARD_HORIZONS
        },
    }


def create_forward_test_from_today(
    today_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> ForwardTestRun:
    """Freeze one saved Today decision without calling Yahoo, agents, Gemini, or a broker."""

    frozen_decision = _freeze_today_decision(today_dir, project_root=project_root)
    existing = _find_forward_for_frozen_decision(frozen_decision, project_root=project_root)
    if existing is not None:
        return ForwardTestRun(artifact_dir=existing, reused=True)

    artifact_dir = _new_forward_artifact_dir(project_root)
    _write_json(artifact_dir / FROZEN_DECISION_FILE, frozen_decision)
    _write_json(artifact_dir / FORWARD_PERFORMANCE_FILE, _waiting_performance())
    summary = {
        "artifact_kind": "forward_test",
        "schema_version": 1,
        "forward_test_id": artifact_dir.name,
        "source_today_artifact_id": frozen_decision["source_today_artifact_id"],
        "original_analysis_timestamp": frozen_decision["original_analysis_timestamp"],
        "evidence_as_of": frozen_decision["evidence_as_of"],
        "data_source": frozen_decision["data_source"],
        "lookback_days": frozen_decision["lookback_days"],
        "universe": frozen_decision["universe"],
        "benchmark": frozen_decision["benchmark"],
        "analysis_only": True,
        "order_submitted": False,
        "agent_run_started": False,
        "rebalancing": False,
    }
    # Completion marker comes last. A failed copy cannot look like a usable Forward Test.
    _write_json(artifact_dir / FORWARD_SUMMARY_FILE, summary)
    _write_json(
        project_root / "artifacts" / "latest_forward.json",
        {"artifact_dir": str(artifact_dir)},
    )
    return ForwardTestRun(artifact_dir=artifact_dir, reused=False)


def _normalise_daily_closes(frame: Any, *, symbol: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ForwardMarketDataError(f"Yahoo did not return completed daily bars for {symbol}")
    close_column = "close" if "close" in frame.columns else "Close"
    if close_column not in frame.columns:
        raise ForwardMarketDataError(f"Yahoo bars for {symbol} did not contain closing prices")
    closes = pd.to_numeric(frame[close_column], errors="coerce")
    timestamps = pd.to_datetime(closes.index, utc=True, errors="coerce")
    valid = closes.notna() & ~timestamps.isna()
    result = pd.Series(closes.loc[valid].to_numpy(), index=timestamps[valid], dtype=float)
    result = result[(result > 0.0) & result.map(math.isfinite)]
    return result[~result.index.duplicated(keep="last")].sort_index()


def _required_symbols(frozen_decision: Mapping[str, Any]) -> tuple[str, ...]:
    allocation = frozen_decision.get("allowed_allocation")
    if not isinstance(allocation, Mapping):
        raise ForwardTestError("Frozen allowed allocation is missing")
    required = [
        symbol
        for symbol in UNIVERSE
        if _finite_weight(allocation.get(symbol), label=f"frozen {symbol} allocation") > 0.0
    ]
    return (*required, BENCHMARK)


def collect_forward_yahoo_prices(
    frozen_decision: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, pd.Series]:
    """Fetch current completed daily Yahoo bars for an explicit Forward Test refresh only."""

    evidence_cutoff = _timestamp(frozen_decision.get("evidence_as_of"), label="evidence cutoff")
    current_time = _utc_now(now)
    requested_days = max(30, (current_time.date() - evidence_cutoff.date()).days + 10)
    try:
        source = YahooData(
            auto_adjust=False,
            datetime_start=current_time,
            datetime_end=current_time + timedelta(minutes=1),
        )
    except Exception as exc:  # provider construction never exposes its detail in the UI
        raise ForwardMarketDataError("Yahoo did not provide a current daily price source") from exc

    prices: dict[str, pd.Series] = {}
    for symbol in _required_symbols(frozen_decision):
        try:
            bars = source.get_historical_prices(symbol, requested_days, "day", timeshift=None)
        except Exception as exc:
            raise ForwardMarketDataError(
                f"Yahoo did not provide completed daily bars for {symbol}"
            ) from exc
        frame = getattr(bars, "pandas_df", None) if bars is not None else None
        normalized = _normalise_daily_closes(frame, symbol=symbol)
        # Keep the provider output itself post-cutoff as a first defense. The performance
        # calculator repeats this full-timestamp check so injected/test providers cannot bypass it.
        prices[symbol] = normalized[normalized.index > evidence_cutoff]
    return prices


def _post_cutoff_daily_series(
    series: Any,
    *,
    cutoff: pd.Timestamp,
    symbol: str,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ForwardTestError(f"Forward price data for {symbol} is invalid")
    values = pd.to_numeric(series, errors="coerce")
    timestamps = pd.to_datetime(values.index, utc=True, errors="coerce")
    valid = values.notna() & ~timestamps.isna()
    result = pd.Series(values.loc[valid].to_numpy(), index=timestamps[valid], dtype=float)
    result = result[(result > 0.0) & result.map(math.isfinite)]
    # The full timestamp check intentionally happens before converting values to a daily index.
    # A cutoff bar can never become a forward observation merely because it shares a calendar date.
    result = result[result.index > cutoff]
    result.index = result.index.normalize()
    return result[~result.index.duplicated(keep="last")].sort_index()


def _validated_frozen_decision(frozen_decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable inputs before using them in a performance calculation."""

    if frozen_decision.get("artifact_kind") != "forward_test_frozen_decision":
        raise ForwardTestError("Frozen Forward Test decision has an invalid type")
    if tuple(frozen_decision.get("universe", ())) != UNIVERSE:
        raise ForwardTestError("Frozen Forward Test universe does not match this lab")
    if frozen_decision.get("benchmark") != BENCHMARK:
        raise ForwardTestError("Frozen Forward Test benchmark does not match this lab")
    _timestamp(frozen_decision.get("original_analysis_timestamp"), label="analysis timestamp")
    evidence_cutoff = _timestamp(frozen_decision.get("evidence_as_of"), label="evidence cutoff")
    if _timestamp(frozen_decision.get("evidence_cutoff_utc"), label="UTC evidence cutoff") != evidence_cutoff:
        raise ForwardTestError("Frozen Forward Test evidence cutoff does not match")
    proposal_record = frozen_decision.get("proposal")
    if not isinstance(proposal_record, Mapping):
        raise ForwardTestError("Frozen Portfolio Manager proposal is missing")
    try:
        proposal = TargetWeightProposal.model_validate(
            {
                "aapl_weight": proposal_record.get("AAPL"),
                "msft_weight": proposal_record.get("MSFT"),
                "nvda_weight": proposal_record.get("NVDA"),
                "cash_weight": proposal_record.get("CASH"),
                "rationale": proposal_record.get("rationale"),
            }
        )
    except ValidationError as exc:
        raise ForwardTestError("Frozen Portfolio Manager proposal has invalid target weights") from exc
    risk_limits = _parse_risk_limits({"risk_limits": frozen_decision.get("risk_limits")})
    allocation = frozen_decision.get("allowed_allocation")
    prices = frozen_decision.get("entry_prices")
    bar_timestamps = frozen_decision.get("entry_bar_timestamps")
    if (
        not isinstance(allocation, Mapping)
        or not isinstance(prices, Mapping)
        or not isinstance(bar_timestamps, Mapping)
    ):
        raise ForwardTestError("Frozen Forward Test allocation or reference prices are missing")
    total = 0.0
    for symbol in (*UNIVERSE, "CASH"):
        total += _finite_weight(allocation.get(symbol), label=f"frozen {symbol} allocation")
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        raise ForwardTestError("Frozen Forward Test allocation does not preserve all capital")
    expected_gate = apply_risk_gate(
        proposal.asset_weights(),
        UNIVERSE,
        max_position_weight=risk_limits["max_position_weight"],
        max_gross_exposure=risk_limits["max_gross_exposure"],
    )
    expected_allowed = {
        **expected_gate.executed_weights,
        "CASH": max(0.0, 1.0 - expected_gate.executed_gross_exposure),
    }
    for symbol, expected_weight in expected_allowed.items():
        if not math.isclose(float(allocation[symbol]), expected_weight, abs_tol=1e-9):
            raise ForwardTestError("Frozen Forward Test allocation does not match the Risk Gate")
    if not math.isclose(
        _finite_weight(frozen_decision.get("cash_weight"), label="frozen cash residual"),
        _finite_weight(allocation["CASH"], label="frozen cash allocation"),
        abs_tol=1e-9,
    ):
        raise ForwardTestError("Frozen Forward Test cash residual does not match its allocation")
    for symbol in (*UNIVERSE, BENCHMARK):
        _finite_price(prices.get(symbol), label=f"frozen {symbol} reference price")
        if _timestamp(bar_timestamps.get(symbol), label=f"frozen {symbol} evidence bar") != evidence_cutoff:
            raise ForwardTestError("Frozen Forward Test evidence bars do not share the cutoff")
    return dict(frozen_decision)


def compute_forward_performance(
    frozen_decision: Mapping[str, Any],
    prices: Mapping[str, pd.Series],
) -> dict[str, Any]:
    """Value a frozen allocation from strictly later common daily Yahoo observations."""

    frozen = _validated_frozen_decision(frozen_decision)
    cutoff = _timestamp(frozen["evidence_as_of"], label="evidence cutoff")
    allocation = frozen["allowed_allocation"]
    entry_prices = frozen["entry_prices"]
    required_symbols = _required_symbols(frozen)
    series_by_symbol: dict[str, pd.Series] = {}
    for symbol in required_symbols:
        if symbol not in prices:
            raise ForwardTestError(f"Forward price data is missing {symbol}")
        series_by_symbol[symbol] = _post_cutoff_daily_series(
            prices[symbol], cutoff=cutoff, symbol=symbol
        )

    aligned = pd.concat(series_by_symbol, axis=1, join="inner").sort_index()
    if aligned.empty:
        return _waiting_performance()

    cash_weight = _finite_weight(allocation["CASH"], label="frozen cash allocation")
    observations: list[dict[str, Any]] = []
    for timestamp, row in aligned.iterrows():
        portfolio_value = cash_weight
        observed_prices: dict[str, float] = {}
        for symbol in UNIVERSE:
            if symbol not in aligned.columns:
                continue
            close = float(row[symbol])
            observed_prices[symbol] = close
            portfolio_value += _finite_weight(
                allocation[symbol], label=f"frozen {symbol} allocation"
            ) * (close / _finite_price(entry_prices[symbol], label=f"{symbol} reference price"))
        spy_close = float(row[BENCHMARK])
        observed_prices[BENCHMARK] = spy_close
        portfolio_return = portfolio_value - 1.0
        spy_return = spy_close / _finite_price(entry_prices[BENCHMARK], label="SPY reference price") - 1.0
        observations.append(
            {
                "date": timestamp.date().isoformat(),
                "prices": observed_prices,
                "portfolio_cumulative_return": portfolio_return,
                "spy_cumulative_return": spy_return,
                "relative_return": portfolio_return - spy_return,
            }
        )

    horizons: dict[str, dict[str, Any]] = {}
    for days in FORWARD_HORIZONS:
        if len(observations) < days:
            horizons[str(days)] = {"status": "pending", "trading_day": days}
            continue
        observation = observations[days - 1]
        horizons[str(days)] = {
            "status": "observed",
            "trading_day": days,
            "date": observation["date"],
            "portfolio_cumulative_return": observation["portfolio_cumulative_return"],
            "spy_cumulative_return": observation["spy_cumulative_return"],
            "relative_return": observation["relative_return"],
        }
    return {
        "status": "observed",
        "observed_trading_days": len(observations),
        "observations": observations,
        "current": observations[-1],
        "horizons": horizons,
    }


def refresh_forward_test(
    forward_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    price_provider: Callable[..., Mapping[str, pd.Series]] = collect_forward_yahoo_prices,
    now: datetime | None = None,
) -> ForwardTestRun:
    """Refresh only post-cutoff outcomes; the frozen decision is never changed."""

    artifact_dir = _resolve_forward_artifact(forward_dir, project_root=project_root)
    frozen_decision = _validated_frozen_decision(_read_json(artifact_dir / FROZEN_DECISION_FILE))
    prices = price_provider(frozen_decision, now=now)
    performance = compute_forward_performance(frozen_decision, prices)
    performance_path = artifact_dir / FORWARD_PERFORMANCE_FILE
    if performance["observed_trading_days"] == 0 and performance_path.is_file():
        previous = _read_json(performance_path)
        try:
            prior_days = int(previous.get("observed_trading_days", 0))
        except (TypeError, ValueError):
            prior_days = 0
        # A transient empty response must not erase a genuinely observed local outcome. Before
        # the first post-cutoff bar, the normal waiting record remains the truthful result.
        if prior_days > 0:
            return ForwardTestRun(artifact_dir=artifact_dir, reused=False)
    _write_json_atomically(performance_path, performance)
    return ForwardTestRun(artifact_dir=artifact_dir, reused=False)


def find_latest_forward_test(*, project_root: Path = PROJECT_ROOT) -> Path | None:
    """Find a complete Forward Test without touching Yahoo, Gemini, or Today artifacts."""

    artifacts_root = project_root / "artifacts"
    latest_path = artifacts_root / "latest_forward.json"
    if latest_path.is_file():
        try:
            candidate = Path(json.loads(latest_path.read_text(encoding="utf-8"))["artifact_dir"])
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            candidate = None
        if candidate is not None:
            candidate = candidate.resolve()
            if _is_within_root(candidate, _forward_root(project_root)) and _is_completed_forward_test(
                candidate
            ):
                return candidate

    forward_root = _forward_root(project_root)
    if not forward_root.is_dir():
        return None
    for artifact_dir in sorted(forward_root.iterdir(), key=lambda path: path.name, reverse=True):
        if _is_completed_forward_test(artifact_dir):
            return artifact_dir.resolve()
    return None
