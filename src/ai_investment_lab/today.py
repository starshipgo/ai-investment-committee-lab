"""Current-data, analysis-only committee runs kept separate from backtests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from lumibot.components.agents import AgentRunResult, agent_tool
from lumibot.components.agents.manager import AgentManager
from lumibot.data_sources import YahooData

from .artifacts import append_jsonl, save_agent_decision, save_risk_decisions
from .backtest import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_GROSS_EXPOSURE,
    DEFAULT_MAX_POSITION_WEIGHT,
    DEFAULT_MODEL,
    MAX_LOOKBACK_DAYS,
    MAX_MAX_POSITION_WEIGHT,
    MIN_LOOKBACK_DAYS,
    MIN_MAX_POSITION_WEIGHT,
    PROJECT_ROOT,
    _require_gemini_key,
)
from .models import BENCHMARK, UNIVERSE, TargetWeightProposal
from .risk import apply_risk_gate
from .strategy import AIInvestmentCommitteeStrategy, recover_recorded_proposal

TODAY_ARTIFACT_DIRECTORY = "today"
TODAY_SUMMARY_FILE = "today_summary.json"
TODAY_COMPLETION_FILES = (
    TODAY_SUMMARY_FILE,
    "evidence.json",
    "agent_decisions.jsonl",
    "portfolio_proposals.jsonl",
    "risk_decisions.csv",
)
TODAY_ROLE_ORDER = ("Researcher", "Bull Analyst", "Bear Analyst", "Portfolio Manager")


class TodayEvidenceError(RuntimeError):
    """Current Yahoo evidence was not complete enough for an analysis-only run."""

    safe_category = "market_data"


@dataclass(frozen=True)
class TodayConfig:
    """Non-secret controls for one current-data, analysis-only observation."""

    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT
    max_gross_exposure: float = DEFAULT_MAX_GROSS_EXPOSURE

    def __post_init__(self) -> None:
        if isinstance(self.lookback_days, bool) or not isinstance(self.lookback_days, int):
            raise TypeError("lookback_days must be an integer")
        if not MIN_LOOKBACK_DAYS <= self.lookback_days <= MAX_LOOKBACK_DAYS:
            raise ValueError(
                f"lookback_days must be between {MIN_LOOKBACK_DAYS} and {MAX_LOOKBACK_DAYS}"
            )
        if not math.isfinite(self.max_position_weight) or not (
            MIN_MAX_POSITION_WEIGHT <= self.max_position_weight <= MAX_MAX_POSITION_WEIGHT
        ):
            raise ValueError(
                "max_position_weight must be between "
                f"{MIN_MAX_POSITION_WEIGHT:.0%} and {MAX_MAX_POSITION_WEIGHT:.0%}"
            )
        if not math.isfinite(self.max_gross_exposure) or not 0.0 < self.max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]")

    def artifact_config(self) -> dict[str, int | float]:
        return {
            "lookback_days": self.lookback_days,
            "max_position_weight": self.max_position_weight,
            "max_gross_exposure": self.max_gross_exposure,
        }


@dataclass(frozen=True)
class TodayEvidence:
    """Frozen current Yahoo evidence given to the read-only committee."""

    analysis_timestamp: str
    evidence_as_of: str
    lookback_days: int
    snapshot: dict[str, dict[str, int | float]]
    bar_timestamps: dict[str, str]
    fingerprint: str

    def tool_payload(self) -> dict[str, Any]:
        # Keep every value returned to an agent tied to the evidence fingerprint.  A new
        # wall-clock analysis timestamp would otherwise make a replayed tool result look stale
        # even when the completed-bar evidence is identical.
        return {
            "evidence_as_of": self.evidence_as_of,
            "lookback_days": self.lookback_days,
            "snapshot": self.snapshot,
        }

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp,
            **self.tool_payload(),
            "bar_timestamps": self.bar_timestamps,
            "evidence_fingerprint": self.fingerprint,
            "data_source": "YahooData",
        }


@dataclass(frozen=True)
class TodayAnalysisRun:
    """One completed current-data analysis artifact directory."""

    artifact_dir: Path


class _TodayVars:
    """The minimal Lumibot strategy-vars interface required by AgentManager."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value


def _utc_now(value: datetime | None = None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _timestamp_text(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise TodayEvidenceError("Current Yahoo evidence had an invalid bar timestamp")
    return timestamp.isoformat()


def _evidence_fingerprint(
    *,
    lookback_days: int,
    evidence_as_of: str,
    snapshot: dict[str, dict[str, int | float]],
    bar_timestamps: dict[str, str],
) -> str:
    payload = {
        "data_source": "YahooData",
        "lookback_days": lookback_days,
        "evidence_as_of": evidence_as_of,
        "snapshot": snapshot,
        "bar_timestamps": bar_timestamps,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_latest_yahoo_evidence(
    lookback_days: int,
    analysis_time: datetime | None = None,
) -> TodayEvidence:
    """Fetch latest *completed* daily Yahoo bars through the installed LumiBot source."""

    analysis_timestamp = _utc_now(analysis_time)
    # YahooData is a backtesting-shaped source. Supplying its current datetime explicitly is
    # necessary: default construction starts it roughly a year ago. Its daily implementation
    # excludes the in-progress daily session, so no separate timeshift is needed here.
    try:
        source = YahooData(
            auto_adjust=False,
            datetime_start=analysis_timestamp,
            datetime_end=analysis_timestamp + timedelta(minutes=1),
        )
    except Exception as exc:  # provider setup failures must not leak through the UI
        raise TodayEvidenceError("Yahoo did not provide a current daily evidence source") from exc
    snapshot: dict[str, dict[str, int | float]] = {}
    bar_timestamps: dict[str, str] = {}

    for symbol in (*UNIVERSE, BENCHMARK):
        try:
            bars = source.get_historical_prices(symbol, lookback_days, "day", timeshift=None)
        except Exception as exc:  # preserve the provider traceback while exposing only a safe category
            raise TodayEvidenceError(
                f"Yahoo did not provide completed daily evidence for {symbol}"
            ) from exc
        frame = getattr(bars, "pandas_df", None) if bars is not None else None
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise TodayEvidenceError(f"Yahoo did not return completed daily evidence for {symbol}")
        close_column = "close" if "close" in frame.columns else "Close"
        if close_column not in frame.columns:
            raise TodayEvidenceError(f"Yahoo evidence for {symbol} did not contain closing prices")
        closes = pd.to_numeric(frame[close_column], errors="coerce").dropna()
        if len(closes) < lookback_days:
            raise TodayEvidenceError(
                f"Yahoo returned fewer than {lookback_days} completed daily bars for {symbol}"
            )
        closes = closes.iloc[-lookback_days:]
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
        bar_timestamps[symbol] = _timestamp_text(closes.index[-1])

    evidence_dates = {pd.Timestamp(timestamp).date() for timestamp in bar_timestamps.values()}
    if len(evidence_dates) != 1:
        raise TodayEvidenceError("Yahoo returned mismatched latest completed sessions across the universe")
    evidence_as_of = min(bar_timestamps.values())
    fingerprint = _evidence_fingerprint(
        lookback_days=lookback_days,
        evidence_as_of=evidence_as_of,
        snapshot=snapshot,
        bar_timestamps=bar_timestamps,
    )
    return TodayEvidence(
        analysis_timestamp=analysis_timestamp.isoformat(),
        evidence_as_of=evidence_as_of,
        lookback_days=lookback_days,
        snapshot=snapshot,
        bar_timestamps=bar_timestamps,
        fingerprint=fingerprint,
    )


class TodayCommittee:
    """A tiny Lumibot AgentManager host with no broker or order API."""

    # LumiBot only activates its local agent replay cache in backtesting mode. This isolated
    # analysis host has no broker and never invokes a strategy lifecycle or order method; the
    # flag only allows a frozen evidence fingerprint to replay safely.
    is_backtesting = True
    market = "US equities"
    timezone = "UTC"

    def __init__(
        self,
        *,
        evidence: TodayEvidence,
        artifact_dir: Path,
        model: str,
        agent_manager_factory: Callable[[Any], Any] = AgentManager,
    ) -> None:
        self.name = "TodayCommittee"
        self._evidence = evidence
        self._evidence_datetime = pd.Timestamp(evidence.evidence_as_of).to_pydatetime()
        self.parameters: dict[str, Any] = {
            "universe": list(UNIVERSE),
            "benchmark": BENCHMARK,
            "lookback_days": evidence.lookback_days,
            "agent_max_model_calls": 4,
        }
        self.vars = _TodayVars()
        # This file is only a path anchor for LumiBot's optional agent-detail artifacts.
        self.stats_file = str(artifact_dir / "today_stats.csv")
        self._model = model
        self.agents = agent_manager_factory(self)

    def get_datetime(self) -> datetime:
        return self._evidence_datetime

    def log_message(self, _message: str, **_kwargs: Any) -> None:
        """Keep agent observability local to saved artifacts, not the Streamlit surface."""

    @agent_tool(
        name="get_current_universe_snapshot",
        description=(
            "Return the frozen current Yahoo daily evidence for AAPL, MSFT, NVDA, and SPY. "
            "It contains the latest completed daily bars only and is read-only."
        ),
    )
    def get_current_universe_snapshot(self) -> dict[str, Any]:
        return self._evidence.tool_payload()

    @staticmethod
    def _decision_text(result: AgentRunResult) -> str:
        return (result.summary or result.text or "").strip()

    def _create_agents(self) -> None:
        self.agents.create(
            name="researcher",
            model=self._model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            tools=[self.get_current_universe_snapshot],
            system_prompt=(
                "You are the Researcher in a current-data, analysis-only learning exercise. "
                "Use the frozen Yahoo snapshot to compare AAPL, MSFT, and NVDA against SPY; "
                "report facts and uncertainty, never orders or price/return predictions."
            ),
        )
        self.agents.create(
            name="bull_analyst",
            model=self._model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            tools=[],
            system_prompt=(
                "You are the Bull Analyst in a current-data, analysis-only learning exercise. "
                "Build the strongest evidence-based case from the Researcher handoff, while remaining "
                "read-only, never placing orders, and never forecasting prices or returns."
            ),
        )
        self.agents.create(
            name="bear_analyst",
            model=self._model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            tools=[],
            system_prompt=(
                "You are the Bear Analyst in a current-data, analysis-only learning exercise. "
                "Challenge the evidence and identify uncertainty, concentration, and drawdown risks; "
                "remain read-only, never place orders, and never forecast prices or returns."
            ),
        )
        # Keep the historical PM tool surface and typed contract. Its legacy in-memory mutation
        # is intentionally ignored: run() always recovers the canonical recorded AgentRunResult.
        proposal_tool = AIInvestmentCommitteeStrategy.record_target_weight_proposal.__get__(
            self, type(self)
        )
        self.agents.create(
            name="portfolio_manager",
            model=self._model,
            allow_trading=False,
            include_builtin_tools=False,
            include_builtin_skills=False,
            tools=[proposal_tool],
            system_prompt=(
                "You are the Portfolio Manager in a current-data, analysis-only learning exercise. "
                "Synthesize the three handoffs into raw long-only target weights; you may only record "
                "a proposal and can never place an order. Do not forecast prices or returns."
            ),
        )

    def run(self) -> dict[str, AgentRunResult]:
        """Run the same four non-trading roles against one frozen current evidence snapshot."""

        self._create_agents()
        context = {
            "analysis_type": "today_analysis",
            "analysis_only": True,
            "no_orders": True,
            "evidence_as_of": self._evidence.evidence_as_of,
            "evidence_fingerprint": self._evidence.fingerprint,
            "universe": list(UNIVERSE),
            "benchmark": BENCHMARK,
            "lookback_days": self._evidence.lookback_days,
        }
        researcher = self.agents["researcher"].run(
            task_prompt=(
                "Call get_current_universe_snapshot once. It contains the configured "
                f"{self._evidence.lookback_days} latest completed Yahoo daily bars. Then produce a "
                "compact evidence table and rank the three stocks. Do not recommend an order or predict "
                "future prices or returns."
            ),
            context=context,
        )
        bull = self.agents["bull_analyst"].run(
            task_prompt=(
                "Make the strongest concise current-evidence case for one candidate. Do not forecast "
                "future prices or returns."
            ),
            context={**context, "researcher_handoff": self._decision_text(researcher)},
        )
        bear = self.agents["bear_analyst"].run(
            task_prompt=(
                "Stress-test the current evidence and the Bull Analyst's preferred candidate. Do not "
                "forecast future prices or returns."
            ),
            context={
                **context,
                "researcher_handoff": self._decision_text(researcher),
                "bull_handoff": self._decision_text(bull),
            },
        )
        portfolio_manager = self.agents["portfolio_manager"].run(
            task_prompt=(
                "Call record_target_weight_proposal exactly once. The four weights must sum to 1.0. "
                "State raw conviction weights without applying any position cap—the deterministic Python "
                "risk gate owns all limits. Do not place or request orders, and do not forecast future "
                "prices or returns."
            ),
            context={
                **context,
                "researcher_handoff": self._decision_text(researcher),
                "bull_handoff": self._decision_text(bull),
                "bear_handoff": self._decision_text(bear),
            },
        )
        return {
            "Researcher": researcher,
            "Bull Analyst": bull,
            "Bear Analyst": bear,
            "Portfolio Manager": portfolio_manager,
        }


def _new_today_artifact_dir(project_root: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = project_root / "artifacts" / TODAY_ARTIFACT_DIRECTORY / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_completed_today_analysis(path: Path) -> bool:
    if not path.is_dir() or not all((path / filename).is_file() for filename in TODAY_COMPLETION_FILES):
        return False
    try:
        summary = json.loads((path / TODAY_SUMMARY_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("artifact_kind") == "today_analysis" and summary.get("order_submitted") is False


def find_latest_today_analysis(*, project_root: Path = PROJECT_ROOT) -> Path | None:
    """Find a completed Today artifact without touching Yahoo, Gemini, or historical runs."""

    artifacts_root = project_root / "artifacts"
    latest_path = artifacts_root / "latest_today.json"
    if latest_path.is_file():
        try:
            latest_dir = Path(json.loads(latest_path.read_text(encoding="utf-8"))["artifact_dir"])
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            latest_dir = None
        if latest_dir is not None and _is_completed_today_analysis(latest_dir):
            return latest_dir.resolve()

    today_root = artifacts_root / TODAY_ARTIFACT_DIRECTORY
    if not today_root.is_dir():
        return None
    for artifact_dir in sorted(today_root.iterdir(), key=lambda path: path.name, reverse=True):
        if _is_completed_today_analysis(artifact_dir):
            return artifact_dir.resolve()
    return None


def run_today_analysis(
    config: TodayConfig,
    *,
    project_root: Path = PROJECT_ROOT,
    evidence_provider: Callable[[int, datetime | None], TodayEvidence] = collect_latest_yahoo_evidence,
    committee_factory: Callable[..., TodayCommittee] = TodayCommittee,
    now: datetime | None = None,
) -> TodayAnalysisRun:
    """Create one new, analysis-only Today artifact after an explicit user action."""

    load_dotenv(project_root / ".env")
    _require_gemini_key()
    os.environ.setdefault("AI_MODEL", DEFAULT_MODEL)
    os.environ.setdefault("LUMIBOT_CACHE_FOLDER", str(project_root / ".cache" / "lumibot"))

    analysis_time = _utc_now(now)
    evidence = evidence_provider(config.lookback_days, analysis_time)
    if evidence.lookback_days != config.lookback_days:
        raise TodayEvidenceError("Current evidence did not match the requested lookback window")

    artifact_dir = _new_today_artifact_dir(project_root)
    _write_json(artifact_dir / "evidence.json", evidence.artifact_payload())
    committee = committee_factory(
        evidence=evidence,
        artifact_dir=artifact_dir,
        model=os.environ["AI_MODEL"],
    )
    results = committee.run()
    if not isinstance(results, dict) or set(results) != set(TODAY_ROLE_ORDER):
        raise RuntimeError("Today Committee did not return all four required agent results")
    for role in TODAY_ROLE_ORDER:
        result = results[role]
        if not isinstance(result, AgentRunResult):
            raise TypeError(f"Today Committee returned an invalid result for {role}")
        save_agent_decision(
            artifact_dir / "agent_decisions.jsonl",
            role=role,
            as_of=evidence.analysis_timestamp,
            result=result,
        )

    # This is intentionally unconditional for cold calls and replay-cache hits alike.
    proposal: TargetWeightProposal = recover_recorded_proposal(results["Portfolio Manager"])
    append_jsonl(
        artifact_dir / "portfolio_proposals.jsonl",
        {"as_of": evidence.analysis_timestamp, **proposal.model_dump()},
    )
    risk_result = apply_risk_gate(
        proposal.asset_weights(),
        UNIVERSE,
        max_position_weight=config.max_position_weight,
        max_gross_exposure=config.max_gross_exposure,
    )
    save_risk_decisions(
        artifact_dir / "risk_decisions.csv",
        as_of=evidence.analysis_timestamp,
        result=risk_result,
        proposed_cash_weight=proposal.cash_weight,
    )

    summary = {
        "artifact_kind": "today_analysis",
        "project_version": "0.1.0",
        "lumibot_version": __import__("lumibot").__version__,
        "model": os.environ["AI_MODEL"],
        "data_source": "YahooData",
        "analysis_timestamp": evidence.analysis_timestamp,
        "evidence_as_of": evidence.evidence_as_of,
        "evidence_fingerprint": evidence.fingerprint,
        "universe": list(UNIVERSE),
        "benchmark": BENCHMARK,
        "lookback_days": config.lookback_days,
        "risk_limits": {
            "max_position_weight": config.max_position_weight,
            "max_gross_exposure": config.max_gross_exposure,
        },
        "analysis_only": True,
        "order_submitted": False,
        "artifact_dir": str(artifact_dir),
    }
    # The summary and pointer are written last; incomplete/failed analyses never look successful.
    _write_json(artifact_dir / TODAY_SUMMARY_FILE, summary)
    _write_json(project_root / "artifacts" / "latest_today.json", {"artifact_dir": str(artifact_dir)})
    return TodayAnalysisRun(artifact_dir=artifact_dir)
