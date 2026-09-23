"""Backtest coordinator for local, repeatable learning experiments."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lumibot.backtesting import YahooDataBacktesting

from .artifacts import export_performance
from .models import BENCHMARK
from .strategy import AIInvestmentCommitteeStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_LOOKBACK_DAYS = 20
MIN_LOOKBACK_DAYS = 10
MAX_LOOKBACK_DAYS = 120
DEFAULT_MAX_POSITION_WEIGHT = 0.25
MIN_MAX_POSITION_WEIGHT = 0.05
MAX_MAX_POSITION_WEIGHT = 0.50
DEFAULT_MAX_GROSS_EXPOSURE = 1.0
DEFAULT_INITIAL_BUDGET = 100_000.0
DEFAULT_START = date(2025, 1, 2)
DEFAULT_END = date(2025, 4, 1)
COMPLETION_FILES = (
    "run_summary.json",
    "metrics.json",
    "equity_curve.csv",
    "agent_decisions.jsonl",
    "risk_decisions.csv",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """The complete non-secret configuration for one historical experiment."""

    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT
    max_gross_exposure: float = DEFAULT_MAX_GROSS_EXPOSURE
    start: date = DEFAULT_START
    end: date = DEFAULT_END
    initial_budget: float = DEFAULT_INITIAL_BUDGET

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
        if not math.isfinite(self.initial_budget) or self.initial_budget <= 0.0:
            raise ValueError("initial_budget must be positive")
        if self.end <= self.start:
            raise ValueError("backtest end must be after start")

    def artifact_config(self) -> dict[str, int | float | str]:
        """Return stable, non-secret data for summary/reuse matching."""

        return {
            "lookback_days": self.lookback_days,
            "max_position_weight": self.max_position_weight,
            "max_gross_exposure": self.max_gross_exposure,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_budget": self.initial_budget,
        }


@dataclass(frozen=True)
class ExperimentRun:
    """A completed artifact directory and whether it avoided a new API call."""

    artifact_dir: Path
    reused: bool


def _parse_date(name: str, default: date) -> date:
    raw = os.environ.get(name, default.isoformat())
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date such as 2025-01-02; got {raw!r}") from exc


def _parse_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


def _parse_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}") from exc


def _require_gemini_key() -> None:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if not value or value == "replace_me":
        raise RuntimeError(
            "GEMINI_API_KEY is required. Copy .env.example to .env and add a key from "
            "https://aistudio.google.com/apikey"
        )


def _is_completed_run(path: Path) -> bool:
    return path.is_dir() and all((path / filename).is_file() for filename in COMPLETION_FILES)


def _matches_config(summary: dict[str, Any], config: ExperimentConfig) -> bool:
    """Compare a completed summary without loading credentials or calling a model."""

    experiment = summary.get("experiment")
    stored = experiment if isinstance(experiment, dict) else {}
    risk_limits = summary.get("risk_limits", {})
    try:
        lookback_days = int(stored.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
        max_position_weight = float(
            stored.get(
                "max_position_weight",
                risk_limits.get("max_position_weight", DEFAULT_MAX_POSITION_WEIGHT),
            )
        )
        max_gross_exposure = float(
            stored.get(
                "max_gross_exposure",
                risk_limits.get("max_gross_exposure", DEFAULT_MAX_GROSS_EXPOSURE),
            )
        )
        # Pre-Milestone-3 summaries did not save initial_budget. Their first portfolio
        # value can differ from the budget after the first simulated market move, so
        # recognize the original default run by its documented default configuration.
        initial_budget = float(stored.get("initial_budget", DEFAULT_INITIAL_BUDGET))
    except (TypeError, ValueError):
        return False

    stored_start = str(stored.get("start", summary.get("start", "")))
    stored_end = str(stored.get("end", summary.get("end", "")))
    return (
        lookback_days == config.lookback_days
        and math.isclose(max_position_weight, config.max_position_weight, abs_tol=1e-12)
        and math.isclose(max_gross_exposure, config.max_gross_exposure, abs_tol=1e-12)
        and math.isclose(initial_budget, config.initial_budget, abs_tol=1e-6)
        and stored_start == config.start.isoformat()
        and stored_end == config.end.isoformat()
    )


def find_completed_experiment(
    config: ExperimentConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path | None:
    """Find the newest successful matching run without touching Gemini or .env."""

    runs_dir = project_root / "artifacts" / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.iterdir(), key=lambda path: path.name, reverse=True)
    for artifact_dir in candidates:
        if not _is_completed_run(artifact_dir):
            continue
        try:
            summary = json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _matches_config(summary, config):
            return artifact_dir.resolve()
    return None


def _new_artifact_dir(project_root: Path) -> Path:
    """Create a unique timestamped directory without overwriting prior experiments."""

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = project_root / "artifacts" / "runs" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def run_experiment(config: ExperimentConfig, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Run one new historical experiment after explicit user authorization."""

    load_dotenv(project_root / ".env")
    _require_gemini_key()

    os.environ.setdefault("AI_MODEL", DEFAULT_MODEL)
    os.environ.setdefault("LUMIBOT_CACHE_FOLDER", str(project_root / ".cache" / "lumibot"))
    artifact_dir = _new_artifact_dir(project_root)
    start = datetime.combine(config.start, time.min)
    end = datetime.combine(config.end, time.min)

    _, strategy = AIInvestmentCommitteeStrategy.run_backtest(
        YahooDataBacktesting,
        backtesting_start=start,
        backtesting_end=end,
        benchmark_asset=BENCHMARK,
        budget=config.initial_budget,
        parameters={
            "artifact_dir": str(artifact_dir),
            "lookback_days": config.lookback_days,
            "max_position_weight": config.max_position_weight,
            "max_gross_exposure": config.max_gross_exposure,
        },
        stats_file=str(artifact_dir / "portfolio_stats.csv"),
        trades_file=str(artifact_dir / "lumibot_trades.csv"),
        show_plot=False,
        show_tearsheet=False,
        show_indicators=False,
        save_tearsheet=False,
        show_progress_bar=False,
        quiet_logs=False,
        save_stats_file=True,
    )
    metrics = export_performance(strategy, artifact_dir, config.initial_budget)

    summary = {
        "project_version": "0.1.0",
        "lumibot_version": __import__("lumibot").__version__,
        "model": os.environ["AI_MODEL"],
        "data_source": "YahooDataBacktesting",
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "universe": list(AIInvestmentCommitteeStrategy.parameters["universe"]),
        "benchmark": BENCHMARK,
        "risk_limits": {
            "max_position_weight": config.max_position_weight,
            "max_gross_exposure": config.max_gross_exposure,
        },
        "experiment": {
            **config.artifact_config(),
            "model": os.environ["AI_MODEL"],
            "universe": list(AIInvestmentCommitteeStrategy.parameters["universe"]),
            "benchmark": BENCHMARK,
            "data_source": "YahooDataBacktesting",
        },
        "metrics": metrics,
        "artifact_dir": str(artifact_dir),
    }
    (artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (project_root / "artifacts" / "latest_run.json").write_text(
        json.dumps({"artifact_dir": str(artifact_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def run_or_reuse_experiment(
    config: ExperimentConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> ExperimentRun:
    """Prefer a completed identical run before creating a new model-backed experiment."""

    existing = find_completed_experiment(config, project_root=project_root)
    if existing is not None:
        return ExperimentRun(artifact_dir=existing, reused=True)
    return ExperimentRun(artifact_dir=run_experiment(config, project_root=project_root), reused=False)


def _config_from_environment() -> ExperimentConfig:
    return ExperimentConfig(
        lookback_days=_parse_int("LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS),
        max_position_weight=_parse_float("MAX_POSITION_WEIGHT", DEFAULT_MAX_POSITION_WEIGHT),
        max_gross_exposure=_parse_float("MAX_GROSS_EXPOSURE", DEFAULT_MAX_GROSS_EXPOSURE),
        start=_parse_date("BACKTESTING_START", DEFAULT_START),
        end=_parse_date("BACKTESTING_END", DEFAULT_END),
        initial_budget=_parse_float("INITIAL_BUDGET", DEFAULT_INITIAL_BUDGET),
    )


def main() -> None:
    try:
        result = run_or_reuse_experiment(_config_from_environment())
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    action = "Reused completed backtest" if result.reused else "Backtest complete"
    print(f"\n{action}.")
    print(f"Artifacts: {result.artifact_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
