"""Local learning interface with artifact-only page loads for AI Investment Lab."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_MAX_POSITION_PERCENT = 25
HISTORICAL_LAB = "Historical Lab"
TODAY_MODE = "Today"
FORWARD_TEST_MODE = "Forward Test"
TODAY_SUMMARY_FILE = "today_summary.json"
FORWARD_SUMMARY_FILE = "forward_summary.json"
FROZEN_DECISION_FILE = "frozen_decision.json"
FORWARD_PERFORMANCE_FILE = "forward_performance.json"
ROLE_ORDER = ("Researcher", "Bull Analyst", "Bear Analyst", "Portfolio Manager")
ROLE_DESCRIPTIONS = {
    "Researcher": "Read-only market evidence",
    "Bull Analyst": "Read-only upside case",
    "Bear Analyst": "Read-only risk challenge",
    "Portfolio Manager": "Proposal only — cannot trade",
}
TABLE_DIVIDER = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")
RESULT_MARKER = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?result\s*:\s*")


@dataclass(frozen=True)
class AgentCard:
    """Presentation-only content for one saved agent decision."""

    role: str
    description: str
    preview: str
    details: str


@dataclass(frozen=True)
class ExperimentSettings:
    """The two user-controlled inputs for a new learning experiment."""

    lookback_days: int
    max_position_weight: float


@dataclass(frozen=True)
class TodaySettings:
    """The two controls for one explicit current-data committee observation."""

    lookback_days: int
    max_position_weight: float


@dataclass(frozen=True)
class RiskGateRow:
    """Presentation-only version of one deterministic risk decision."""

    symbol: str
    proposed_weight: float
    executed_weight: float
    reason: str
    overridden: bool


@dataclass(frozen=True)
class RunView:
    """A stable adapter between saved artifacts and Streamlit components."""

    run_label: str
    artifact_id: str
    data_source: str
    lookback_window: str
    max_position_size: str
    max_gross_exposure: str
    period: str
    total_return: float
    benchmark_return: float
    max_drawdown: float
    sharpe_ratio: float
    equity_chart: pd.DataFrame
    agent_cards: tuple[AgentCard, ...]
    risk_rows: tuple[RiskGateRow, ...]
    trades_table: pd.DataFrame


@dataclass(frozen=True)
class TodayView:
    """Presentation-only content for one saved current-data analysis."""

    run_label: str
    artifact_id: str
    data_source: str
    analysis_timestamp: str
    evidence_as_of: str
    lookback_window: str
    max_position_size: str
    max_gross_exposure: str
    cash_residual: float
    agent_cards: tuple[AgentCard, ...]
    risk_rows: tuple[RiskGateRow, ...]


@dataclass(frozen=True)
class ForwardHorizon:
    """One saved, observed-or-pending Forward Test milestone."""

    trading_day: int
    status: str
    date: str | None
    portfolio_return: float | None
    benchmark_return: float | None
    relative_return: float | None


@dataclass(frozen=True)
class ForwardTestView:
    """Presentation-only data for a frozen Today decision and its later observations."""

    run_label: str
    artifact_id: str
    data_source: str
    original_analysis_timestamp: str
    evidence_as_of: str
    lookback_window: str
    max_position_size: str
    max_gross_exposure: str
    cash_residual: float
    risk_rows: tuple[RiskGateRow, ...]
    observed_trading_days: int
    current_portfolio_return: float | None
    current_benchmark_return: float | None
    current_relative_return: float | None
    current_observation_date: str | None
    horizons: tuple[ForwardHorizon, ...]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_run_dir() -> Path:
    """Resolve the latest completed local run without ever consulting credentials."""

    latest_path = ARTIFACTS_ROOT / "latest_run.json"
    if latest_path.exists():
        artifact_dir = Path(_read_json(latest_path)["artifact_dir"]).resolve()
        if (artifact_dir / "run_summary.json").exists():
            return artifact_dir

    candidates = sorted(
        (path for path in (ARTIFACTS_ROOT / "runs").glob("*") if (path / "run_summary.json").exists()),
        key=lambda path: path.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No completed backtest runs were found under artifacts/runs.")


def _is_today_analysis_dir(path: Path) -> bool:
    """Recognize only completed Today artifacts, never a historical backtest."""

    summary_path = path / TODAY_SUMMARY_FILE
    if not summary_path.is_file():
        return False
    try:
        summary = _read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("artifact_kind") == "today_analysis" and summary.get("order_submitted") is False


def _safe_today_dir() -> Path:
    """Resolve the latest completed current-data analysis without model or market calls."""

    latest_path = ARTIFACTS_ROOT / "latest_today.json"
    if latest_path.exists():
        try:
            artifact_dir = Path(_read_json(latest_path)["artifact_dir"]).resolve()
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            artifact_dir = None
        if artifact_dir is not None and _is_today_analysis_dir(artifact_dir):
            return artifact_dir

    candidates = sorted(
        (path for path in (ARTIFACTS_ROOT / "today").glob("*") if _is_today_analysis_dir(path)),
        key=lambda path: path.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No completed Today analyses were found under artifacts/today.")


def _is_forward_test_dir(path: Path) -> bool:
    """Recognize only a complete, separate Forward Test artifact."""

    summary_path = path / FORWARD_SUMMARY_FILE
    frozen_path = path / FROZEN_DECISION_FILE
    if not summary_path.is_file() or not frozen_path.is_file():
        return False
    try:
        summary = _read_json(summary_path)
        frozen = _read_json(frozen_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("artifact_kind") == "forward_test"
        and summary.get("analysis_only") is True
        and summary.get("order_submitted") is False
        and frozen.get("artifact_kind") == "forward_test_frozen_decision"
    )


def _is_forward_child(path: Path) -> bool:
    return path.resolve().parent == (ARTIFACTS_ROOT / "forward").resolve()


def _safe_forward_dir() -> Path:
    """Resolve a saved Forward Test without fetching Yahoo or calling any agent."""

    latest_path = ARTIFACTS_ROOT / "latest_forward.json"
    if latest_path.exists():
        try:
            artifact_dir = Path(_read_json(latest_path)["artifact_dir"]).resolve()
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            artifact_dir = None
        if artifact_dir is not None and _is_forward_child(artifact_dir) and _is_forward_test_dir(
            artifact_dir
        ):
            return artifact_dir

    candidates = sorted(
        (
            path
            for path in (ARTIFACTS_ROOT / "forward").glob("*")
            if _is_forward_child(path) and _is_forward_test_dir(path)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No saved Forward Tests were found under artifacts/forward.")


@st.cache_data(show_spinner=False)
def load_run(run_dir_text: str) -> dict[str, Any]:
    """Load only saved artifact files; this function makes no model or network calls."""

    run_dir = Path(run_dir_text)
    decisions: list[dict[str, Any]] = []
    decisions_path = run_dir / "agent_decisions.jsonl"
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                decisions.append(json.loads(line))

    return {
        "summary": _read_json(run_dir / "run_summary.json"),
        "metrics": _read_json(run_dir / "metrics.json"),
        "equity": pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"]),
        "risk": pd.read_csv(run_dir / "risk_decisions.csv"),
        "trades": pd.read_csv(run_dir / "trades.csv")
        if (run_dir / "trades.csv").exists()
        else pd.DataFrame(),
        "decisions": decisions,
    }


@st.cache_data(show_spinner=False)
def load_today_analysis(analysis_dir_text: str) -> dict[str, Any]:
    """Load only a saved Today artifact; no market or model call can occur here."""

    analysis_dir = Path(analysis_dir_text)
    decisions: list[dict[str, Any]] = []
    decisions_path = analysis_dir / "agent_decisions.jsonl"
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                decisions.append(json.loads(line))

    return {
        "summary": _read_json(analysis_dir / TODAY_SUMMARY_FILE),
        "evidence": _read_json(analysis_dir / "evidence.json"),
        "risk": pd.read_csv(analysis_dir / "risk_decisions.csv"),
        "decisions": decisions,
    }


def _forward_artifact_revision(forward_dir: Path) -> tuple[int, ...]:
    """Give the cached artifact reader a new key after an explicit outcome refresh."""

    return tuple(
        (forward_dir / filename).stat().st_mtime_ns
        if (forward_dir / filename).is_file()
        else 0
        for filename in (FORWARD_SUMMARY_FILE, FROZEN_DECISION_FILE, FORWARD_PERFORMANCE_FILE)
    )


@st.cache_data(show_spinner=False)
def load_forward_test(forward_dir_text: str, revision: tuple[int, ...]) -> dict[str, Any]:
    """Load only frozen/saved Forward Test artifacts; this function never fetches prices."""

    del revision  # Its only purpose is to invalidate this artifact-only cache after a refresh.
    forward_dir = Path(forward_dir_text)
    performance_path = forward_dir / FORWARD_PERFORMANCE_FILE
    return {
        "summary": _read_json(forward_dir / FORWARD_SUMMARY_FILE),
        "frozen": _read_json(forward_dir / FROZEN_DECISION_FILE),
        "performance": _read_json(performance_path)
        if performance_path.is_file()
        else {
            "status": "waiting_for_future_data",
            "observed_trading_days": 0,
            "current": None,
            "horizons": {},
        },
    }


def _run_selected_experiment(settings: ExperimentSettings) -> Any:
    """Lazily import Lumibot/Gemini code only after the button is clicked."""

    from ai_investment_lab.backtest import ExperimentConfig, run_or_reuse_experiment

    return run_or_reuse_experiment(
        ExperimentConfig(
            lookback_days=settings.lookback_days,
            max_position_weight=settings.max_position_weight,
        )
    )


def _run_today_analysis(settings: TodaySettings) -> Any:
    """Import current-data/model code only after the Today button is clicked."""

    from ai_investment_lab.today import TodayConfig, run_today_analysis

    return run_today_analysis(
        TodayConfig(
            lookback_days=settings.lookback_days,
            max_position_weight=settings.max_position_weight,
        )
    )


def _create_forward_test(today_dir: Path) -> Any:
    """Lazily import the frozen-artifact workflow only after an explicit Today action."""

    from ai_investment_lab.forward import create_forward_test_from_today

    return create_forward_test_from_today(today_dir)


def _refresh_forward_test(forward_dir: Path) -> Any:
    """Lazily fetch only later completed Yahoo bars after an explicit refresh action."""

    from ai_investment_lab.forward import refresh_forward_test

    return refresh_forward_test(forward_dir)


def run_experiment_if_requested(
    run_requested: bool,
    settings: ExperimentSettings,
    *,
    runner: Callable[[ExperimentSettings], Any] | None = None,
) -> Any | None:
    """Keep page loads and slider changes incapable of starting a model-backed run."""

    if not run_requested:
        return None
    return (runner or _run_selected_experiment)(settings)


def run_today_if_requested(
    run_requested: bool,
    settings: TodaySettings,
    *,
    runner: Callable[[TodaySettings], Any] | None = None,
) -> Any | None:
    """Keep Today page loads and slider changes incapable of starting an analysis."""

    if not run_requested:
        return None
    return (runner or _run_today_analysis)(settings)


def create_forward_test_if_requested(
    creation_requested: bool,
    today_dir: Path,
    *,
    runner: Callable[[Path], Any] | None = None,
) -> Any | None:
    """Keep a Today page load incapable of creating a Forward Test or touching an agent."""

    if not creation_requested:
        return None
    return (runner or _create_forward_test)(today_dir)


def refresh_forward_test_if_requested(
    refresh_requested: bool,
    forward_dir: Path,
    *,
    runner: Callable[[Path], Any] | None = None,
) -> Any | None:
    """Keep Forward Test page loads incapable of fetching Yahoo observations."""

    if not refresh_requested:
        return None
    return (runner or _refresh_forward_test)(forward_dir)


def safe_experiment_failure(error: Exception) -> tuple[str, str]:
    """Classify a local experiment failure without exposing provider or secret details."""

    if getattr(error, "safe_category", None) == "portfolio_proposal":
        return (
            "Portfolio proposal rejected",
            (
                "The Portfolio Manager did not produce a valid recorded target-weight proposal. "
                "No orders were submitted; the last successful run remains displayed."
            ),
        )
    if getattr(error, "safe_category", None) == "market_data":
        return (
            "Current market data unavailable",
            (
                "Yahoo did not provide a complete current daily evidence snapshot. "
                "No completed Today analysis was saved; the last successful analysis remains displayed."
            ),
        )
    if isinstance(error, RuntimeError) and str(error).startswith("GEMINI_API_KEY is required"):
        return (
            "Gemini configuration unavailable",
            (
                "A valid local Gemini API key is required to run a new experiment. "
                "No completed run was saved."
            ),
        )
    if isinstance(error, (TypeError, ValueError)):
        return (
            "Experiment settings invalid",
            "The selected experiment settings are not valid. Adjust the controls and try again.",
        )
    return (
        "Experiment did not complete",
        "No completed run was saved. The last successful run remains displayed.",
    )


def safe_forward_failure(error: Exception, *, action: str) -> tuple[str, str]:
    """Classify Forward Test failures without exposing provider details or local paths."""

    if getattr(error, "safe_category", None) == "market_data":
        return (
            "Current market data unavailable",
            (
                "Yahoo did not provide a usable completed daily observation. The frozen decision "
                "and its last saved outcome remain unchanged."
            ),
        )
    if getattr(error, "safe_category", None) == "forward_test":
        return (
            "Forward Test could not be updated",
            "The saved decision could not be safely used for this Forward Test. No order was submitted.",
        )
    return (
        "Forward Test did not complete",
        f"The {action} did not change the saved decision or submit an order.",
    )


def render_experiment_controls() -> tuple[ExperimentSettings, bool]:
    """Render the only interactive parameters and the explicit run action."""

    st.caption(
        "Changes below create or reuse a separate experiment. They never modify the saved "
        "result displayed above."
    )
    lookback_column, max_position_column = st.columns(2)
    with lookback_column:
        lookback_days = st.slider(
            "Lookback window",
            min_value=10,
            max_value=120,
            value=DEFAULT_LOOKBACK_DAYS,
            step=1,
        )
        st.caption("Changes what evidence the AI sees.")
    with max_position_column:
        max_position_percent = st.slider(
            "Max position size",
            min_value=5,
            max_value=50,
            value=DEFAULT_MAX_POSITION_PERCENT,
            step=1,
            format="%d%%",
        )
        st.caption("Changes what the AI is allowed to execute.")

    st.info(
        "Run New Experiment may invoke the Gemini model API. An identical completed experiment "
        "is loaded instead when available."
    )
    run_requested = st.button("Run New Experiment", type="primary")
    return (
        ExperimentSettings(
            lookback_days=lookback_days,
            max_position_weight=max_position_percent / 100.0,
        ),
        run_requested,
    )


def render_today_controls() -> tuple[TodaySettings, bool]:
    """Render the explicit controls for a current-data analysis-only observation."""

    st.caption(
        "Changes below create a separate analysis. They never modify the saved analysis "
        "displayed above."
    )
    lookback_column, max_position_column = st.columns(2)
    with lookback_column:
        lookback_days = st.slider(
            "Lookback window",
            min_value=10,
            max_value=120,
            value=DEFAULT_LOOKBACK_DAYS,
            step=1,
            key="today_lookback_days",
        )
        st.caption("Changes the latest completed daily evidence the AI sees.")
    with max_position_column:
        max_position_percent = st.slider(
            "Max position size",
            min_value=5,
            max_value=50,
            value=DEFAULT_MAX_POSITION_PERCENT,
            step=1,
            format="%d%%",
            key="today_max_position_percent",
        )
        st.caption("Changes only what deterministic Python allows to pass the Risk Gate.")

    st.info(
        "Run Today's Committee may invoke the Gemini model API and fetch current Yahoo data. "
        "It is analysis only and cannot submit an order."
    )
    run_requested = st.button("Run Today's Committee", type="primary")
    return (
        TodaySettings(
            lookback_days=lookback_days,
            max_position_weight=max_position_percent / 100.0,
        ),
        run_requested,
    )


def run_requested_experiment(settings: ExperimentSettings) -> Path | None:
    """Run or reuse only after an explicit click, with a minimal visible status."""

    with st.status("Running experiment…", expanded=True) as status:
        status.write("Checking saved experiments before calling the model API.")
        try:
            result = run_experiment_if_requested(True, settings)
        except Exception as exc:  # noqa: BLE001 - render only the safe category below
            category, message = safe_experiment_failure(exc)
            status.update(label=category, state="error")
            st.error(message)
            return None

        artifact_dir = Path(result.artifact_dir).resolve()
        if not (artifact_dir / "run_summary.json").is_file():
            status.update(label="Incomplete experiment artifact", state="error")
            st.error(
                "The experiment did not produce a completed artifact run. "
                "The last successful run remains displayed."
            )
            return None
        if result.reused:
            status.update(label="Loaded matching completed experiment", state="complete")
            status.write("A matching completed backtest was reused.")
        else:
            status.update(label="Experiment complete", state="complete")
            status.write("A new timestamped backtest artifact was saved.")
        return artifact_dir


def run_requested_today_analysis(settings: TodaySettings) -> Path | None:
    """Run exactly one Today observation only after its explicit button click."""

    with st.status("Running today's committee…", expanded=True) as status:
        status.write("Fetching current completed Yahoo evidence before calling the model API.")
        try:
            result = run_today_if_requested(True, settings)
        except Exception as exc:  # noqa: BLE001 - render only the safe category below
            category, message = safe_experiment_failure(exc)
            status.update(label=category, state="error")
            st.error(message)
            return None

        artifact_dir = Path(result.artifact_dir).resolve()
        if not _is_today_analysis_dir(artifact_dir):
            status.update(label="Incomplete Today artifact", state="error")
            st.error(
                "The analysis did not produce a completed Today artifact. "
                "The last successful analysis remains displayed."
            )
            return None
        status.update(label="Today's analysis complete", state="complete")
        status.write("A new timestamped Today analysis artifact was saved.")
        return artifact_dir


def run_requested_forward_test_creation(today_dir: Path) -> Path | None:
    """Freeze a saved Today decision only after the secondary action is clicked."""

    with st.status("Freezing today's saved decision…", expanded=True) as status:
        status.write("This copies saved evidence and allocation only. It does not call Gemini or submit an order.")
        try:
            result = create_forward_test_if_requested(True, today_dir)
        except Exception as exc:  # noqa: BLE001 - display the safe category only
            category, message = safe_forward_failure(exc, action="Forward Test creation")
            status.update(label=category, state="error")
            st.error(message)
            return None
        artifact_dir = Path(result.artifact_dir).resolve()
        if not _is_forward_child(artifact_dir) or not _is_forward_test_dir(artifact_dir):
            status.update(label="Incomplete Forward Test artifact", state="error")
            st.error("The saved Today decision was not converted into a complete Forward Test.")
            return None
        if result.reused:
            status.update(label="Loaded existing Forward Test", state="complete")
            status.write("This Today decision already has a frozen Forward Test.")
        else:
            status.update(label="Forward Test created", state="complete")
            status.write("A separate frozen Forward Test artifact was saved. No order was submitted.")
        return artifact_dir


def run_requested_forward_refresh(forward_dir: Path) -> Path | None:
    """Fetch later completed Yahoo bars only after an explicit Forward Test refresh."""

    with st.status("Refreshing observed outcomes…", expanded=True) as status:
        status.write("Fetching completed Yahoo bars only. The AI decision will not be rerun or changed.")
        try:
            result = refresh_forward_test_if_requested(True, forward_dir)
        except Exception as exc:  # noqa: BLE001 - display the safe category only
            category, message = safe_forward_failure(exc, action="outcome refresh")
            status.update(label=category, state="error")
            st.error(message)
            return None
        artifact_dir = Path(result.artifact_dir).resolve()
        if not _is_forward_child(artifact_dir) or not _is_forward_test_dir(artifact_dir):
            status.update(label="Incomplete Forward Test artifact", state="error")
            st.error("The Forward Test outcome could not be saved safely.")
            return None
        status.update(label="Observed outcomes refreshed", state="complete")
        status.write("Only later market outcomes were refreshed; the original allocation remains frozen.")
        return artifact_dir


def _displayed_run_dir() -> Path:
    """Prefer an explicitly selected completed run, then fall back to the latest."""

    selected = st.session_state.get("selected_run_dir")
    if selected:
        candidate = Path(str(selected)).resolve()
        if (candidate / "run_summary.json").is_file():
            return candidate
    return _safe_run_dir()


def _displayed_today_dir() -> Path:
    """Prefer a selected Today artifact without falling back to historical runs."""

    selected = st.session_state.get("selected_today_dir")
    if selected:
        candidate = Path(str(selected)).resolve()
        if _is_today_analysis_dir(candidate):
            return candidate
    return _safe_today_dir()


def _displayed_forward_dir() -> Path:
    """Prefer a selected Forward Test without falling back to Today or historical artifacts."""

    selected = st.session_state.get("selected_forward_test_dir")
    if selected:
        candidate = Path(str(selected)).resolve()
        if _is_forward_child(candidate) and _is_forward_test_dir(candidate):
            return candidate
    return _safe_forward_dir()


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def format_date(value: Any) -> str:
    """Format a saved ISO date without tying the artifact format to the UI."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(timestamp):
        return "—"
    return f"{timestamp.strftime('%B')} {timestamp.day}, {timestamp.year}"


def format_timestamp(value: Any) -> str:
    """Present a saved analysis/evidence timestamp without changing its artifact value."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(timestamp):
        return "—"
    zone = timestamp.strftime("%Z")
    suffix = f" {zone}" if zone else ""
    return f"{timestamp.strftime('%B')} {timestamp.day}, {timestamp.year} at {timestamp:%H:%M}{suffix}"


def format_saved_run_label(kind: str, artifact_id: str) -> str:
    """Turn a timestamped artifact directory name into a human-readable saved-run label."""

    match = re.fullmatch(
        r"(?P<date>\d{8})T(?P<time>\d{6})(?:\d+)?Z",
        artifact_id,
    )
    if match is None:
        return f"Saved {kind}"
    date = match.group("date")
    time = match.group("time")
    created_at = (
        f"{date[:4]}-{date[4:6]}-{date[6:]}T"
        f"{time[:2]}:{time[2:4]}:{time[4:]}+00:00"
    )
    return f"{kind} · saved {format_timestamp(created_at)}"


def decision_by_role(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        role = decision.get("role")
        if role in ROLE_ORDER:
            by_role[role] = decision
    return by_role


def find_lookback_window(
    summary: dict[str, Any], decisions_by_role: dict[str, dict[str, Any]]
) -> str:
    """Prefer the saved experiment configuration, with a legacy-artifact fallback."""

    experiment = summary.get("experiment") or {}
    try:
        configured_lookback = int(experiment["lookback_days"])
    except (KeyError, TypeError, ValueError):
        configured_lookback = 0
    if configured_lookback > 0:
        return f"{configured_lookback} daily bars"

    researcher = decisions_by_role.get("Researcher", {})
    saved_text = _saved_decision_text(researcher, "summary", "text")
    match = re.search(
        r"(\d+)(?:\s*-\s*|\s+)(?:completed\s+)?(?:daily\s+)?bars",
        saved_text,
        re.IGNORECASE,
    )
    return f"{match.group(1)} daily bars" if match else "Fixed by the backtest"


def _plain_text_from_markdown(markdown: str) -> str:
    """Make a compact card preview without leaking Markdown punctuation."""

    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or TABLE_DIVIDER.fullmatch(line) or re.fullmatch(r"[-*_]{3,}", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = line.strip("|")
        if "|" in line:
            line = " — ".join(cell.strip() for cell in line.split("|") if cell.strip())
        line = line.replace("**", "").replace("__", "").replace(chr(96), "").replace("~~", "")
        line = re.sub(r"(?<!\w)[*_~]+|[*_~]+(?!\w)", "", line)
        lines.append(re.sub(r"\s+", " ", line).strip())
    return " ".join(line for line in lines if line)


def concise_summary(markdown: Any, limit: int = 360) -> str:
    """Derive a short, readable preview of no more than three saved-response sentences."""

    saved_text = str(markdown or "").strip()
    if not saved_text:
        return "No saved summary is available for this role."
    result = RESULT_MARKER.search(saved_text)
    plain_text = _plain_text_from_markdown(saved_text[result.end() :] if result else saved_text)
    sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", plain_text))
    if len(sentence_ends) > 3:
        plain_text = plain_text[: sentence_ends[2].end()].strip()
    if len(plain_text) <= limit:
        return plain_text or "No saved summary is available for this role."
    sentence_end = max(
        plain_text.rfind(". ", 0, limit),
        plain_text.rfind("! ", 0, limit),
        plain_text.rfind("? ", 0, limit),
    )
    if sentence_end > limit // 2:
        return plain_text[: sentence_end + 1]
    return f"{plain_text[:limit].rstrip()}…"


def _saved_decision_text(decision: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = decision.get(field)
        if value:
            return str(value).strip()
    return ""


def build_agent_cards(decisions: list[dict[str, Any]]) -> tuple[AgentCard, ...]:
    """Adapt only safe saved decision fields into card-ready presentation data."""

    decisions_by_role = decision_by_role(decisions)
    cards: list[AgentCard] = []
    for role in ROLE_ORDER:
        decision = decisions_by_role.get(role, {})
        summary = _saved_decision_text(decision, "summary", "text")
        details = _saved_decision_text(decision, "text", "summary")
        cards.append(
            AgentCard(
                role=role,
                description=ROLE_DESCRIPTIONS[role],
                preview=concise_summary(summary),
                details=details,
            )
        )
    return tuple(cards)


def _as_boolean(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_risk_rows(risk: pd.DataFrame) -> tuple[RiskGateRow, ...]:
    """Adapt artifact rows once so the renderer never depends on CSV field names."""

    rows: list[RiskGateRow] = []
    for _, row in risk.iterrows():
        raw_reason = row.get("reason", "")
        reason = format_risk_reason(raw_reason)
        rows.append(
            RiskGateRow(
                symbol=str(row["symbol"]),
                proposed_weight=float(row["proposed_weight"]),
                executed_weight=float(row["executed_weight"]),
                reason=reason or "Approved",
                overridden=_as_boolean(row["overridden"]),
            )
        )
    return tuple(rows)


def format_risk_reason(raw_reason: Any) -> str:
    """Replace stored machine labels with concise, human-readable UI copy."""

    if pd.isna(raw_reason):
        return "Approved"
    raw_text = str(raw_reason).strip()
    normalized = re.sub(r"[\s-]+", "_", raw_text.lower())
    position_limit = re.fullmatch(
        r"max_position_(?P<limit>\d+(?:\.\d+)?)pct",
        normalized,
    )
    if position_limit is not None:
        limit = position_limit.group("limit").rstrip("0").rstrip(".")
        return f"Capped by {limit}% position limit"
    if normalized == "residual_after_risk_gate":
        return "Unallocated capital"
    if normalized == "approved":
        return "Approved"
    return raw_text.replace("_", " ") or "Approved"


def build_equity_chart(equity: pd.DataFrame) -> pd.DataFrame:
    return equity.set_index("date")[["portfolio_equity", "benchmark_equity"]].rename(
        columns={"portfolio_equity": "AI portfolio", "benchmark_equity": "SPY benchmark"}
    )


def build_trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "side", "quantity", "fill price", "notional"])
    return trades[["symbol", "side", "quantity", "price", "notional"]].rename(
        columns={"price": "fill price"}
    )


def build_presenter_data(run_dir: Path, run: dict[str, Any]) -> RunView:
    """Isolate saved-artifact fields from every Streamlit rendering component."""

    summary = run["summary"]
    metrics = run["metrics"]
    decisions_by_role = decision_by_role(run["decisions"])
    risk_limits = summary.get("risk_limits", {})
    return RunView(
        run_label=format_saved_run_label("Historical backtest", run_dir.name),
        artifact_id=run_dir.name,
        data_source=str(summary.get("data_source", "Saved artifacts only")),
        lookback_window=find_lookback_window(summary, decisions_by_role),
        max_position_size=format_percent(float(risk_limits.get("max_position_weight", 0.25))),
        max_gross_exposure=format_percent(float(risk_limits.get("max_gross_exposure", 1.0))),
        period=f"{format_date(summary.get('start'))} → {format_date(summary.get('end'))}",
        total_return=float(metrics["total_return"]),
        benchmark_return=float(metrics["benchmark_total_return"]),
        max_drawdown=float(metrics["max_drawdown"]),
        sharpe_ratio=float(metrics["sharpe_ratio"]),
        equity_chart=build_equity_chart(run["equity"]),
        agent_cards=build_agent_cards(run["decisions"]),
        risk_rows=build_risk_rows(run["risk"]),
        trades_table=build_trades_table(run["trades"]),
    )


def build_today_presenter_data(analysis_dir: Path, analysis: dict[str, Any]) -> TodayView:
    """Adapt a Today artifact without depending on any historical metric schema."""

    summary = analysis["summary"]
    evidence = analysis["evidence"]
    risk_limits = summary.get("risk_limits", {})
    risk_rows = build_risk_rows(analysis["risk"])
    cash = next((row.executed_weight for row in risk_rows if row.symbol == "CASH"), 0.0)
    try:
        lookback_days = int(evidence.get("lookback_days", summary["lookback_days"]))
    except (KeyError, TypeError, ValueError):
        lookback_days = 0
    return TodayView(
        run_label=format_saved_run_label("Today analysis", analysis_dir.name),
        artifact_id=analysis_dir.name,
        data_source=str(summary.get("data_source", "YahooData")),
        analysis_timestamp=format_timestamp(summary.get("analysis_timestamp")),
        evidence_as_of=format_timestamp(summary.get("evidence_as_of")),
        lookback_window=f"{lookback_days} completed daily bars" if lookback_days else "Saved evidence",
        max_position_size=format_percent(float(risk_limits.get("max_position_weight", 0.25))),
        max_gross_exposure=format_percent(float(risk_limits.get("max_gross_exposure", 1.0))),
        cash_residual=cash,
        agent_cards=build_agent_cards(analysis["decisions"]),
        risk_rows=risk_rows,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def build_forward_presenter_data(forward_dir: Path, forward_test: dict[str, Any]) -> ForwardTestView:
    """Adapt only a Forward artifact; never reload or recalculate its source Today decision."""

    summary = forward_test["summary"]
    frozen = forward_test["frozen"]
    performance = forward_test["performance"]
    risk_limits = frozen.get("risk_limits", {})
    risk_rows = build_risk_rows(pd.DataFrame(frozen.get("risk_decisions", [])))
    allocation = frozen.get("allowed_allocation", {})
    current = performance.get("current")
    if not isinstance(current, dict):
        current = {}
    horizon_values = performance.get("horizons", {})
    if not isinstance(horizon_values, dict):
        horizon_values = {}
    horizons: list[ForwardHorizon] = []
    for trading_day in (1, 5, 20):
        record = horizon_values.get(str(trading_day), {})
        if not isinstance(record, dict):
            record = {}
        horizons.append(
            ForwardHorizon(
                trading_day=trading_day,
                status=str(record.get("status", "pending")),
                date=str(record["date"]) if record.get("date") else None,
                portfolio_return=_optional_float(record.get("portfolio_cumulative_return")),
                benchmark_return=_optional_float(record.get("spy_cumulative_return")),
                relative_return=_optional_float(record.get("relative_return")),
            )
        )
    try:
        observed_days = int(performance.get("observed_trading_days", 0))
    except (TypeError, ValueError):
        observed_days = 0
    try:
        lookback_days = int(frozen.get("lookback_days", summary.get("lookback_days", 0)))
    except (TypeError, ValueError):
        lookback_days = 0
    return ForwardTestView(
        run_label=format_saved_run_label("Forward Test", forward_dir.name),
        artifact_id=forward_dir.name,
        data_source=str(frozen.get("data_source", summary.get("data_source", "YahooData"))),
        original_analysis_timestamp=format_timestamp(frozen.get("original_analysis_timestamp")),
        evidence_as_of=format_timestamp(frozen.get("evidence_as_of")),
        lookback_window=f"{lookback_days} completed daily bars" if lookback_days else "Frozen evidence",
        max_position_size=format_percent(float(risk_limits.get("max_position_weight", 0.25))),
        max_gross_exposure=format_percent(float(risk_limits.get("max_gross_exposure", 1.0))),
        cash_residual=float(allocation.get("CASH", frozen.get("cash_weight", 0.0))),
        risk_rows=risk_rows,
        observed_trading_days=max(0, observed_days),
        current_portfolio_return=_optional_float(current.get("portfolio_cumulative_return")),
        current_benchmark_return=_optional_float(current.get("spy_cumulative_return")),
        current_relative_return=_optional_float(current.get("relative_return")),
        current_observation_date=str(current["date"]) if current.get("date") else None,
        horizons=tuple(horizons),
    )


def render_lab_styles() -> None:
    """Apply restrained static styling without ever interpolating artifact content."""

    st.html(
        """
        <style>
          .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
          }
          .st-key-lab-current-experiment,
          .st-key-lab-today-context,
          .st-key-lab-forward-context {
            background: #111924;
            border: 1px solid #2a394b;
          }
          .st-key-lab-portfolio-manager {
            background: #121c2a;
            border: 1px solid #4a6b96;
            border-left: 3px solid #83aef5;
          }
          .st-key-lab-risk-gate {
            background: #17191d;
            border: 1px solid #9d7729;
            border-left: 3px solid #d6a744;
          }
          .st-key-lab-analysis-only {
            background: #18202a;
            border: 1px solid #61758d;
            border-left: 3px solid #b9c8dc;
          }
          .st-key-lab-risk-gate [data-testid="stProgressBar"] > div > div {
            background-color: #83aef5;
          }
        </style>
        """
    )


def render_historical_context(view: RunView) -> None:
    """Render saved-run context before any backtest outcomes."""

    st.markdown("### Current / displayed experiment")
    with st.container(border=True, key="lab-current-experiment"):
        st.caption("SAVED HISTORICAL BACKTEST · READ-ONLY ARTIFACT")
        st.markdown(f"**{view.run_label}**")
        period, evidence, policy = st.columns((1.7, 1.1, 1.2))
        period.markdown("**Backtest period**")
        period.write(view.period)
        evidence.markdown("**Evidence window**")
        evidence.write(view.lookback_window)
        policy.markdown("**Max position limit**")
        policy.write(view.max_position_size)
        st.caption(
            f"Source: {view.data_source} · Gross-exposure limit: {view.max_gross_exposure} · "
            "Loading this page does not call Gemini."
        )
        with st.expander("Saved artifact details", expanded=False):
            st.caption(f"Raw artifact ID: {view.artifact_id}")


def render_today_context(view: TodayView) -> None:
    """Render current-analysis provenance without implying a future performance result."""

    with st.container(border=True, key="lab-analysis-only"):
        st.badge("ANALYSIS ONLY", color="orange")
        st.markdown("### Analysis only — no order submitted")
        st.write(
            "This is a recorded committee observation using currently available evidence. "
            "It is not a trade, forecast, or performance result."
        )

    st.markdown("### Current / displayed analysis")
    with st.container(border=True, key="lab-today-context"):
        st.caption("SAVED TODAY ANALYSIS · READ-ONLY ARTIFACT")
        st.markdown(f"**{view.run_label}**")
        analysis_run, market_data = st.columns(2)
        analysis_run.markdown("**Committee analysis ran**")
        analysis_run.write(view.analysis_timestamp)
        market_data.markdown("**Market data available through**")
        market_data.write(view.evidence_as_of)
        st.caption(
            "The committee can run after the latest completed daily market bar, so these timestamps "
            "can differ."
        )
        evidence, policy, cash = st.columns((1.25, 1.15, 1.0))
        evidence.markdown("**Evidence window**")
        evidence.write(view.lookback_window)
        policy.markdown("**Max position limit**")
        policy.write(view.max_position_size)
        cash.markdown("**Unallocated capital**")
        cash.write(format_percent(view.cash_residual))
        st.caption(
            f"Source: {view.data_source} · This saved analysis does not submit an order."
        )
        with st.expander("Saved artifact details", expanded=False):
            st.caption(f"Raw artifact ID: {view.artifact_id}")


def render_metric_summary(view: RunView) -> None:
    """Render the four saved historical outcomes in a compact native layout."""

    st.markdown("### Backtest outcomes")
    metrics_columns = st.columns(4)
    metrics_columns[0].metric(
        "Portfolio return", format_percent(view.total_return), border=True
    )
    metrics_columns[1].metric("SPY return", format_percent(view.benchmark_return), border=True)
    metrics_columns[2].metric("Max drawdown", format_percent(view.max_drawdown), border=True)
    metrics_columns[3].metric("Sharpe", f"{view.sharpe_ratio:.2f}", border=True)


def render_equity_curve(equity_chart: pd.DataFrame) -> None:
    """Render the saved portfolio-versus-benchmark curve only for Historical Lab."""

    st.markdown("### Equity curve versus SPY")
    st.caption("Saved simulated portfolio equity compared with the saved SPY benchmark curve.")
    st.line_chart(equity_chart, width="stretch", height=340)


def render_agent_card(
    card: AgentCard,
    *,
    key: str,
    portfolio_manager: bool = False,
) -> None:
    """Render one concise, artifact-backed role card with optional saved Markdown detail."""

    with st.container(border=True, key=key):
        if portfolio_manager:
            st.badge("STRUCTURED PROPOSAL ONLY", color="blue")
        st.markdown(f"#### {card.role}")
        st.caption(card.description)
        st.write(card.preview)
        if portfolio_manager:
            st.info("This role proposes target weights only. It cannot submit an order.")
        if card.details:
            with st.expander("View saved detail", expanded=False):
                st.markdown(card.details, unsafe_allow_html=False)


def render_committee_workflow(cards: tuple[AgentCard, ...]) -> None:
    """Make the four saved roles readable as a small decision workflow."""

    cards_by_role = {card.role: card for card in cards}
    researcher = cards_by_role["Researcher"]
    bull = cards_by_role["Bull Analyst"]
    bear = cards_by_role["Bear Analyst"]
    portfolio_manager = cards_by_role["Portfolio Manager"]

    st.markdown("## How the committee decided")
    st.caption(
        "Saved summaries are shown first. Detail is collapsed; raw traces and hidden reasoning are not displayed."
    )
    st.markdown("#### Researcher → Bull / Bear → Portfolio Manager")
    render_agent_card(researcher, key="lab-researcher")
    bull_column, bear_column = st.columns(2)
    with bull_column:
        render_agent_card(bull, key="lab-bull")
    with bear_column:
        render_agent_card(bear, key="lab-bear")
    render_agent_card(
        portfolio_manager,
        key="lab-portfolio-manager",
        portfolio_manager=True,
    )


def _weight_progress_value(weight: float) -> int:
    """Keep native progress bars bounded even when inspecting a damaged local artifact."""

    if pd.isna(weight):
        return 0
    return max(0, min(100, round(weight * 100)))


def render_risk_gate(
    risk_rows: tuple[RiskGateRow, ...],
    *,
    max_position_size: str,
    max_gross_exposure: str,
    allowed_label: str,
    proposal_label: str = "AI proposal",
) -> None:
    """Render policy approval as the primary, visibly separate system boundary."""

    asset_overrides = tuple(
        row for row in risk_rows if row.overridden and row.symbol != "CASH"
    )
    all_overrides = tuple(row for row in risk_rows if row.overridden)

    st.markdown("## Deterministic Risk Gate")
    st.caption("The only layer allowed to turn a model proposal into an allocation.")
    with st.container(border=True, key="lab-risk-gate"):
        st.markdown(f"### {proposal_label} → policy → {allowed_label}")
        st.caption(
            f"Policy: {max_position_size} maximum per position · "
            f"{max_gross_exposure} maximum gross exposure"
        )

        if asset_overrides:
            focus = max(
                asset_overrides,
                key=lambda row: abs(row.proposed_weight - row.executed_weight),
            )
            st.warning(
                f"Policy override: {focus.symbol} proposed "
                f"{format_percent(focus.proposed_weight)} → {allowed_label.lower()} "
                f"{format_percent(focus.executed_weight)}."
            )
        elif all_overrides:
            st.warning("Policy left unallocated capital after applying the approved allocations.")
        else:
            st.success("No asset allocation required a policy override in this saved result.")

        for row in risk_rows:
            symbol, proposal, allowed, decision = st.columns((0.8, 1.45, 1.45, 1.0))
            symbol.markdown(f"**{row.symbol}**")
            if row.symbol == "CASH":
                symbol.caption("Unallocated capital")
            proposal.caption(proposal_label)
            proposal.progress(
                _weight_progress_value(row.proposed_weight),
                text=format_percent(row.proposed_weight),
            )
            allowed.caption(allowed_label)
            allowed.progress(
                _weight_progress_value(row.executed_weight),
                text=format_percent(row.executed_weight),
            )
            if row.symbol == "CASH":
                decision.badge("UNALLOCATED CAPITAL", color="gray")
                decision.caption(row.reason)
            elif row.overridden:
                decision.badge("OVERRIDDEN", color="orange")
                decision.caption(row.reason)
            else:
                decision.badge("APPROVED", color="green")
                decision.caption(row.reason)


def render_trades(trades_table: pd.DataFrame) -> None:
    """Keep simulated execution available but visually secondary to the Risk Gate."""

    with st.expander("Simulated execution details", expanded=False):
        if trades_table.empty:
            st.info("No simulated fills were saved for this run.")
            return
        st.dataframe(
            trades_table,
            width="stretch",
            hide_index=True,
            column_config={
                "quantity": st.column_config.NumberColumn("quantity", format="%.0f"),
                "fill price": st.column_config.NumberColumn("fill price", format="$%.2f"),
                "notional": st.column_config.NumberColumn("notional", format="$%.2f"),
            },
        )


def render_system_flow() -> None:
    """Keep the full historical system boundary available without competing with the Risk Gate."""

    with st.expander("System boundary", expanded=False):
        st.markdown(
            "Market evidence → Researcher → Bull/Bear → Portfolio Manager proposal → "
            "deterministic Risk Gate → simulated broker → backtest result"
        )
        st.write(
            "Agents interpret evidence and produce a proposal. Deterministic Python enforces "
            "policy before the simulated broker can execute a saved backtest allocation."
        )


def render_today_system_flow() -> None:
    """Keep the analysis-only system boundary available without implying a broker action."""

    with st.expander("System boundary", expanded=False):
        st.markdown(
            "Current evidence → Researcher → Bull/Bear → Portfolio Manager proposal → "
            "deterministic Risk Gate"
        )
        st.write(
            "This records a current-data proposal for later learning. No broker or simulated order "
            "is involved, and no future outcome is known."
        )


def render_start_forward_test_action(analysis_dir: Path) -> None:
    """Offer an explicit, no-model way to freeze the displayed Today decision."""

    st.markdown("### Forward Test")
    st.caption(
        "Freeze this saved decision for later observation. This does not call Gemini, rerun the "
        "committee, submit an order, or change this Today analysis."
    )
    requested = st.button(
        "Start Forward Test from this decision",
        type="secondary",
        key=f"start-forward-test-{analysis_dir.name}",
    )
    if not requested:
        return
    forward_dir = run_requested_forward_test_creation(analysis_dir)
    if forward_dir is None:
        return
    st.session_state["selected_forward_test_dir"] = str(forward_dir)
    # The mode widget is initialized on the next rerun, before it is constructed in main().
    st.session_state["requested_lab_mode"] = FORWARD_TEST_MODE
    st.rerun()


def render_forward_context(view: ForwardTestView) -> None:
    """Make the immutable decision/evidence boundary obvious before later outcomes."""

    st.markdown("## Original decision — frozen")
    with st.container(border=True, key="lab-forward-context"):
        st.caption("SAVED FORWARD TEST · FROZEN TODAY DECISION")
        st.markdown(f"**{view.run_label}**")
        analysis_run, market_data = st.columns(2)
        analysis_run.markdown("**Original analysis was recorded**")
        analysis_run.write(view.original_analysis_timestamp)
        market_data.markdown("**Market data available through**")
        market_data.write(view.evidence_as_of)
        st.caption(
            "The analysis can be recorded after its latest completed market bar. Only bars that became "
            "available after this market-data cutoff can affect this Forward Test."
        )
        evidence, policy, cash = st.columns((1.25, 1.15, 1.0))
        evidence.markdown("**Frozen evidence window**")
        evidence.write(view.lookback_window)
        policy.markdown("**Frozen position limit**")
        policy.write(view.max_position_size)
        cash.markdown("**Frozen cash**")
        cash.write(format_percent(view.cash_residual))
        st.info("The AI decision is not being updated. Only subsequent market outcomes are changing.")
        st.caption(
            f"Source: {view.data_source} · Research / educational use · No live-money trading · "
            "No order submitted · Not investment advice."
        )
        with st.expander("Saved artifact details", expanded=False):
            st.caption(f"Raw artifact ID: {view.artifact_id}")


def render_forward_results(view: ForwardTestView) -> None:
    """Render saved observed returns only; no outcome appears until later bars exist."""

    st.markdown("## Observed forward result versus SPY")
    st.caption("Buy-and-hold paper observation of the frozen allowed allocation. Cash earns zero return.")
    if (
        view.observed_trading_days == 0
        or view.current_portfolio_return is None
        or view.current_benchmark_return is None
        or view.current_relative_return is None
    ):
        st.info("Waiting for future completed market data.")
        return
    portfolio, benchmark, relative, days = st.columns(4)
    portfolio.metric("Portfolio cumulative return", format_percent(view.current_portfolio_return), border=True)
    benchmark.metric("SPY cumulative return", format_percent(view.current_benchmark_return), border=True)
    relative.metric("Relative return vs SPY", format_percent(view.current_relative_return), border=True)
    days.metric("Forward trading days", str(view.observed_trading_days), border=True)
    if view.current_observation_date:
        st.caption(f"Latest completed forward observation: {format_date(view.current_observation_date)}.")


def render_forward_horizons(view: ForwardTestView) -> None:
    """Show only observed milestone outcomes and label unavailable horizons as pending."""

    st.markdown("### Milestone observations")
    columns = st.columns(len(view.horizons))
    for column, horizon in zip(columns, view.horizons, strict=True):
        with column:
            if horizon.status == "observed" and horizon.portfolio_return is not None:
                st.metric(
                    f"{horizon.trading_day}-day portfolio return",
                    format_percent(horizon.portfolio_return),
                    border=True,
                )
                if horizon.date:
                    st.caption(f"Observed through {format_date(horizon.date)}")
                if horizon.benchmark_return is not None and horizon.relative_return is not None:
                    st.caption(
                        f"SPY {format_percent(horizon.benchmark_return)} · "
                        f"Relative {format_percent(horizon.relative_return)}"
                    )
            else:
                st.metric(f"{horizon.trading_day}-day observation", "Pending", border=True)
                st.caption(
                    f"Awaiting {horizon.trading_day} completed forward trading days; "
                    "no outcome is fabricated."
                )


def render_forward_refresh_action(forward_dir: Path) -> None:
    """Keep Yahoo refresh explicit so viewing a Forward Test is read-only and offline."""

    st.caption(
        "Refresh is optional and fetches only completed Yahoo market data. It never reruns the AI "
        "committee or changes the frozen allocation."
    )
    requested = st.button(
        "Refresh observed market data",
        type="secondary",
        key=f"refresh-forward-test-{forward_dir.name}",
    )
    if not requested:
        return
    refreshed_dir = run_requested_forward_refresh(forward_dir)
    if refreshed_dir is None:
        return
    st.session_state["selected_forward_test_dir"] = str(refreshed_dir)
    st.rerun()


def render_historical_configuration() -> None:
    """Put explicit model-backed controls apart from the saved result they cannot alter."""

    with st.expander("Configure new experiment", expanded=False):
        settings, run_requested = render_experiment_controls()
        if not run_requested:
            return
        completed_run = run_requested_experiment(settings)
        if completed_run is not None:
            st.session_state["selected_run_dir"] = str(completed_run)
            st.rerun()


def render_today_configuration() -> None:
    """Put explicit current-data controls apart from the saved analysis they cannot alter."""

    with st.expander("Configure new Today analysis", expanded=False):
        settings, run_requested = render_today_controls()
        if not run_requested:
            return
        completed_analysis = run_requested_today_analysis(settings)
        if completed_analysis is not None:
            st.session_state["selected_today_dir"] = str(completed_analysis)
            st.rerun()


def render_historical_lab() -> None:
    """Render saved backtest evidence and keep new model runs explicit and separate."""

    try:
        run_dir = _displayed_run_dir()
        run = load_run(str(run_dir))
        view = build_presenter_data(run_dir, run)
    except FileNotFoundError:
        st.info("No completed historical backtest is saved yet. Configure one below to begin.")
        render_historical_configuration()
        return
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        st.error("Could not load the selected completed backtest artifact.")
        render_historical_configuration()
        return

    render_historical_context(view)
    st.divider()
    render_metric_summary(view)
    st.divider()
    render_equity_curve(view.equity_chart)
    st.divider()
    render_committee_workflow(view.agent_cards)
    st.divider()
    render_risk_gate(
        view.risk_rows,
        max_position_size=view.max_position_size,
        max_gross_exposure=view.max_gross_exposure,
        allowed_label="Executed allocation",
    )
    st.divider()
    render_trades(view.trades_table)
    render_system_flow()
    st.divider()
    render_historical_configuration()


def render_today_mode() -> None:
    """Render a separate current-data proposal workflow with no outcome claims."""

    st.caption(
        "What would the AI Investment Committee propose today, given currently available information?"
    )
    try:
        analysis_dir = _displayed_today_dir()
        analysis = load_today_analysis(str(analysis_dir))
        view = build_today_presenter_data(analysis_dir, analysis)
    except FileNotFoundError:
        st.info("No completed Today analysis is saved yet. Configure one below to begin.")
        render_today_configuration()
        return
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        st.error("Could not load the selected completed Today analysis artifact.")
        render_today_configuration()
        return

    render_today_context(view)
    st.divider()
    render_committee_workflow(view.agent_cards)
    st.divider()
    render_risk_gate(
        view.risk_rows,
        max_position_size=view.max_position_size,
        max_gross_exposure=view.max_gross_exposure,
        allowed_label="Allowed allocation",
    )
    st.divider()
    render_start_forward_test_action(analysis_dir)
    st.divider()
    render_today_system_flow()
    st.divider()
    render_today_configuration()


def render_forward_test_mode() -> None:
    """Render a separate saved Forward Test without touching Today, Gemini, or Yahoo on load."""

    st.caption("Observe a frozen Today decision only after later completed market data becomes available.")
    try:
        forward_dir = _displayed_forward_dir()
        forward_test = load_forward_test(
            str(forward_dir),
            _forward_artifact_revision(forward_dir),
        )
        view = build_forward_presenter_data(forward_dir, forward_test)
    except FileNotFoundError:
        st.info("No Forward Test is saved yet. Open a saved Today analysis to freeze its decision.")
        return
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        st.error("Could not load the selected saved Forward Test artifact.")
        return

    render_forward_context(view)
    st.divider()
    render_risk_gate(
        view.risk_rows,
        max_position_size=view.max_position_size,
        max_gross_exposure=view.max_gross_exposure,
        allowed_label="Frozen allowed allocation",
        proposal_label="Frozen AI proposal",
    )
    st.divider()
    render_forward_results(view)
    st.divider()
    render_forward_horizons(view)
    st.divider()
    render_forward_refresh_action(forward_dir)


def main() -> None:
    st.set_page_config(page_title="AI Investment Committee Lab", page_icon="◈", layout="wide")
    render_lab_styles()
    st.title("AI Investment Committee Lab")
    st.caption("LLMs propose. Deterministic policy controls execution.")
    requested_mode = st.session_state.pop("requested_lab_mode", None)
    if requested_mode in {HISTORICAL_LAB, TODAY_MODE, FORWARD_TEST_MODE}:
        st.session_state["lab_mode"] = requested_mode
    mode = st.radio(
        "Mode",
        (HISTORICAL_LAB, TODAY_MODE, FORWARD_TEST_MODE),
        horizontal=True,
        label_visibility="collapsed",
        key="lab_mode",
    )
    if mode == TODAY_MODE:
        render_today_mode()
    elif mode == FORWARD_TEST_MODE:
        render_forward_test_mode()
    else:
        render_historical_lab()


if __name__ == "__main__":
    main()
