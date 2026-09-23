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
TODAY_SUMMARY_FILE = "today_summary.json"
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

    run_name: str
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

    run_name: str
    data_source: str
    analysis_timestamp: str
    evidence_as_of: str
    lookback_window: str
    max_position_size: str
    max_gross_exposure: str
    cash_residual: float
    agent_cards: tuple[AgentCard, ...]
    risk_rows: tuple[RiskGateRow, ...]


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


def render_experiment_controls() -> tuple[ExperimentSettings, bool]:
    """Render the only interactive parameters and the explicit run action."""

    st.subheader("New learning experiment")
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

    st.warning(
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

    st.subheader("Today's committee")
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

    st.warning(
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
            status.write(f"Reused saved run: {artifact_dir.name}")
        else:
            status.update(label="Experiment complete", state="complete")
            status.write(f"Saved new run: {artifact_dir.name}")
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
        status.write(f"Saved new analysis: {artifact_dir.name}")
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
    """Derive a short, readable preview from a saved agent response."""

    saved_text = str(markdown or "").strip()
    if not saved_text:
        return "No saved summary is available for this role."
    result = RESULT_MARKER.search(saved_text)
    plain_text = _plain_text_from_markdown(saved_text[result.end() :] if result else saved_text)
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
        reason = "" if pd.isna(raw_reason) else str(raw_reason).replace("_", " ")
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
        run_name=run_dir.name,
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
        run_name=analysis_dir.name,
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


def render_metric_summary(view: RunView) -> None:
    st.subheader("Backtest summary")
    metrics_columns = st.columns(4)
    metrics_columns[0].metric("Portfolio total return", format_percent(view.total_return))
    metrics_columns[1].metric("SPY benchmark return", format_percent(view.benchmark_return))
    metrics_columns[2].metric("Max drawdown", format_percent(view.max_drawdown))
    metrics_columns[3].metric("Sharpe ratio", f"{view.sharpe_ratio:.2f}")
    st.markdown(f"**Backtest period:** {view.period}")


def render_equity_curve(equity_chart: pd.DataFrame) -> None:
    st.subheader("Equity curve vs SPY")
    st.line_chart(equity_chart, width="stretch")


def render_agent_card(column: Any, card: AgentCard) -> None:
    with column.container(border=True):
        st.markdown(f"### {card.role}")
        st.caption(card.description)
        st.write(card.preview)
        if card.details:
            with st.expander("View details", expanded=False):
                st.markdown(card.details, unsafe_allow_html=False)


def render_agent_cards(cards: tuple[AgentCard, ...]) -> None:
    st.subheader("AI committee decisions")
    st.caption("Concise saved summaries first. Raw traces, tool logs, and hidden reasoning are not shown.")
    for left_card, right_card in zip(cards[::2], cards[1::2], strict=True):
        left, right = st.columns(2)
        render_agent_card(left, left_card)
        render_agent_card(right, right_card)


def render_risk_gate(risk_rows: tuple[RiskGateRow, ...]) -> None:
    st.subheader("Risk Gate")
    st.markdown("### AI proposed weights → deterministic risk gate → executed weights")
    overrides = tuple(row for row in risk_rows if row.overridden)
    aapl = next((row for row in risk_rows if row.symbol == "AAPL"), None)
    if aapl is not None and aapl.overridden:
        st.error(
            "Override applied: the Portfolio Manager proposed "
            f"{format_percent(aapl.proposed_weight)} AAPL, but the deterministic risk "
            f"gate limited execution to {format_percent(aapl.executed_weight)}."
        )
    elif overrides:
        st.error(f"{len(overrides)} deterministic risk override(s) were applied.")
    else:
        st.success("No weights required a deterministic risk override in this run.")

    heading = st.columns((1.1, 1.5, 2.4, 1.6))
    heading[0].caption("Symbol")
    heading[1].caption("AI proposed")
    heading[2].caption("Risk Gate")
    heading[3].caption("Executed")
    for row in risk_rows:
        columns = st.columns((1.1, 1.5, 2.4, 1.6))
        columns[0].markdown(f"**{row.symbol}**")
        columns[1].markdown(f"### {format_percent(row.proposed_weight)}")
        if row.overridden:
            columns[2].error(row.reason)
        else:
            columns[2].success("Approved")
        columns[3].markdown(f"### {format_percent(row.executed_weight)}")


def render_trades(trades_table: pd.DataFrame) -> None:
    st.subheader("Simulated trades")
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


def render_sidebar(view: RunView) -> None:
    with st.sidebar:
        st.header("Experiment settings")
        st.caption("Display-only for this completed run")
        st.text_input(
            "Max position size",
            value=view.max_position_size,
            disabled=True,
        )
        st.text_input(
            "Max gross exposure",
            value=view.max_gross_exposure,
            disabled=True,
        )
        st.text_input("Lookback window", value=view.lookback_window, disabled=True)
        st.divider()
        st.caption(f"Run: {view.run_name}")
        st.caption(f"Data: {view.data_source}")
        st.caption("This page never calls Gemini.")


def render_today_sidebar(view: TodayView) -> None:
    with st.sidebar:
        st.header("Today analysis settings")
        st.caption("Saved with this analysis")
        st.text_input("Max position size", value=view.max_position_size, disabled=True)
        st.text_input("Max gross exposure", value=view.max_gross_exposure, disabled=True)
        st.text_input("Lookback window", value=view.lookback_window, disabled=True)
        st.divider()
        st.caption(f"Analysis: {view.run_name}")
        st.caption(f"Data: {view.data_source}")
        st.caption("Loading this page never calls Gemini.")


def render_system_flow() -> None:
    with st.expander("How this system works"):
        st.markdown(
            "### Market data → Researcher → Bull/Bear → Portfolio Manager proposal → "
            "deterministic Risk Gate → simulated broker → backtest result"
        )
        st.write(
            "The agents interpret saved market evidence and produce a proposal. Deterministic Python "
            "rules then cap that proposal before the simulated broker can execute it."
        )


def render_today_summary(view: TodayView) -> None:
    """Render current evidence metadata without implying any realized performance."""

    st.success("Analysis only — no order was submitted.")
    information, policy, cash = st.columns(3)
    information.markdown(f"**Analysis timestamp:** {view.analysis_timestamp}")
    policy.markdown(f"**Evidence:** {view.lookback_window}")
    cash.metric("Cash residual after Risk Gate", format_percent(view.cash_residual))
    st.caption(f"Latest completed Yahoo daily bar: {view.evidence_as_of}")


def render_today_system_flow() -> None:
    with st.expander("How this system works"):
        st.markdown(
            "### Current evidence → Researcher → Bull/Bear → Portfolio Manager proposal → "
            "deterministic Risk Gate"
        )
        st.write(
            "This records a current-data proposal for later learning. No broker or simulated order "
            "is involved, and the future outcome is not known yet."
        )


def render_historical_lab() -> None:
    """Keep the established backtest-and-saved-run workflow isolated from Today Mode."""

    settings, run_requested = render_experiment_controls()
    if run_requested:
        completed_run = run_requested_experiment(settings)
        if completed_run is not None:
            st.session_state["selected_run_dir"] = str(completed_run)

    try:
        run_dir = _displayed_run_dir()
        run = load_run(str(run_dir))
        view = build_presenter_data(run_dir, run)
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"Could not load a completed backtest run: {exc}")
        st.stop()

    render_sidebar(view)
    render_metric_summary(view)
    st.divider()
    render_equity_curve(view.equity_chart)
    st.divider()
    render_agent_cards(view.agent_cards)
    st.divider()
    render_risk_gate(view.risk_rows)
    st.divider()
    render_trades(view.trades_table)
    render_system_flow()


def render_today_mode() -> None:
    """Render a separate current-data proposal workflow with no performance outcome claims."""

    st.caption(
        "What would the AI Investment Committee propose today, given currently available information?"
    )
    settings, run_requested = render_today_controls()
    if run_requested:
        completed_analysis = run_requested_today_analysis(settings)
        if completed_analysis is not None:
            st.session_state["selected_today_dir"] = str(completed_analysis)

    try:
        analysis_dir = _displayed_today_dir()
        analysis = load_today_analysis(str(analysis_dir))
        view = build_today_presenter_data(analysis_dir, analysis)
    except FileNotFoundError:
        st.info("No completed Today analysis is saved yet. Run Today's Committee to create one.")
        return
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        st.error("Could not load a completed Today analysis artifact.")
        return

    render_today_sidebar(view)
    render_today_summary(view)
    st.divider()
    render_agent_cards(view.agent_cards)
    st.divider()
    render_risk_gate(view.risk_rows)
    render_today_system_flow()


def main() -> None:
    st.set_page_config(page_title="AI Investment Lab", page_icon="📈", layout="wide")
    st.title("AI Investment Lab")
    st.caption("A local learning interface. Loading or refreshing this page never calls Gemini.")
    mode = st.radio("Mode", (HISTORICAL_LAB, TODAY_MODE), horizontal=True)
    if mode == TODAY_MODE:
        render_today_mode()
    else:
        render_historical_lab()


if __name__ == "__main__":
    main()
