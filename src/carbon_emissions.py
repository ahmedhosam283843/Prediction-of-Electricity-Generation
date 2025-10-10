"""
Carbon emissions and scheduling utilities.

This module computes the carbon emissions (kg CO₂) for a workload given:
- Carbon intensity (CI) curves (g/kWh),
- Hardware/server power draw characteristics,
- Scheduling decisions (when to start a job and for how long).

It also provides:
- A fast in-memory CI curve slice mechanism (set_global_ci),
- A fallback CI fetch via codegreen_core (compute_ci),
- Plotting of optimal energy windows against forecasts/actuals,
- Scheduling statistics and summaries across a test set.
"""

from __future__ import annotations
from typing import Sequence, Tuple
from matplotlib import pyplot as plt
from datetime import timedelta
from functools import lru_cache
import warnings
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from codegreen_core.tools.carbon_intensity import compute_ci
from . import config as CFG


# ───────────────────────── internal helpers ────────────────────────────────────
# Global CI curve (optional, speeds up slicing)
_CI_GLOBAL: pd.DataFrame | None = None


def set_global_ci(df: pd.DataFrame):
    """
    Set the global Carbon Intensity (CI) curve for fast in-memory slicing.

    This method is optional but recommended to avoid repeated API calls.
    The DataFrame must contain:
      - startTimeUTC: timezone-aware timestamps in UTC,
      - ci_default:   hourly CI values in g/kWh.

    Args:
        df: CI DataFrame with 'startTimeUTC' and 'ci_default' columns.

    Returns:
        None
    """
    global _CI_GLOBAL
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["startTimeUTC"]):
        df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
    _CI_GLOBAL = df.sort_values("startTimeUTC").reset_index(drop=True)


def _task_kw(server: dict) -> float:
    """
    Compute the constant power draw (kW) of a job on the given server.

    Uses CPU core count, memory size, and Power Usage Effectiveness (PUE).

    Args:
        server: Dictionary describing the server (merged with CFG.DEFAULT_SERVER).

    Returns:
        kW drawn as a float.
    """
    srv = CFG.DEFAULT_SERVER | server
    watts = (
        srv["number_core"] * srv["power_draw_core"] * srv["usage_factor_core"]
        + srv["memory_gb"] * srv["power_draw_mem"]
    )
    # Convert W to kW and apply PUE
    return watts * srv["power_usage_efficiency"] / 1_000.0   # → kW


@lru_cache(maxsize=128)
def _ci_span(country: str, span_start_iso: str, span_end_iso: str) -> pd.DataFrame:
    """
    Fetch CI data for a given country and time span (slow path).

    Called only when a global CI curve was not set via set_global_ci().

    Args:
        country: ISO country code (e.g., 'DE').
        span_start_iso: ISO8601 start timestamp string.
        span_end_iso: ISO8601 end timestamp string.

    Returns:
        DataFrame containing CI values with a 'startTimeUTC' column.
    """
    df = compute_ci(
        country,
        pd.Timestamp(span_start_iso).tz_localize(None),
        pd.Timestamp(span_end_iso).tz_localize(None)
    )
    if not pd.api.types.is_datetime64_any_dtype(df["startTimeUTC"]):
        df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
    return df


def _slice_ci(start_ts: pd.Timestamp, runtime_h: int, country: str) -> pd.DataFrame:
    """
    Slice the CI curve covering [start_ts, start_ts + runtime_h).

    Uses the global CI curve if available, otherwise fetches per-window via _ci_span.

    Args:
        start_ts: Anchor timestamp (UTC; timezone-aware or naive).
        runtime_h: Job duration in hours.
        country: Country code used for fetching CI when global is not set.

    Returns:
        Sliced DataFrame of CI values for the requested window.
    """
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")

    end_ts = start_ts + timedelta(hours=runtime_h)

    if _CI_GLOBAL is not None:
        # Fast in-memory slice
        df = _CI_GLOBAL
        return df[(df["startTimeUTC"] >= start_ts) & (df["startTimeUTC"] < end_ts)]
    else:
        # Per-window fetch (slow path)
        return _ci_span(country, start_ts.isoformat(), end_ts.isoformat())


def _integrate_ci(ci_df: pd.DataFrame, server: dict) -> float:
    """
    Integrate CI × energy to get kg CO₂ for a given window.

    CI is hourly (g/kWh). Multiplying by constant power draw (kW) and summing
    across hours yields total grams of CO₂, converted to kg.

    Args:
        ci_df: Sliced CI DataFrame with column 'ci_default'.
        server: Server configuration dict.

    Returns:
        Total emissions for the window in kg CO₂.
    """
    kw = _task_kw(server)
    # grams per hour = ci[g/kWh] * kW
    g_total = float((ci_df["ci_default"].to_numpy(dtype=float) * kw).sum())
    return g_total / 1_000.0   # → kg


# ───────────────────────── public, cached entry point ──────────────────────────
@lru_cache(maxsize=40_000)
def _true_kg_cached(ts_iso: str, runtime_h: int, srv_sig: tuple) -> float:
    """
    Cached emissions calculator.

    Args:
        ts_iso: ISO8601 timestamp string (anchor time).
        runtime_h: Job duration in hours.
        srv_sig: Hashable server signature (tuple of sorted items).

    Returns:
        kg CO₂ emitted for the job window.

    Notes:
        Falls back to a proxy CI (CFG.AVG_CI_G_PER_KWH) if CI slicing fails.
    """
    srv = dict(srv_sig)
    try:
        ci_df = _slice_ci(pd.Timestamp(ts_iso), runtime_h, srv["country"])
        if ci_df.empty:
            raise RuntimeError("empty CI slice")
        return _integrate_ci(ci_df, srv)
    except Exception as exc:
        warnings.warn(f"compute_ce failed – using 400 g proxy ({exc})",
                      RuntimeWarning, stacklevel=2)
        kw = _task_kw(srv)
        return kw * runtime_h * CFG.AVG_CI_G_PER_KWH / 1_000.0   # kg


def _true_kg(ts: pd.Timestamp, _unused: float,
             runtime_h: int, server: dict | None = None) -> float:
    """
    Convenience wrapper with a legacy signature.

    Args:
        ts: Anchor timestamp.
        _unused: Placeholder to preserve backward-compatible signature (ignored).
        runtime_h: Job duration in hours.
        server: Server configuration. If None, CFG.DEFAULT_SERVER is used.

    Returns:
        kg CO₂ emitted for the job window.
    """
    # allow callers to pass None → use default hardware spec
    if server is None:
        server = CFG.DEFAULT_SERVER

    return _true_kg_cached(
        ts_iso=ts.isoformat(),
        runtime_h=runtime_h,
        srv_sig=tuple(sorted(server.items()))
    )


# ───────────────────────────  API expected by benchmark  ───────────────────────
CG_AVAILABLE = True  # informs benchmark that proper CI is available


def _energy_best_idx(avg_ren: np.ndarray, threshold: float | None) -> int:
    """
    Pick index of the best window by average renewable share.

    If a threshold is provided, choose among windows meeting the threshold.

    Args:
        avg_ren: 1-D array of average renewable shares per window.
        threshold: Optional minimum average renewable share in [0, 1].

    Returns:
        Index of the selected window.
    """
    if threshold is not None:
        cand = np.where(avg_ren >= threshold)[0]
        if cand.size:
            return int(cand[np.argmax(avg_ren[cand])])
    return int(np.argmax(avg_ren))


# ───────────────────────────────────────── carbon_plot_single ──────────
def carbon_plot_single(pred_renew: np.ndarray,
                       start_ts: pd.Timestamp,
                       runtime_h: int = CFG.SCHEDULER_RUNTIME_H,
                       threshold: float = CFG.SCHEDULER_THRESHOLD,
                       country: str | None = None,
                       server: dict = CFG.DEFAULT_SERVER,
                       true_renew: np.ndarray | None = None,
                       model_name: str = "model") -> tuple:
    """
    Plot renewable forecast vs. actual (optional), and highlight the selected window.

    Args:
        pred_renew: Renewable forecast path in [0, 1], length H.
        start_ts: Start timestamp for the path.
        runtime_h: Job duration in hours.
        threshold: Min average renewable percentage to meet (0–1).
        country: Country code override (default from server/CFG).
        server: Server configuration dict.
        true_renew: Optional actual renewable path in [0, 1], length H.
        model_name: Label to include in the plot title.

    Returns:
        (fig, (forecast_window_start, forecast_window_end), (oracle_start, oracle_end))
        where oracle span is (None, None) if true_renew is not provided.
    """
    server = dict(CFG.DEFAULT_SERVER | server)
    server["country"] = country or server.get("country") or CFG.COUNTRY

    H = len(pred_renew)
    t_idx = pd.date_range(start=start_ts, periods=H, freq="h")

    # Use the forecast to find the best *index*
    w = np.ones(runtime_h) / runtime_h
    avg_ren_pred = np.convolve(
        pred_renew, w, mode="valid") if pred_renew is not None else None
    best_pred_i = _energy_best_idx(
        avg_ren_pred, threshold) if avg_ren_pred is not None else 0

    # Emissions now vs at predicted best start
    kg_now = _true_kg(t_idx[0], 0, runtime_h, server)
    kg_pred = _true_kg(t_idx[best_pred_i], 0, runtime_h, server)

    # Oracle (optional, if ground-truth path is provided)
    best_true_i = None
    kg_oracle = None
    oracle_span = (None, None)
    if true_renew is not None and len(true_renew) == H:
        avg_ren_true = np.convolve(true_renew, w, mode="valid")
        best_true_i = _energy_best_idx(avg_ren_true, threshold)
        kg_oracle = _true_kg(t_idx[best_true_i], 0, runtime_h, server)
        oracle_span = (t_idx[best_true_i],
                       t_idx[best_true_i] + timedelta(hours=runtime_h))

    saved_vs_now_pct = 100.0 * (kg_now - kg_pred) / max(kg_now, 1e-9)
    attainment = None
    if kg_oracle is not None and (kg_now - kg_oracle) > 1e-9:
        attainment = 100.0 * (kg_now - kg_pred) / (kg_now - kg_oracle)

    # --------------------------- plot ---------------------------------
    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    if pred_renew is not None:
        ax1.plot(t_idx, pred_renew * 100, lw=2, label="Predicted % renewable")
    if true_renew is not None:
        ax1.plot(t_idx, true_renew * 100, lw=2,
                 ls='--', label="Actual % renewable")
    ax1.set_ylabel("% renewable")
    ax1.set_ylim(0, 100)

    fc_start = t_idx[best_pred_i]
    fc_end = fc_start + timedelta(hours=runtime_h)
    ax1.axvspan(fc_start, fc_end, alpha=.35, color="orange",
                edgecolor="orange", lw=2, label="Energy-forecast")

    if oracle_span[0] is not None:
        ax1.axvspan(*oracle_span, alpha=.20, edgecolor="blue", lw=2,
                    label="Energy-targets")

    lbl = f"Savings vs now: {saved_vs_now_pct:.0f}%"
    if attainment is not None:
        lbl += f"\n(= {attainment:.0f}% of optimal)"
    ax1.text(fc_start + timedelta(hours=runtime_h/2), 50, lbl,
             ha="center", va="center", fontsize=6, weight="bold")

    lines = [f"Start now:             {kg_now:.2f} kg",
             f"Emissions (forecast): {kg_pred:.2f} kg"]
    if kg_oracle is not None:
        lines.append(f"Emissions (target):   {kg_oracle:.2f} kg")
    text_str = "\n".join(lines)
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)

    # Place a text box in bottom right in axes coords
    ax1.text(0.98, 0.02, text_str, transform=ax1.transAxes, fontsize=6,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=props, family="monospace")

    ax1.set_xlabel("Time")
    ax1.set_title(
        f"Optimal energy window ({runtime_h} h — {server['country']})\nModel: {model_name}", fontsize=8)
    ax1.grid(ls="--", alpha=.3)
    ax1.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, (fc_start, fc_end), oracle_span


def scheduling_stats(pred_renew: np.ndarray,
                     start_ts: pd.Timestamp,
                     runtime_h: int = CFG.SCHEDULER_RUNTIME_H,
                     thresholds: Sequence[float] = CFG.SCHEDULER_THRESHOLDS,
                     server: dict = CFG.DEFAULT_SERVER) -> pd.DataFrame:
    """
    Summarize scheduling choices and their emissions compared to starting now.

    Args:
        pred_renew: Forecast renewable path in [0, 1].
        start_ts: Start timestamp of the path.
        runtime_h: Job duration in hours.
        thresholds: Sequence of average renewable thresholds to enforce.
        server: Server configuration dict.

    Returns:
        DataFrame with rows for:
          - Start now
          - Energy-max (no target)
          - Energy-max for each threshold in `thresholds`
    """
    H = len(pred_renew)
    t_idx = pd.date_range(start=start_ts, periods=H, freq="h")
    w = np.ones(runtime_h) / runtime_h
    avg_ren = np.convolve(pred_renew, w, mode="valid")
    # Proxy CI only used for display; _true_kg uses actual CI if available
    proxy_ci = CFG.AVG_CI_G_PER_KWH * (1 - avg_ren)

    kg_now = _true_kg(t_idx[0], proxy_ci[0], runtime_h, server)

    rows = [dict(criterion="Start now",
                 suggested_start=t_idx[0],
                 avg_renew_pct=avg_ren[0] * 100,
                 emissions_kg=kg_now,
                 saved_kg=0.0, saved_pct=0.0)]

    best_any = _energy_best_idx(avg_ren, threshold=None)
    kg_any = _true_kg(t_idx[best_any], proxy_ci[best_any], runtime_h, server)
    rows.append(dict(criterion="Energy-max (no target)",
                     suggested_start=t_idx[best_any],
                     avg_renew_pct=avg_ren[best_any] * 100,
                     emissions_kg=kg_any,
                     saved_kg=kg_now - kg_any,
                     saved_pct=100 * (kg_now - kg_any) / max(kg_now, 1e-9)))

    for thr in thresholds:
        best_thr = _energy_best_idx(avg_ren, threshold=thr)
        kg_thr = _true_kg(
            t_idx[best_thr], proxy_ci[best_thr], runtime_h, server)
        rows.append(dict(criterion=f"Energy-max (avg ≥ {int(thr*100)} %)",
                         suggested_start=t_idx[best_thr],
                         avg_renew_pct=avg_ren[best_thr] * 100,
                         emissions_kg=kg_thr,
                         saved_kg=kg_now - kg_thr,
                         saved_pct=100 * (kg_now - kg_thr) / max(kg_now, 1e-9)))
    return pd.DataFrame(rows)


def scheduler_metrics(pred_renew: np.ndarray,
                      true_renew: np.ndarray | None,
                      start_ts: pd.Timestamp,
                      runtime_h: int = CFG.SCHEDULER_RUNTIME_H,
                      threshold: float = CFG.SCHEDULER_THRESHOLD,
                      country: str | None = None,
                      server: dict = CFG.DEFAULT_SERVER) -> dict:
    """
    Compute detailed scheduler metrics for one test sample.

    Args:
        pred_renew: Forecast renewable path in [0, 1].
        true_renew: Optional actual renewable path in [0, 1].
        start_ts: Start timestamp of the path.
        runtime_h: Job duration in hours.
        threshold: Minimum average renewable share in [0, 1].
        country: Country override (default from server/CFG).
        server: Server configuration dict.

    Returns:
        Dict with keys such as:
          - pred_start_idx, pred_start_ts, oracle_start_idx, oracle_start_ts,
          - avg_ren_pred_win, avg_ren_true_win,
          - kg_now, kg_pred, kg_oracle,
          - saved_kg_pred, saved_pct_pred, saved_kg_oracle, saved_pct_oracle,
          - attainment_pct, regret_kg, thr_met_pred, thr_met_oracle, overlap_h
    """
    server = dict(CFG.DEFAULT_SERVER | server)
    server["country"] = country or server.get("country") or CFG.COUNTRY

    H = len(pred_renew)
    t_idx = pd.date_range(start=start_ts, periods=H, freq="h")
    w = np.ones(runtime_h) / runtime_h

    avg_ren_pred = np.convolve(pred_renew, w, mode="valid")
    proxy_ci_pred = CFG.AVG_CI_G_PER_KWH * (1 - avg_ren_pred)
    best_pred_i = _energy_best_idx(avg_ren_pred, threshold)

    kg_now = _true_kg(
        t_idx[0],           proxy_ci_pred[0],           runtime_h, server)
    kg_pred = _true_kg(t_idx[best_pred_i],
                       proxy_ci_pred[best_pred_i], runtime_h, server)

    saved_kg_pred = kg_now - kg_pred
    saved_pct_pred = 100.0 * saved_kg_pred / max(kg_now, 1e-9)

    best_true_i = kg_oracle = avg_ren_true = None
    saved_kg_oracle = saved_pct_oracle = attainment = overlap_h = regret_kg = None
    if true_renew is not None and len(true_renew) == H:
        avg_ren_true = np.convolve(true_renew, w, mode="valid")
        proxy_ci_true = CFG.AVG_CI_G_PER_KWH * (1 - avg_ren_true)
        best_true_i = _energy_best_idx(avg_ren_true, threshold)
        kg_oracle = _true_kg(t_idx[best_true_i], proxy_ci_true[best_true_i],
                             runtime_h, server)
        saved_kg_oracle = kg_now - kg_oracle
        saved_pct_oracle = 100.0 * saved_kg_oracle / max(kg_now, 1e-9)

        # Attainment: fraction of possible savings achieved by the model.
        # Guard against negative/zero oracle savings.
        if saved_kg_pred < 0:
            attainment = 0.0
        elif saved_kg_oracle <= 1e-9:
            attainment = 100.0
        else:
            attainment = 100.0 * saved_kg_pred / saved_kg_oracle

        regret_kg = kg_pred - kg_oracle
        # Overlap: shared hours of predicted and oracle windows
        overlap_h = max(0, runtime_h - abs(best_pred_i - best_true_i))

    return dict(
        start_ts=start_ts,
        runtime_h=runtime_h,
        threshold=threshold,
        pred_start_idx=best_pred_i,
        pred_start_ts=t_idx[best_pred_i],
        oracle_start_idx=best_true_i,
        oracle_start_ts=t_idx[best_true_i] if best_true_i is not None else None,
        avg_ren_pred_win=float(avg_ren_pred[best_pred_i]),
        avg_ren_true_win=float(
            avg_ren_true[best_true_i]) if avg_ren_true is not None else None,
        kg_now=float(kg_now),
        kg_pred=float(kg_pred),
        kg_oracle=float(kg_oracle) if kg_oracle is not None else None,
        saved_kg_pred=float(saved_kg_pred),
        saved_pct_pred=float(saved_pct_pred),
        saved_kg_oracle=float(
            saved_kg_oracle) if saved_kg_oracle is not None else None,
        saved_pct_oracle=float(
            saved_pct_oracle) if saved_pct_oracle is not None else None,
        attainment_pct=float(attainment) if attainment is not None else None,
        regret_kg=float(regret_kg) if regret_kg is not None else None,
        thr_met_pred=bool(avg_ren_pred[best_pred_i] >= threshold),
        thr_met_oracle=bool(avg_ren_true[best_true_i] >= threshold)
        if avg_ren_true is not None else None,
        overlap_h=overlap_h
    )


def summarize_scheduler(y_pred: np.ndarray,
                        y_true: np.ndarray,
                        Tte: np.ndarray,
                        model_name: str,
                        out_dir: str,
                        runtime_h: int = CFG.SCHEDULER_RUNTIME_H,
                        threshold: float = CFG.SCHEDULER_THRESHOLD,
                        country: str = CFG.COUNTRY) -> tuple[pd.DataFrame, dict]:
    """
    Run scheduler metrics across the test set and save per-sample CSV + summary JSON.

    Args:
        y_pred: Model predictions (N, H).
        y_true: Ground truth paths (N, H).
        Tte:    Timestamps array (N, H) for the future windows.
        model_name: Name of the model (used in filenames).
        out_dir: Output directory to write results.
        runtime_h: Duration (hours) of the scheduled job.
        threshold: Minimum average renewable share in [0, 1].
        country: ISO country code.

    Returns:
        (df, summary) where df is the per-sample metrics DataFrame and summary is a dict.
    """
    rows = []
    for i in tqdm(range(len(y_pred)), desc=f"[{model_name}] scheduler", unit="sample"):
        start_ts = pd.to_datetime(Tte[i, 0])
        m = scheduler_metrics(
            pred_renew=y_pred[i],
            true_renew=y_true[i],
            start_ts=start_ts,
            runtime_h=runtime_h,
            threshold=threshold,
            country=country,
        )
        rows.append(m)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(
        out_dir, f"{model_name}_scheduler_samples.csv"), index=False)

    exact_match = float(
        (df["pred_start_idx"] == df["oracle_start_idx"]).mean() * 100.0)
    total_saved_kg = df["saved_kg_pred"].sum()
    total_kg_now = df["kg_now"].sum()
    overall_saved_pct = 100.0 * total_saved_kg / max(total_kg_now, 1e-9)

    summary = dict(
        model=model_name, N=len(df),
        runtime_h=runtime_h, threshold=threshold,
        avg_saved_kg=float(df["saved_kg_pred"].mean()),
        avg_saved_pct=float(overall_saved_pct),
        avg_attainment_pct=float(df["attainment_pct"].mean()),
        threshold_hit_rate_pred=float(df["thr_met_pred"].mean() * 100.0),
        avg_overlap_h=float(df["overlap_h"].mean()),
        exact_match_rate_pct=exact_match,
        avg_delay_h=float(df["pred_start_idx"].mean()),
        avg_regret_kg=float(df["regret_kg"].mean()),
    )
    with open(os.path.join(out_dir, f"{model_name}_scheduler_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[scheduler] {model_name:>15} | saved {summary['avg_saved_kg']:.2f} kg "
          f"({summary['avg_saved_pct']:.1f} %), attain {summary['avg_attainment_pct']:.1f} %, "
          f"hit-rate {summary['threshold_hit_rate_pred']:.1f} %")
    return df, summary


def table_R2_last_sample(model_name: str,
                         model_preds: dict[str, np.ndarray],
                         y_true_ref: np.ndarray,
                         Tte: np.ndarray,
                         out_dir: str,
                         runtime_h: int = CFG.SCHEDULER_RUNTIME_H,
                         threshold: float = CFG.SCHEDULER_THRESHOLD):
    """
    Save a scheduling comparison table for the last test sample (R2 figure table).

    Args:
        model_name: Which model to use from model_preds.
        model_preds: Dict of model_name → predictions (N, H).
        y_true_ref: Reference ground truth (N, H).
        Tte: Timestamps array (N, H).
        out_dir: Output directory.
        runtime_h: Job runtime in hours.
        threshold: Minimum average renewable share.

    Returns:
        None (writes CSV and prints path).
    """
    i = len(y_true_ref) - 1
    start_ts = pd.to_datetime(Tte[i, 0])
    pred = model_preds[model_name][i]
    true = y_true_ref[i]
    df_sched = scheduling_stats(
        pred_renew=pred, start_ts=start_ts, runtime_h=runtime_h,
        thresholds=(threshold,), server=None,
    )
    m = scheduler_metrics(pred, true, start_ts, runtime_h,
                          threshold, country=CFG.COUNTRY)
    oracle_row = dict(
        criterion="Energy-max (oracle)",
        suggested_start=m["oracle_start_ts"],
        avg_renew_pct=(m["avg_ren_true_win"] *
                       100.0 if m["avg_ren_true_win"] is not None else None),
        emissions_kg=m["kg_oracle"],
        saved_kg=(m["kg_now"] - m["kg_oracle"]
                  ) if m["kg_oracle"] is not None else None,
        saved_pct=(100.0 * (m["kg_now"] - m["kg_oracle"]) /
                   m["kg_now"]) if m["kg_oracle"] is not None else None
    )
    df_sched = pd.concat(
        [df_sched, pd.DataFrame([oracle_row])], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    df_sched_path = os.path.join(out_dir, f"{model_name}_R2_last_sample.csv")
    df_sched.to_csv(df_sched_path, index=False)
    print(
        f"[R2] Saved last-sample scheduling table for {model_name} → {df_sched_path}")


def set_ci_from_test_span(Tte: np.ndarray, horizon_h: int, country: str):
    """
    Prefetch CI for the entire test span and cache it globally.

    Args:
        Tte: Test timestamps array (N, H).
        horizon_h: Forecast horizon in hours (H).
        country: ISO country code.

    Returns:
        None
    """
    span_start = pd.to_datetime(Tte[0, 0])
    span_end = pd.to_datetime(Tte[-1, 0]) + timedelta(hours=horizon_h)
    df = compute_ci(country, span_start, span_end)
    set_global_ci(df)


def run_scheduler_suite(model_preds: dict[str, np.ndarray], y_true_ref: np.ndarray, Tte, country: str, out_dir: str,
                        runtime_h: int = CFG.SCHEDULER_RUNTIME_H, threshold: float = CFG.SCHEDULER_THRESHOLD):
    """
    Run scheduler metrics and per-model tables across the test set.

    Args:
        model_preds: Dict of model_name → (N, H) predictions.
        y_true_ref: Ground truth paths (N, H).
        Tte: Timestamps array (N, H).
        country: ISO country code.
        out_dir: Output directory for summaries.
        runtime_h: Job duration in hours.
        threshold: Minimum average renewable share in [0, 1].

    Returns:
        None (writes JSON/CSV files).
    """
    print(
        f"\n=== Scheduling metrics across test set (R={runtime_h}h, thr={threshold}) ===")
    scheduler_summaries: list[dict] = []
    for mname in model_preds:
        df_sched, summary = summarize_scheduler(
            y_pred=model_preds[mname],
            y_true=y_true_ref,
            Tte=Tte,
            model_name=mname,
            out_dir=out_dir,
            runtime_h=runtime_h,
            threshold=threshold,
            country=country
        )
        scheduler_summaries.append(summary)
        table_R2_last_sample(
            model_name=mname,
            model_preds=model_preds,
            y_true_ref=y_true_ref,
            Tte=Tte,
            out_dir=out_dir,
            runtime_h=runtime_h,
            threshold=threshold
        )
    if scheduler_summaries:
        with open(os.path.join(out_dir, "all_scheduler_summaries.json"), "w") as f:
            json.dump(scheduler_summaries, f, indent=2)


def save_optimal_window_plots(results: list[dict], model_preds: dict[str, np.ndarray], y_true_ref: np.ndarray,
                              Tte, country: str, out_dir: str, runtime_h: int = CFG.SCHEDULER_RUNTIME_H, threshold: float = CFG.SCHEDULER_THRESHOLD):
    """
    Save optimal-window plots for a subset of base and ensemble models.

    Picks top-2 base models and top-2 ensembles by path MAE, then plots:
      - Last test sample,
      - A sample 30 days earlier (if available), else falls back to last.

    Args:
        results: List of per-model result dicts (contains 'model', 'mae_path').
        model_preds: Dict model_name → (N, H) predictions.
        y_true_ref: Ground truth paths (N, H).
        Tte: Test timestamps array (N, H).
        country: ISO country code.
        out_dir: Output directory for plots.
        runtime_h: Job runtime in hours.
        threshold: Minimum average renewable share.

    Returns:
        None (writes PNG files).
    """
    base_sorted = sorted(
        [r for r in results if r.get(
            "mae_path") is not None and not r["model"].startswith("Ensemble_")],
        key=lambda r: r["mae_path"]
    )[:2]
    ens_sorted = sorted(
        [r for r in results if r.get(
            "mae_path") is not None and r["model"].startswith("Ensemble_")],
        key=lambda r: r["mae_path"]
    )[:2]

    start_ts = pd.to_datetime(Tte[-1, 0])
    start_ts_earlier = start_ts - timedelta(days=30)
    true_last = y_true_ref[-1]

    def _save(model_name: str, when_ts: pd.Timestamp, pred: np.ndarray, true: np.ndarray, suffix: str):
        fig, _, _ = carbon_plot_single(
            pred_renew=pred, true_renew=true, start_ts=when_ts,
            runtime_h=runtime_h, threshold=threshold, country=country, model_name=model_name + suffix
        )
        fname = f"{model_name}{'_month_earlier' if suffix else ''}_optimal_window.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300)

    for r in base_sorted + ens_sorted:
        m = r["model"]
        if m not in model_preds:
            continue
        pred_last = model_preds[m][-1]
        _save(m, start_ts, pred_last, true_last, "")
        idx_earlier = np.where(Tte[:, 0] == np.datetime64(start_ts_earlier))[0]
        if len(idx_earlier) > 0:
            pred_earlier = model_preds[m][idx_earlier[0]]
            true_earlier = y_true_ref[idx_earlier[0]]
        else:
            pred_earlier, true_earlier = pred_last, true_last
        _save(m, start_ts_earlier, pred_earlier,
              true_earlier, " (30 days earlier)")
