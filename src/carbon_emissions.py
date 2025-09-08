from __future__ import annotations
from typing import Sequence  # only for type hints

from matplotlib import pyplot as plt
"""
Fast carbon-emission helper used by benchmark.py.

Key ideas
─────────
1.  Fetch the carbon-intensity (CI) curve of the whole evaluation span **once**
    and keep it in memory (`set_global_ci`).
2.  Every per-sample query then becomes a cheap Pandas slice + dot product
    instead of an extra call to `compute_ci()`.
3.  Fallback to the old behaviour (and, finally, to the 400 g proxy) if the
    global curve was not provided.
"""
from datetime import timedelta
from functools import lru_cache
import warnings
import pandas as pd
import numpy as np

# --- bridge to the framework ---------------------------------------------------
from codegreen_core.tools.carbon_intensity import compute_ci
from .config import COUNTRY as CFG_COUNTRY
# ------------------------------------------------------------------------------

# ───────────────────────────── configuration & defaults ────────────────────────
AVG_CI_G_PER_KWH = 400.0          # proxy used only as last resort

DEFAULT_SERVER = dict(
    country=CFG_COUNTRY,
    number_core=8,
    memory_gb=32,
    power_draw_core=15.8,  # W / core  (Green-Algorithms defaults)
    usage_factor_core=1.0,
    power_draw_mem=0.3725,  # W / GiB
    power_usage_efficiency=1.6,
)

# ───────────────────────── internal helpers ────────────────────────────────────
_CI_GLOBAL: pd.DataFrame | None = None          # will hold the big CI curve


def set_global_ci(df: pd.DataFrame):
    """
    Provide the already-downloaded CI curve to this module.

    df must contain: startTimeUTC  (timezone-aware UTC)   + ci_default
    """
    global _CI_GLOBAL
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["startTimeUTC"]):
        df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
    _CI_GLOBAL = df.sort_values("startTimeUTC").reset_index(drop=True)


def _task_kw(server: dict) -> float:
    """kW drawn by the job (constant over time)."""
    srv = DEFAULT_SERVER | server
    watts = (
        srv["number_core"] * srv["power_draw_core"] * srv["usage_factor_core"]
        + srv["memory_gb"] * srv["power_draw_mem"]
    )
    return watts * srv["power_usage_efficiency"] / 1_000.0   # → kW


@lru_cache(maxsize=128)
def _ci_span(country: str, span_start_iso: str, span_end_iso: str) -> pd.DataFrame:
    """
    Slow path – call the framework if the global curve is not set.
    Only reached if `set_global_ci()` was never called.
    """
    df = compute_ci(
        country,
        pd.Timestamp(span_start_iso),
        pd.Timestamp(span_end_iso)
    )
    if not pd.api.types.is_datetime64_any_dtype(df["startTimeUTC"]):
        df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
    return df


def _slice_ci(start_ts: pd.Timestamp, runtime_h: int, country: str) -> pd.DataFrame:
    """
    Return the slice of the CI curve covering [start_ts, start_ts+runtime_h).
    Uses the global frame if available; otherwise falls back to `_ci_span`.
    """
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")

    end_ts = start_ts + timedelta(hours=runtime_h)

    if _CI_GLOBAL is not None:
        # fast in-memory slice
        df = _CI_GLOBAL
        return df[(df["startTimeUTC"] >= start_ts) & (df["startTimeUTC"] < end_ts)]
    else:
        # per-window fetch (slow)
        return _ci_span(country, start_ts.isoformat(), end_ts.isoformat())


def _integrate_ci(ci_df: pd.DataFrame, server: dict) -> float:
    """
    Integrate CI × energy → kg CO₂ for the given window (ci_df already sliced).
    """
    kw = _task_kw(server)
    # grams per hour = ci[g/kWh] * kW
    g_total = float((ci_df["ci_default"].to_numpy(dtype=float) * kw).sum())
    return g_total / 1_000.0   # → kg


# ───────────────────────── public, cached entry point ──────────────────────────
@lru_cache(maxsize=40_000)
def _true_kg_cached(ts_iso: str, runtime_h: int, srv_sig: tuple) -> float:
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
        return kw * runtime_h * AVG_CI_G_PER_KWH / 1_000.0   # kg


def _true_kg(ts: pd.Timestamp, _unused: float,
             runtime_h: int, server: dict | None = None) -> float:
    """
     Signature kept for benchmark helpers (ts is anchor timestamp).
    """
    # allow callers to pass None → use default hardware spec
    if server is None:
        server = DEFAULT_SERVER

    return _true_kg_cached(
        ts_iso=ts.isoformat(),
        runtime_h=runtime_h,
        srv_sig=tuple(sorted(server.items()))
    )


# ───────────────────────────  API expected by benchmark  ───────────────────────
CG_AVAILABLE = True        # informs benchmark that proper CI is available


def precompute_emissions(*_, **__) -> int:
    """No extra pre-compute needed once `set_global_ci()` is called."""
    return 0


def clear_emissions_cache():
    _true_kg_cached.cache_clear()


def _energy_best_idx(avg_ren: np.ndarray, threshold: float | None) -> int:
    if threshold is not None:
        cand = np.where(avg_ren >= threshold)[0]
        if cand.size:
            return int(cand[np.argmax(avg_ren[cand])])
    return int(np.argmax(avg_ren))


# ───────────────────────────────────────── carbon_plot_single ──────────
def carbon_plot_single(pred_renew: np.ndarray,
                       start_ts: pd.Timestamp,
                       runtime_h: int = 8,
                       threshold: float = .75,
                       country: str | None = None,
                       server: dict = DEFAULT_SERVER,
                       cei0: float = AVG_CI_G_PER_KWH,
                       true_renew: np.ndarray | None = None):

    server = dict(DEFAULT_SERVER | server)
    server["country"] = country or server.get("country") or CFG_COUNTRY

    H = len(pred_renew)
    t_idx = pd.date_range(start=start_ts, periods=H, freq="h")

    k = runtime_h
    w = np.ones(k) / k
    avg_ren_pred = np.convolve(pred_renew, w, mode="valid")
    proxy_ci_pred = cei0 * (1 - avg_ren_pred)

    best_pred_i = _energy_best_idx(avg_ren_pred, threshold)

    kg_now = _true_kg(
        t_idx[0],           proxy_ci_pred[0],           runtime_h, server)
    kg_pred = _true_kg(t_idx[best_pred_i],
                       proxy_ci_pred[best_pred_i], runtime_h, server)

    # Oracle (optional, if ground-truth path is provided)
    best_true_i = None
    kg_oracle = None
    oracle_span = (None, None)
    if true_renew is not None and len(true_renew) == H:
        avg_ren_true = np.convolve(true_renew, w, mode="valid")
        proxy_ci_true = cei0 * (1 - avg_ren_true)
        best_true_i = _energy_best_idx(avg_ren_true, threshold)
        kg_oracle = _true_kg(t_idx[best_true_i], proxy_ci_true[best_true_i],
                             runtime_h, server)
        oracle_span = (t_idx[best_true_i],
                       t_idx[best_true_i] + timedelta(hours=runtime_h))

    saved_vs_now_pct = 100.0 * (kg_now - kg_pred) / max(kg_now, 1e-9)
    attainment = None
    if kg_oracle is not None and (kg_now - kg_oracle) > 1e-9:
        attainment = 100.0 * (kg_now - kg_pred) / (kg_now - kg_oracle)

    # --------------------------- plot ---------------------------------
    fig = plt.figure(figsize=(10.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(t_idx, pred_renew * 100, lw=2, label="Predicted % renewable")
    if true_renew is not None:
        ax1.plot(t_idx, true_renew * 100, lw=2,
                 ls='--', label="Actual % renewable")
    ax1.set_ylabel("% renewable")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.set_ylabel("kg CO₂ for runtime")
    ax2.set_ylim(0, max(kg_now, kg_pred, kg_oracle or 0) * 1.25)
    ax2.bar(t_idx[0], kg_now, width=.04, label="Start now")

    fc_start = t_idx[best_pred_i]
    fc_end = fc_start + timedelta(hours=runtime_h)
    ax1.axvspan(fc_start, fc_end, alpha=.35, color="orange",
                edgecolor="darkorange", lw=2, label="Energy-max (forecast)")

    if oracle_span[0] is not None:
        ax1.axvspan(*oracle_span, alpha=.20, edgecolor="navy", lw=2,
                    label="Energy-max (target)")

    lbl = f"Savings vs now: {saved_vs_now_pct:.0f}%"
    if attainment is not None:
        lbl += f"\n(= {attainment:.0f}% of optimal)"
    ax1.text(fc_start + timedelta(hours=runtime_h/2), 50, lbl,
             ha="center", va="center", fontsize=10, weight="bold")

    ax_box = fig.add_subplot(gs[1])
    ax_box.axis("off")
    lines = [f"Start now:             {kg_now:.2f} kg",
             f"Energy-max (forecast): {kg_pred:.2f} kg"]
    if kg_oracle is not None:
        lines.append(f"Energy-max (target):   {kg_oracle:.2f} kg")
    ax_box.text(0.21, 0.98, "\n".join(lines),
                ha="left", va="top", family="monospace", fontsize=8)

    ax1.set_xlabel("Time")
    ax1.set_title(
        f"Optimal energy window ({runtime_h} h — {server['country']})")
    ax1.grid(ls="--", alpha=.3)
    ax1.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, (fc_start, fc_end), oracle_span


# ----------------------------------------------------------------------
#  Remaining helpers (scheduling_stats, scheduler_metrics)
#  – unchanged except they now call the cached _true_kg.
# ----------------------------------------------------------------------


def scheduling_stats(pred_renew: np.ndarray,
                     start_ts: pd.Timestamp,
                     runtime_h: int = 8,
                     thresholds: Sequence[float] = (.7, .75, .9),
                     server: dict = DEFAULT_SERVER,
                     cei0: float = AVG_CI_G_PER_KWH):

    H = len(pred_renew)
    t_idx = pd.date_range(start=start_ts, periods=H, freq="h")
    w = np.ones(runtime_h) / runtime_h
    avg_ren = np.convolve(pred_renew, w, mode="valid")
    proxy_ci = cei0 * (1 - avg_ren)

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
                      runtime_h: int = 8,
                      threshold: float = .75,
                      country: str | None = None,
                      server: dict = DEFAULT_SERVER,
                      cei0: float = AVG_CI_G_PER_KWH) -> dict:

    server = dict(DEFAULT_SERVER | server)
    server["country"] = country or server.get("country") or CFG_COUNTRY

    H = len(pred_renew)
    t_idx = pd.date_range(start=start_ts, periods=H, freq="h")
    w = np.ones(runtime_h) / runtime_h

    avg_ren_pred = np.convolve(pred_renew, w, mode="valid")
    proxy_ci_pred = cei0 * (1 - avg_ren_pred)
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
        proxy_ci_true = cei0 * (1 - avg_ren_true)
        best_true_i = _energy_best_idx(avg_ren_true, threshold)
        kg_oracle = _true_kg(t_idx[best_true_i], proxy_ci_true[best_true_i],
                             runtime_h, server)
        saved_kg_oracle = kg_now - kg_oracle
        saved_pct_oracle = 100.0 * saved_kg_oracle / max(kg_now, 1e-9)
        if saved_kg_oracle > 1e-9:
            attainment = 100.0 * saved_kg_pred / saved_kg_oracle
        regret_kg = kg_pred - kg_oracle
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
