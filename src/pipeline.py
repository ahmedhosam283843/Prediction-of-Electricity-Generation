from __future__ import annotations
import os
import json
import numpy as np
from . import config as CFG
from .data_loader import create_modeling_datasets as prepare_data
from .train_eval import run_model_any, compute_metrics_original
from .carbon_emissions import set_ci_from_test_span, run_scheduler_suite, save_optimal_window_plots
from .utils import (
    init_env, make_out_dir, make_loaders, make_loaders_cycle,
    compute_cycle_hour_index, print_markdown_table, save_json,
    append_results, plot_72h_path_last_sample, plot_predictions, save_forecast_plots
)
from .models import (
    CNNLSTMModel, CycleLSTMModel, GRUForecaster, InformerForecaster, LSTMForecaster,
    Seq2SeqForecaster, TCNForecaster, TransformerForecaster, XGBoostForecaster,
)


def _torch_registry(F: int, H: int) -> dict[str, callable]:
    return {
        "LSTM": lambda: LSTMForecaster(input_dim=F, lookback=CFG.LOOKBACK_HOURS, horizon=H, **CFG.MODEL_DEFAULTS["LSTM"]),
        "GRU": lambda: GRUForecaster(input_dim=F, horizon=H, **CFG.MODEL_DEFAULTS["GRU"]),
        "TCN": lambda: TCNForecaster(input_dim=F, horizon=H, **CFG.MODEL_DEFAULTS["TCN"]),
        "CNN-LSTM": lambda: CNNLSTMModel(input_dim=F, output_dim=H, **CFG.MODEL_DEFAULTS["CNN-LSTM"]),
        "Seq2Seq": lambda: Seq2SeqForecaster(in_dim=F, horizon=H, **CFG.MODEL_DEFAULTS["Seq2Seq"]),
        "Informer": lambda: InformerForecaster(input_dim=F, horizon=H, **CFG.MODEL_DEFAULTS["Informer"]),
        "Transformer": lambda: TransformerForecaster(input_dim=F, horizon=H, **CFG.MODEL_DEFAULTS["Transformer"]),
    }


def _model_specs(F: int, H: int) -> list[dict]:
    # Torch family
    all_specs = [
        dict(name=name, type="torch", build=ctor)
        for name, ctor in _torch_registry(F, H).items()
    ]
    # CycleLSTM (needs extra index)
    all_specs.append(dict(
        name="CycleLSTM", type="torch", needs_cycle_idx=True,
        build=lambda: CycleLSTMModel(
            input_size=F, output_size=H, **CFG.MODEL_DEFAULTS["CycleLSTM"]
        )
    ))
    # XGBoost + ARIMA
    all_specs.append(dict(name="XGBoost", type="xgb",
                          build=lambda: XGBoostForecaster))
    all_specs.append(dict(name="ARIMA(1,1,1)", type="arima",
                     **CFG.MODEL_DEFAULTS["ARIMA"]))

    # Filter specs based on the run list from config
    run_list = CFG.MODEL_RUN_LIST
    return [s for s in all_specs if s["name"] in run_list]


def _add_ensembles(results: list[dict], model_preds: dict[str, np.ndarray], y_true_ref: np.ndarray,
                   Tte: np.ndarray, H: int, out_dir: str):
    """
    Build mean ensembles of top-k base models and log metrics/plots.
    """
    eligible = [r for r in results if (
        r.get("mae_path") is not None) and (r["model"] in model_preds)]
    eligible_sorted = sorted(eligible, key=lambda r: r["mae_path"])

    for k in [2, 3, 4, 5]:
        if len(eligible_sorted) < k:
            continue
        top_models = [r["model"] for r in eligible_sorted[:k]]
        preds_list = [model_preds[name] for name in top_models]
        ens_pred = np.mean(np.stack(preds_list, axis=0), axis=0)
        ens_metrics = compute_metrics_original(ens_pred, y_true_ref)
        row = dict(model=f"Ensemble_top{k}", params=f"{k} models",
                   mae_t72=ens_metrics["mae_t72"], rmse_t72=ens_metrics["rmse_t72"],
                   mae_path=ens_metrics["mae_path"], rmse_path=ens_metrics["rmse_path"],
                   soft_dtw=ens_metrics["soft_dtw"],
                   train_time_s=0.0, infer_ms_per_sample=0.0)
        results.append(row)

        # Save metrics JSON
        with open(os.path.join(out_dir, f"Ensemble_top{k}_metrics.json"), "w") as f:
            json.dump(ens_metrics, f, indent=2)

        # Plots
        if CFG.SAVE_PLOTS:
            plot_72h_path_last_sample(
                ens_pred, y_true_ref, Tte, f"Ensemble_top{k}", out_dir)
            plot_predictions(ens_pred, y_true_ref, Tte, horizon=H,
                             model_name=f"Ensemble_top{k}", save_dir=out_dir)

        model_preds[f"Ensemble_top{k}"] = ens_pred


def run_benchmark():
    device = init_env(CFG.SEED)
    COUNTRY = getattr(CFG, "COUNTRY", "DE")

    # Data and loaders
    data = prepare_data(CFG.START_DATE, CFG.END_DATE, CFG.LOOKBACK_HOURS,
                        CFG.HORIZON_HOURS, CFG.VAL_FRAC, CFG.TEST_FRAC)
    loaders = make_loaders(
        data["Xtr"], data["ytr"], data["Xva"], data["yva"], data["Xte"], data["yte"])
    idx_tr = compute_cycle_hour_index(data["Ttr"], CFG.LOOKBACK_HOURS, 24)
    idx_va = compute_cycle_hour_index(data["Tva"], CFG.LOOKBACK_HOURS, 24)
    idx_te = compute_cycle_hour_index(data["Tte"], CFG.LOOKBACK_HOURS, 24)
    cycle_loaders = make_loaders_cycle(
        data["Xtr"], data["ytr"], idx_tr, data["Xva"], data["yva"], idx_va, data["Xte"], data["yte"], idx_te
    )

    F = data["Xtr"].shape[-1]
    H = CFG.HORIZON_HOURS
    out_dir = make_out_dir(COUNTRY)

    results: list[dict] = []
    model_preds: dict[str, np.ndarray] = {}
    y_true_ref: np.ndarray | None = None

    # Unified loop over all model specs
    for spec in _model_specs(F, H):
        metrics, y_pred, y_true = run_model_any(
            spec=spec, data=data, loaders=loaders, cycle_loaders=cycle_loaders,
            device=device, epochs=CFG.EPOCHS, out_dir=out_dir
        )

        # Append, save, plot
        append_results(results, spec["name"], metrics["param_count"], metrics)
        save_json([metrics], out_dir, f"{spec['name']}_metrics.json")
        if CFG.SAVE_PLOTS:
            save_forecast_plots(
                y_pred, y_true, data["Tte"], spec["name"], out_dir, H)

        model_preds[spec["name"]] = y_pred
        if y_true_ref is None:
            y_true_ref = y_true

    # Ensembles
    _add_ensembles(results, model_preds, y_true_ref, data["Tte"], H, out_dir)
    save_json(results, out_dir, "summary_all.json")

    # Print comparison
    headers = ["model", "mae_t72", "rmse_t72", "mae_path", "rmse_path", "soft_dtw", "params",
               "train_time_s", "infer_ms_per_sample", "carbon_kg"]
    print("\nComparison (original units):")
    print_markdown_table(results, headers)

    # Carbon intensity and scheduling
    if CFG.USE_GLOBAL_CI_CURVE:
        set_ci_from_test_span(data["Tte"], CFG.HORIZON_HOURS, COUNTRY)
    run_scheduler_suite(model_preds, y_true_ref, data["Tte"], COUNTRY, out_dir,
                        runtime_h=CFG.SCHEDULER_RUNTIME_H, threshold=CFG.SCHEDULER_THRESHOLD)

    if CFG.SAVE_PLOTS:
        save_optimal_window_plots(results, model_preds,
                                  y_true_ref, data["Tte"], COUNTRY, out_dir,
                                  runtime_h=CFG.SCHEDULER_RUNTIME_H, threshold=CFG.SCHEDULER_THRESHOLD)

    print(
        f"\nSaved metrics, plots, scheduling summaries, and window plots to: {out_dir}")
