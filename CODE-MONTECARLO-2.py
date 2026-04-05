#!/usr/bin/env python3
"""
Hybrid Monte Carlo - Neural Surrogate framework for European option pricing.

This script was written to support a journal-style article on financial option pricing
with Monte Carlo simulation and neural-network surrogates. The code is deliberately
verbose and extensively commented so that each computational step can be audited,
reproduced, adapted, and discussed in the accompanying manuscript.

Main goals
----------
1. Generate a reproducible Monte Carlo benchmark for European call option pricing under
   geometric Brownian motion using exact terminal sampling under the risk-neutral measure.
2. Estimate pathwise deltas so that the neural model can jointly learn prices and Greeks.
3. Train a neural surrogate that maps contract/state variables to price and delta.
4. Produce tables and figures, already organized into folders 5.1, 5.2, and 5.3, so the
   outputs can be inserted directly into Sections 5.1, 5.2, and 5.3 of an article.
5. Compare accuracy, runtime, and discrete-time hedging performance.

Outputs
-------
results/
  5.1/
  5.2/
  5.3/
The script also creates a project ZIP with code, metadata, and all generated outputs.

Design choices
--------------
* The pricing model is the Black-Scholes-Merton framework under risk-neutral dynamics.
* Monte Carlo labels are generated with antithetic variates and exact terminal sampling,
  which isolates the effect of sampling error from time-discretization error.
* A feedforward neural network is trained as a multi-output regressor for normalized
  option price and delta.
* Analytic Black-Scholes values are used only as an external benchmark to quantify the
  fidelity of the Monte Carlo engine and the learned surrogate.

The implementation aims to be fully functional with common scientific Python packages
already available in many execution environments.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import textwrap
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------

SEED = 20260320
RNG = np.random.default_rng(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
SEC_51_DIR = RESULTS_DIR / "5.1"
SEC_52_DIR = RESULTS_DIR / "5.2"
SEC_53_DIR = RESULTS_DIR / "5.3"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
ZIP_PATH = PROJECT_DIR / "hybrid_option_pricing_project.zip"

for directory in [RESULTS_DIR, SEC_51_DIR, SEC_52_DIR, SEC_53_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Dataclasses for structured bookkeeping
# --------------------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    n_train: int = 2800
    n_valid: int = 500
    n_test: int = 900
    k_strike: float = 100.0
    moneyness_low: float = 0.80
    moneyness_high: float = 1.20
    maturity_low: float = 0.05
    maturity_high: float = 2.00
    rate_low: float = 0.00
    rate_high: float = 0.08
    vol_low: float = 0.10
    vol_high: float = 0.60


@dataclass
class LabelSpec:
    # Number of antithetic pairs used to create high-fidelity Monte Carlo labels.
    n_pairs_train_label: int = 2048
    n_pairs_test_label: int = 4096


@dataclass
class ModelSpec:
    hidden_layer_sizes: Tuple[int, ...] = (96, 96, 48)
    activation: str = "relu"
    alpha: float = 1e-5
    batch_size: int = 128
    learning_rate_init: float = 1e-3
    max_iter: int = 300
    early_stopping: bool = True
    validation_fraction: float = 0.12
    n_iter_no_change: int = 25
    random_state: int = SEED


DATASET_SPEC = DatasetSpec()
LABEL_SPEC = LabelSpec()
MODEL_SPEC = ModelSpec()

# --------------------------------------------------------------------------------------
# Financial mathematics helpers
# --------------------------------------------------------------------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density function."""
    return np.exp(-0.5 * x * x) / SQRT_2PI



def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal cumulative distribution function.

    The implementation uses the error function from Python's math module, vectorized
    through NumPy. This avoids an additional SciPy dependency while remaining highly
    accurate for the present application.
    """
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))



def black_scholes_call_price_delta(
    s0: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Black-Scholes-Merton European call price and delta.

    Parameters
    ----------
    s0 : np.ndarray
        Spot price.
    k : np.ndarray
        Strike price.
    r : np.ndarray
        Continuously compounded risk-free rate.
    sigma : np.ndarray
        Volatility.
    t : np.ndarray
        Time to maturity in years.
    """
    s0 = np.asarray(s0, dtype=float)
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    price = s0 * normal_cdf(d1) - k * np.exp(-r * t) * normal_cdf(d2)
    delta = normal_cdf(d1)
    return price, delta



def generate_parameter_grid(n: int, spec: DatasetSpec, rng: np.random.Generator) -> pd.DataFrame:
    """Sample option parameters from broad continuous ranges.

    The parameterization uses *moneyness* rather than the spot directly, with a fixed
    strike K. This improves scale stability for the neural surrogate while still
    preserving the economically meaningful ratio S0 / K.
    """
    moneyness = rng.uniform(spec.moneyness_low, spec.moneyness_high, size=n)
    maturity = rng.uniform(spec.maturity_low, spec.maturity_high, size=n)
    rate = rng.uniform(spec.rate_low, spec.rate_high, size=n)
    vol = rng.uniform(spec.vol_low, spec.vol_high, size=n)
    strike = np.full(n, spec.k_strike)
    spot = moneyness * strike
    return pd.DataFrame(
        {
            "moneyness": moneyness,
            "spot": spot,
            "strike": strike,
            "maturity": maturity,
            "rate": rate,
            "volatility": vol,
        }
    )



def mc_price_delta_antithetic_batch(
    spot: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    vol: np.ndarray,
    maturity: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Vectorized Monte Carlo estimator for price and pathwise delta.

    Notes
    -----
    * Exact terminal sampling under geometric Brownian motion is used:
      S_T = S_0 exp((r - 0.5 sigma^2) T + sigma sqrt(T) Z)
    * Antithetic variates use Z and -Z for each contract.
    * Delta is estimated with the pathwise derivative estimator:
      Delta = exp(-rT) E[ 1_{S_T > K} * S_T / S_0 ]
    * Standard errors are estimated from the per-path discounted payoff samples.

    The function processes a full batch of heterogeneous contracts simultaneously.
    This is far more efficient than looping contract-by-contract in Python.
    """
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    rate = np.asarray(rate, dtype=float)
    vol = np.asarray(vol, dtype=float)
    maturity = np.asarray(maturity, dtype=float)

    n_contracts = spot.shape[0]
    sqrt_t = np.sqrt(maturity)
    drift = (rate - 0.5 * vol * vol) * maturity
    diffusion = vol * sqrt_t
    discount = np.exp(-rate * maturity)

    z = rng.standard_normal(size=(n_contracts, n_pairs))
    z_antithetic = -z

    # Terminal prices for original and antithetic samples.
    st_pos = spot[:, None] * np.exp(drift[:, None] + diffusion[:, None] * z)
    st_neg = spot[:, None] * np.exp(drift[:, None] + diffusion[:, None] * z_antithetic)

    payoff_pos = np.maximum(st_pos - strike[:, None], 0.0)
    payoff_neg = np.maximum(st_neg - strike[:, None], 0.0)

    discounted_pos = discount[:, None] * payoff_pos
    discounted_neg = discount[:, None] * payoff_neg

    # Average the antithetic pair at the sample level. This yields n_pairs effective
    # paired observations per contract, which is the natural basis for the standard
    # error estimate under antithetic sampling.
    paired_discounted = 0.5 * (discounted_pos + discounted_neg)
    mc_price = paired_discounted.mean(axis=1)
    mc_price_std = paired_discounted.std(axis=1, ddof=1)
    mc_price_se = mc_price_std / np.sqrt(n_pairs)

    delta_pos = discount[:, None] * ((st_pos > strike[:, None]) * (st_pos / spot[:, None]))
    delta_neg = discount[:, None] * ((st_neg > strike[:, None]) * (st_neg / spot[:, None]))
    paired_delta = 0.5 * (delta_pos + delta_neg)
    mc_delta = paired_delta.mean(axis=1)
    mc_delta_std = paired_delta.std(axis=1, ddof=1)
    mc_delta_se = mc_delta_std / np.sqrt(n_pairs)

    return {
        "mc_price": mc_price,
        "mc_price_std": mc_price_std,
        "mc_price_se": mc_price_se,
        "mc_delta": mc_delta,
        "mc_delta_std": mc_delta_std,
        "mc_delta_se": mc_delta_se,
    }



def label_dataframe_with_mc(
    df: pd.DataFrame,
    n_pairs: int,
    rng: np.random.Generator,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Attach Monte Carlo and analytical labels to a parameter dataframe.

    The batching strategy limits peak memory consumption while retaining most of the
    benefits of vectorization.
    """
    records: List[pd.DataFrame] = []
    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch = df.iloc[start:stop].copy()
        mc = mc_price_delta_antithetic_batch(
            batch["spot"].to_numpy(),
            batch["strike"].to_numpy(),
            batch["rate"].to_numpy(),
            batch["volatility"].to_numpy(),
            batch["maturity"].to_numpy(),
            n_pairs=n_pairs,
            rng=rng,
        )
        bs_price, bs_delta = black_scholes_call_price_delta(
            batch["spot"].to_numpy(),
            batch["strike"].to_numpy(),
            batch["rate"].to_numpy(),
            batch["volatility"].to_numpy(),
            batch["maturity"].to_numpy(),
        )
        batch["bs_price"] = bs_price
        batch["bs_delta"] = bs_delta
        for key, value in mc.items():
            batch[key] = value
        batch["normalized_bs_price"] = batch["bs_price"] / batch["strike"]
        batch["normalized_mc_price"] = batch["mc_price"] / batch["strike"]
        batch["price_abs_mc_error"] = np.abs(batch["mc_price"] - batch["bs_price"])
        batch["delta_abs_mc_error"] = np.abs(batch["mc_delta"] - batch["bs_delta"])
        records.append(batch)
    return pd.concat(records, ignore_index=True)


# --------------------------------------------------------------------------------------
# Neural surrogate helpers
# --------------------------------------------------------------------------------------


def make_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Construct the neural-network input matrix.

    The inputs are deliberately compact and finance-aware:
    * moneyness = S0 / K
    * maturity T
    * rate r
    * volatility sigma
    """
    return df[["moneyness", "maturity", "rate", "volatility"]].to_numpy(dtype=float)



def make_target_matrix(df: pd.DataFrame) -> np.ndarray:
    """Construct the multi-output target matrix.

    Targets use the high-fidelity Monte Carlo labels, not the analytical formula, to stay
    faithful to the hybrid Monte Carlo-neural philosophy of the study.
    """
    return df[["normalized_mc_price", "mc_delta"]].to_numpy(dtype=float)



def fit_neural_surrogate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    spec: ModelSpec,
) -> Dict[str, object]:
    """Fit a multi-output neural network with standardization.

    The MLPRegressor natively supports multi-output regression, which is sufficient for
    the current two-target setting (price and delta). Targets are standardized manually
    to stabilize joint optimization.
    """
    x_train = make_feature_matrix(train_df)
    y_train = make_target_matrix(train_df)
    x_valid = make_feature_matrix(valid_df)
    y_valid = make_target_matrix(valid_df)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_scaled = x_scaler.fit_transform(x_train)
    x_valid_scaled = x_scaler.transform(x_valid)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_valid_scaled = y_scaler.transform(y_valid)

    model = MLPRegressor(
        hidden_layer_sizes=spec.hidden_layer_sizes,
        activation=spec.activation,
        solver="adam",
        alpha=spec.alpha,
        batch_size=spec.batch_size,
        learning_rate="adaptive",
        learning_rate_init=spec.learning_rate_init,
        max_iter=spec.max_iter,
        early_stopping=spec.early_stopping,
        validation_fraction=spec.validation_fraction,
        n_iter_no_change=spec.n_iter_no_change,
        random_state=spec.random_state,
        verbose=False,
    )

    start_time = time.perf_counter()
    model.fit(x_train_scaled, y_train_scaled)
    train_seconds = time.perf_counter() - start_time

    y_train_pred_scaled = model.predict(x_train_scaled)
    y_valid_pred_scaled = model.predict(x_valid_scaled)
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
    y_valid_pred = y_scaler.inverse_transform(y_valid_pred_scaled)

    diagnostics = {
        "train_seconds": train_seconds,
        "n_iter_": int(model.n_iter_),
        "final_loss_": float(model.loss_),
        "best_validation_score_": (
            float(model.best_validation_score_)
            if hasattr(model, "best_validation_score_") and model.best_validation_score_ is not None
            else None
        ),
        "train_price_rmse": float(np.sqrt(np.mean((y_train_pred[:, 0] - y_train[:, 0]) ** 2))),
        "train_delta_rmse": float(np.sqrt(np.mean((y_train_pred[:, 1] - y_train[:, 1]) ** 2))),
        "valid_price_rmse": float(np.sqrt(np.mean((y_valid_pred[:, 0] - y_valid[:, 0]) ** 2))),
        "valid_delta_rmse": float(np.sqrt(np.mean((y_valid_pred[:, 1] - y_valid[:, 1]) ** 2))),
    }

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "diagnostics": diagnostics,
    }



def neural_predict(df: pd.DataFrame, fitted: Dict[str, object]) -> pd.DataFrame:
    """Generate neural predictions for a dataframe and append them.

    The price target is learned in normalized form and then rescaled by the strike.
    """
    out = df.copy()
    x = make_feature_matrix(df)
    x_scaled = fitted["x_scaler"].transform(x)
    y_scaled_pred = fitted["model"].predict(x_scaled)
    y_pred = fitted["y_scaler"].inverse_transform(y_scaled_pred)
    # Enforce two economically meaningful constraints after inverse scaling:
    # 1. European call prices cannot be negative.
    # 2. Under the Black-Scholes setting, call delta should remain in [0, 1].
    out["nn_normalized_price"] = np.maximum(y_pred[:, 0], 0.0)
    out["nn_delta"] = np.clip(y_pred[:, 1], 0.0, 1.0)
    out["nn_price"] = out["nn_normalized_price"] * out["strike"]
    out["price_abs_nn_error"] = np.abs(out["nn_price"] - out["bs_price"])
    out["delta_abs_nn_error"] = np.abs(out["nn_delta"] - out["bs_delta"])
    return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)))



def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))



def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    denom = np.maximum(np.abs(actual), eps)
    return float(np.mean(np.abs(predicted - actual) / denom) * 100.0)



def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------------------
# Section 5.1 Monte Carlo benchmark analysis
# --------------------------------------------------------------------------------------


def benchmark_monte_carlo_path_counts(test_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Benchmark the Monte Carlo engine over a range of path counts.

    Each row measures the average pricing error against exact Black-Scholes values for
    the same contracts, along with the average estimated Monte Carlo standard error and
    wall-clock runtime.
    """
    subset = test_df.sample(n=min(300, len(test_df)), random_state=SEED).reset_index(drop=True)
    path_counts = [128, 256, 512, 1024, 2048, 4096, 8192]
    rows = []

    for n_pairs in path_counts:
        start = time.perf_counter()
        estimates = mc_price_delta_antithetic_batch(
            subset["spot"].to_numpy(),
            subset["strike"].to_numpy(),
            subset["rate"].to_numpy(),
            subset["volatility"].to_numpy(),
            subset["maturity"].to_numpy(),
            n_pairs=n_pairs,
            rng=rng,
        )
        elapsed = time.perf_counter() - start

        price = estimates["mc_price"]
        delta = estimates["mc_delta"]
        rows.append(
            {
                "antithetic_pairs": n_pairs,
                "effective_paths": 2 * n_pairs,
                "price_mae": mae(subset["bs_price"], price),
                "price_rmse": rmse(subset["bs_price"], price),
                "delta_mae": mae(subset["bs_delta"], delta),
                "delta_rmse": rmse(subset["bs_delta"], delta),
                "avg_price_se": float(np.mean(estimates["mc_price_se"])),
                "avg_delta_se": float(np.mean(estimates["mc_delta_se"])),
                "runtime_seconds": elapsed,
            }
        )

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(SEC_51_DIR / "table_1_monte_carlo_path_benchmark.csv", index=False)
    return benchmark_df



def plot_mc_benchmark(benchmark_df: pd.DataFrame) -> None:
    """Create plots for Section 5.1."""
    # Figure 1: Pricing RMSE decay versus the number of effective paths.
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    ax.plot(benchmark_df["effective_paths"], benchmark_df["price_rmse"], marker="o", label="Price RMSE")
    ax.plot(benchmark_df["effective_paths"], benchmark_df["avg_price_se"], marker="s", label="Average price SE")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Effective Monte Carlo paths (log2 scale)")
    ax.set_ylabel("Error magnitude")
    ax.set_title("Monte Carlo pricing error decreases as path count increases")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SEC_51_DIR / "figure_1_mc_error_decay.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Runtime versus path count.
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    ax.plot(benchmark_df["effective_paths"], benchmark_df["runtime_seconds"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Effective Monte Carlo paths (log2 scale)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Monte Carlo runtime grows with the number of simulated paths")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SEC_51_DIR / "figure_2_mc_runtime_growth.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Section 5.2 Neural surrogate analysis
# --------------------------------------------------------------------------------------


def summarize_neural_accuracy(test_pred_df: pd.DataFrame, diagnostics: Dict[str, float]) -> pd.DataFrame:
    """Create a compact accuracy table for the neural surrogate."""
    summary = pd.DataFrame(
        [
            {
                "metric": "Price MAE",
                "value": mae(test_pred_df["bs_price"], test_pred_df["nn_price"]),
            },
            {
                "metric": "Price RMSE",
                "value": rmse(test_pred_df["bs_price"], test_pred_df["nn_price"]),
            },
            {
                "metric": "Price MAPE (%) for BS price > 1",
                "value": mape(
                    test_pred_df.loc[test_pred_df["bs_price"] > 1.0, "bs_price"],
                    test_pred_df.loc[test_pred_df["bs_price"] > 1.0, "nn_price"],
                ),
            },
            {
                "metric": "Price R^2",
                "value": r2_score(test_pred_df["bs_price"], test_pred_df["nn_price"]),
            },
            {
                "metric": "Delta MAE",
                "value": mae(test_pred_df["bs_delta"], test_pred_df["nn_delta"]),
            },
            {
                "metric": "Delta RMSE",
                "value": rmse(test_pred_df["bs_delta"], test_pred_df["nn_delta"]),
            },
            {
                "metric": "Delta R^2",
                "value": r2_score(test_pred_df["bs_delta"], test_pred_df["nn_delta"]),
            },
            {
                "metric": "Training time (s)",
                "value": diagnostics["train_seconds"],
            },
            {
                "metric": "Training epochs (iterations)",
                "value": diagnostics["n_iter_"],
            },
        ]
    )
    summary.to_csv(SEC_52_DIR / "table_2_neural_surrogate_accuracy.csv", index=False)
    return summary



def plot_neural_accuracy(test_pred_df: pd.DataFrame) -> None:
    """Create diagnostic plots for Section 5.2."""
    # Figure 3: predicted versus exact price.
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200)
    ax.scatter(test_pred_df["bs_price"], test_pred_df["nn_price"], s=14, alpha=0.55)
    min_val = float(min(test_pred_df["bs_price"].min(), test_pred_df["nn_price"].min()))
    max_val = float(max(test_pred_df["bs_price"].max(), test_pred_df["nn_price"].max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Exact Black-Scholes price")
    ax.set_ylabel("Neural surrogate price")
    ax.set_title("Price predictions closely track the analytical benchmark")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(SEC_52_DIR / "figure_3_price_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: predicted versus exact delta.
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200)
    ax.scatter(test_pred_df["bs_delta"], test_pred_df["nn_delta"], s=14, alpha=0.55)
    min_val = float(min(test_pred_df["bs_delta"].min(), test_pred_df["nn_delta"].min()))
    max_val = float(max(test_pred_df["bs_delta"].max(), test_pred_df["nn_delta"].max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Exact Black-Scholes delta")
    ax.set_ylabel("Neural surrogate delta")
    ax.set_title("Joint learning extends naturally from prices to Greeks")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(SEC_52_DIR / "figure_4_delta_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 5: Heatmap of mean absolute pricing error by binned moneyness and maturity.
    temp = test_pred_df.copy()
    temp["moneyness_bin"] = pd.cut(temp["moneyness"], bins=np.linspace(0.80, 1.20, 9), include_lowest=True)
    temp["maturity_bin"] = pd.cut(temp["maturity"], bins=np.linspace(0.05, 2.00, 9), include_lowest=True)
    pivot = (
        temp.groupby(["maturity_bin", "moneyness_bin"], observed=False)["price_abs_nn_error"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=200)
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_title("Mean absolute price error across moneyness-maturity cells")
    ax.set_xlabel("Moneyness bin")
    ax.set_ylabel("Maturity bin")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{interval.left:.2f}-{interval.right:.2f}" for interval in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{interval.left:.2f}-{interval.right:.2f}" for interval in pivot.index])
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean absolute error")
    fig.tight_layout()
    fig.savefig(SEC_52_DIR / "figure_5_price_error_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Section 5.3 Runtime and hedging analysis
# --------------------------------------------------------------------------------------


def runtime_comparison(test_pred_df: pd.DataFrame, fitted: Dict[str, object], rng: np.random.Generator) -> pd.DataFrame:
    """Compare runtime per contract for three evaluators.

    The comparison is deliberately done on the same test contracts:
    * analytical Black-Scholes formula
    * Monte Carlo with a realistic path budget
    * trained neural surrogate
    """
    df = test_pred_df.copy().reset_index(drop=True)

    # Analytical benchmark runtime.
    start = time.perf_counter()
    _ = black_scholes_call_price_delta(
        df["spot"].to_numpy(),
        df["strike"].to_numpy(),
        df["rate"].to_numpy(),
        df["volatility"].to_numpy(),
        df["maturity"].to_numpy(),
    )
    bs_seconds = time.perf_counter() - start

    # Monte Carlo runtime at an accuracy level commonly used for production-style surrogates.
    start = time.perf_counter()
    _ = mc_price_delta_antithetic_batch(
        df["spot"].to_numpy(),
        df["strike"].to_numpy(),
        df["rate"].to_numpy(),
        df["volatility"].to_numpy(),
        df["maturity"].to_numpy(),
        n_pairs=4096,
        rng=rng,
    )
    mc_seconds = time.perf_counter() - start

    # Neural inference runtime.
    start = time.perf_counter()
    _ = neural_predict(df, fitted)
    nn_seconds = time.perf_counter() - start

    n = len(df)
    runtime_df = pd.DataFrame(
        [
            {
                "method": "Analytical Black-Scholes",
                "total_seconds": bs_seconds,
                "seconds_per_contract": bs_seconds / n,
                "contracts_per_second": n / bs_seconds if bs_seconds > 0 else np.nan,
            },
            {
                "method": "Monte Carlo (4096 antithetic pairs)",
                "total_seconds": mc_seconds,
                "seconds_per_contract": mc_seconds / n,
                "contracts_per_second": n / mc_seconds if mc_seconds > 0 else np.nan,
            },
            {
                "method": "Neural surrogate",
                "total_seconds": nn_seconds,
                "seconds_per_contract": nn_seconds / n,
                "contracts_per_second": n / nn_seconds if nn_seconds > 0 else np.nan,
            },
        ]
    )
    runtime_df.to_csv(SEC_53_DIR / "table_3_runtime_comparison.csv", index=False)
    return runtime_df



def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate full GBM paths for hedging experiments.

    The hedging study requires intermediate asset values, so a full path simulation is
    performed instead of exact terminal sampling.
    """
    dt = maturity / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)
    z = rng.standard_normal(size=(n_paths, n_steps))
    increments = drift + diffusion * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
    return s0 * np.exp(log_paths)



def neural_delta_for_state(
    spot: np.ndarray,
    strike: float,
    remaining_time: float,
    rate: float,
    vol: float,
    fitted: Dict[str, object],
) -> np.ndarray:
    """Evaluate the neural delta for a given vector of states."""
    df = pd.DataFrame(
        {
            "moneyness": spot / strike,
            "spot": spot,
            "strike": strike,
            "maturity": np.full_like(spot, remaining_time, dtype=float),
            "rate": np.full_like(spot, rate, dtype=float),
            "volatility": np.full_like(spot, vol, dtype=float),
            # Dummy analytical columns are not needed for the predictor utility.
        }
    )
    x = make_feature_matrix(df)
    x_scaled = fitted["x_scaler"].transform(x)
    y_scaled = fitted["model"].predict(x_scaled)
    y = fitted["y_scaler"].inverse_transform(y_scaled)
    return np.clip(y[:, 1], 0.0, 1.0)



def discrete_delta_hedging_experiment(
    fitted: Dict[str, object],
    rng: np.random.Generator,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.20,
    maturity: float = 1.0,
    n_steps: int = 52,
    n_paths: int = 1500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare hedging error distributions for three strategies.

    Strategies
    ----------
    1. Analytical delta hedge: rebalances using exact Black-Scholes delta.
    2. Neural delta hedge: rebalances using the surrogate delta.
    3. Frozen delta hedge: uses the initial exact delta and never rebalances.

    Portfolio convention
    --------------------
    The trader sells one call, receives the premium, and attempts to replicate the option.
    Hedging error = terminal portfolio value - option payoff.
    Lower dispersion around zero indicates better replication.
    """
    dt = maturity / n_steps
    time_grid = np.linspace(0.0, maturity, n_steps + 1)
    paths = simulate_gbm_paths(s0, r, sigma, maturity, n_steps, n_paths, rng)

    initial_price, initial_delta = black_scholes_call_price_delta(
        np.array([s0]), np.array([strike]), np.array([r]), np.array([sigma]), np.array([maturity])
    )
    initial_price = float(initial_price[0])
    initial_delta = float(initial_delta[0])

    strategy_errors: Dict[str, np.ndarray] = {}

    for strategy in ["Analytical delta", "Neural delta", "Frozen initial delta"]:
        if strategy == "Frozen initial delta":
            delta = np.full(n_paths, initial_delta)
        elif strategy == "Analytical delta":
            delta = np.full(n_paths, initial_delta)
        else:
            delta = np.full(n_paths, initial_delta)

        # The cash account is initialized so that the replicating portfolio matches the
        # option premium at time zero: V0 = Delta0 * S0 + B0 = C0.
        cash = np.full(n_paths, initial_price - delta * s0)

        for step in range(n_steps):
            # Cash accrues interest over the interval [t_i, t_{i+1}].
            cash *= math.exp(r * dt)
            s_next = paths[:, step + 1]

            # Determine the new hedge ratio after observing the new state.
            remaining_time = max(maturity - time_grid[step + 1], 1e-8)
            if strategy == "Frozen initial delta":
                new_delta = delta
            elif strategy == "Analytical delta":
                _, new_delta = black_scholes_call_price_delta(
                    s_next,
                    np.full(n_paths, strike),
                    np.full(n_paths, r),
                    np.full(n_paths, sigma),
                    np.full(n_paths, remaining_time),
                )
            else:
                new_delta = neural_delta_for_state(s_next, strike, remaining_time, r, sigma, fitted)

            # Rebalancing: buy/sell stock and finance through the cash account.
            cash -= (new_delta - delta) * s_next
            delta = new_delta

        terminal_portfolio = delta * paths[:, -1] + cash
        option_payoff = np.maximum(paths[:, -1] - strike, 0.0)
        hedge_error = terminal_portfolio - option_payoff
        strategy_errors[strategy] = hedge_error

    summary_rows = []
    for strategy, errors in strategy_errors.items():
        summary_rows.append(
            {
                "strategy": strategy,
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors, ddof=1)),
                "rmse_error": float(np.sqrt(np.mean(errors ** 2))),
                "mae_error": float(np.mean(np.abs(errors))),
                "p05": float(np.quantile(errors, 0.05)),
                "median": float(np.quantile(errors, 0.50)),
                "p95": float(np.quantile(errors, 0.95)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(strategy_errors)
    summary_df.to_csv(SEC_53_DIR / "table_4_hedging_error_summary.csv", index=False)
    detailed_df.to_csv(SEC_53_DIR / "hedging_errors_raw.csv", index=False)
    return summary_df, detailed_df



def plot_runtime_and_hedging(runtime_df: pd.DataFrame, hedging_summary_df: pd.DataFrame, hedging_raw_df: pd.DataFrame) -> None:
    """Create plots for Section 5.3."""
    # Figure 6: runtime comparison.
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200)
    ax.bar(runtime_df["method"], runtime_df["seconds_per_contract"])
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per contract (log scale)")
    ax.set_title("Neural inference is much faster than Monte Carlo evaluation")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(SEC_53_DIR / "figure_6_runtime_bar_chart.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 7: hedging error distributions.
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    bins = np.linspace(
        min(float(hedging_raw_df.min().min()), -5.0),
        max(float(hedging_raw_df.max().max()), 5.0),
        70,
    )
    for column in hedging_raw_df.columns:
        ax.hist(hedging_raw_df[column], bins=bins, alpha=0.45, density=True, label=column)
    ax.set_xlabel("Terminal hedging error")
    ax.set_ylabel("Density")
    ax.set_title("Dynamic hedging sharply reduces replication error dispersion")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SEC_53_DIR / "figure_7_hedging_error_distributions.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Utility writers
# --------------------------------------------------------------------------------------


def write_readme(metadata: Dict[str, object]) -> None:
    """Write a short README explaining the project structure."""
    readme_text = f"""
    Hybrid Monte Carlo - Neural Option Pricing Project
    ==================================================

    This directory contains the executable code and generated outputs for a research-style
    article on option pricing, Greeks, and hedging under a Monte Carlo-neural workflow.

    Key files
    ---------
    * hybrid_option_pricing_article.py  -> main executable script
    * models/                          -> serialized neural model and scalers
    * results/5.1/                     -> Section 5.1 tables and figures
    * results/5.2/                     -> Section 5.2 tables and figures
    * results/5.3/                     -> Section 5.3 tables and figures
    * reports/run_metadata.json        -> numerical metadata and diagnostics

    Reproducibility
    ---------------
    The script uses a fixed random seed ({SEED}). Results are therefore deterministic up to
    platform-level floating-point differences.

    Metadata snapshot
    -----------------
    {json.dumps(metadata, indent=2)}
    """
    (PROJECT_DIR / "README.txt").write_text(textwrap.dedent(readme_text).strip() + "\n", encoding="utf-8")



def save_model_bundle(fitted: Dict[str, object]) -> None:
    """Serialize the trained neural surrogate and its preprocessors."""
    joblib.dump(fitted["model"], MODELS_DIR / "mlp_surrogate.joblib")
    joblib.dump(fitted["x_scaler"], MODELS_DIR / "x_scaler.joblib")
    joblib.dump(fitted["y_scaler"], MODELS_DIR / "y_scaler.joblib")



def create_zip_archive() -> None:
    """Create a ZIP archive that bundles code, model, reports, and results."""
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PROJECT_DIR.rglob("*")):
            if path == ZIP_PATH:
                continue
            if path.is_dir():
                continue
            zf.write(path, arcname=path.relative_to(PROJECT_DIR))



def format_table_for_markdown(df: pd.DataFrame, decimals: int = 4) -> str:
    """Convert a dataframe to a compact markdown table string.

    This helper is used later by the article-generation pipeline so that the numerical
    tables inserted into the manuscript match the executed results.
    """
    temp = df.copy()
    for column in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[column]):
            temp[column] = temp[column].map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")
    return temp.to_markdown(index=False)


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def main() -> None:
    project_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Build train/validation/test datasets.
    # ------------------------------------------------------------------
    train_base = generate_parameter_grid(DATASET_SPEC.n_train, DATASET_SPEC, RNG)
    valid_base = generate_parameter_grid(DATASET_SPEC.n_valid, DATASET_SPEC, RNG)
    test_base = generate_parameter_grid(DATASET_SPEC.n_test, DATASET_SPEC, RNG)

    train_df = label_dataframe_with_mc(train_base, LABEL_SPEC.n_pairs_train_label, RNG)
    valid_df = label_dataframe_with_mc(valid_base, LABEL_SPEC.n_pairs_train_label, RNG)
    test_df = label_dataframe_with_mc(test_base, LABEL_SPEC.n_pairs_test_label, RNG)

    train_df.to_csv(REPORTS_DIR / "train_dataset.csv", index=False)
    valid_df.to_csv(REPORTS_DIR / "valid_dataset.csv", index=False)
    test_df.to_csv(REPORTS_DIR / "test_dataset.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Fit the neural surrogate.
    # ------------------------------------------------------------------
    fitted = fit_neural_surrogate(train_df, valid_df, MODEL_SPEC)
    save_model_bundle(fitted)

    test_pred_df = neural_predict(test_df, fitted)
    test_pred_df.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Produce Section 5.1 outputs.
    # ------------------------------------------------------------------
    mc_benchmark_df = benchmark_monte_carlo_path_counts(test_df, RNG)
    plot_mc_benchmark(mc_benchmark_df)

    # ------------------------------------------------------------------
    # 4. Produce Section 5.2 outputs.
    # ------------------------------------------------------------------
    nn_summary_df = summarize_neural_accuracy(test_pred_df, fitted["diagnostics"])
    plot_neural_accuracy(test_pred_df)

    # ------------------------------------------------------------------
    # 5. Produce Section 5.3 outputs.
    # ------------------------------------------------------------------
    runtime_df = runtime_comparison(test_pred_df, fitted, RNG)
    hedging_summary_df, hedging_raw_df = discrete_delta_hedging_experiment(fitted, RNG)
    plot_runtime_and_hedging(runtime_df, hedging_summary_df, hedging_raw_df)

    # ------------------------------------------------------------------
    # 6. Save machine-readable metadata for article generation.
    # ------------------------------------------------------------------
    metadata = {
        "seed": SEED,
        "dataset_spec": asdict(DATASET_SPEC),
        "label_spec": asdict(LABEL_SPEC),
        "model_spec": asdict(MODEL_SPEC),
        "model_diagnostics": fitted["diagnostics"],
        "monte_carlo_benchmark": mc_benchmark_df.to_dict(orient="records"),
        "neural_summary": nn_summary_df.to_dict(orient="records"),
        "runtime_comparison": runtime_df.to_dict(orient="records"),
        "hedging_summary": hedging_summary_df.to_dict(orient="records"),
        "project_runtime_seconds": time.perf_counter() - project_start,
    }
    (REPORTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(metadata)
    create_zip_archive()

    # ------------------------------------------------------------------
    # 7. Print a concise terminal summary.
    # ------------------------------------------------------------------
    summary = {
        "project_runtime_seconds": round(metadata["project_runtime_seconds"], 3),
        "price_rmse_nn": round(float(nn_summary_df.loc[nn_summary_df["metric"] == "Price RMSE", "value"].iloc[0]), 6),
        "delta_rmse_nn": round(float(nn_summary_df.loc[nn_summary_df["metric"] == "Delta RMSE", "value"].iloc[0]), 6),
        "zip_path": str(ZIP_PATH),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
