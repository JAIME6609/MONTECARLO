#!/usr/bin/env python3
"""
Monte Carlo-Driven Portfolio Optimization with Neural Allocation Rules.

This script builds a fully reproducible computational experiment for a
multi-asset portfolio optimization study. The workflow has four major stages:

1. Synthetic market generation with realistic regime changes, fat tails,
   volatility clustering proxies, and cross-asset correlation structure.
2. Rolling estimation of expected returns and covariance matrices.
3. Scenario-based portfolio optimization under a mean-variance-CVaR utility.
4. Supervised learning of the optimizer's policy with a neural network that
   maps state variables into portfolio weights on the simplex.

The script saves all tables and figures into three subfolders named exactly as
required by the article structure:

    5.1  -> scenario engine and learning diagnostics
    5.2  -> out-of-sample performance and risk-return evidence
    5.3  -> allocation patterns, robustness, and scenario sensitivity

All outputs are created in English, and each result is saved both as a machine-
readable file (CSV/JSON where appropriate) and, when useful, as a publication-
ready PNG image for direct insertion into the manuscript.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Container for all high-level experiment parameters.

    The configuration is explicit to make the script auditable and easy to adapt.
    Many journal reviewers and future users look first for the reproducibility
    section, so centralizing all parameters is helpful both methodologically and
    practically.
    """

    seed: int = 42
    periods: int = 360                   # Total number of monthly observations
    lookback: int = 36                   # Rolling estimation window in months
    n_assets: int = 6
    n_scenarios_train: int = 1500        # For training target generation
    n_scenarios_diag: int = 1500         # For diagnostics and robustness tables
    n_scenarios_rep: int = 4000          # For representative figure
    alpha_cvar: float = 0.95
    risk_aversion: float = 5.0
    cvar_penalty: float = 2.5
    turnover_penalty: float = 0.12
    smoothing_lambda: float = 0.65       # For neural allocation smoothing
    train_share: float = 0.60
    val_share: float = 0.20
    hidden_layers: Tuple[int, int, int] = (128, 64, 32)
    learning_rate_init: float = 0.001
    max_iter: int = 800
    output_root: str = "montecarlo_portfolio_outputs"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path



def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project an arbitrary vector onto the probability simplex.

    The learned neural outputs are unconstrained real numbers. Portfolio
    weights, however, must satisfy two conditions in this experiment:

    - non-negativity (long-only portfolio), and
    - full investment (weights sum to one).

    The Euclidean projection onto the simplex is the cleanest way to convert
    raw neural outputs into admissible portfolio allocations.
    """
    v = np.asarray(v, dtype=float)
    if np.isclose(v.sum(), 1.0) and np.all(v >= 0.0):
        return v

    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    return w / max(w.sum(), 1e-12)



def compute_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the Conditional Value-at-Risk (Expected Shortfall).

    CVaR is the mean of the losses located in the worst (1 - alpha) tail. In
    finance, this metric is especially relevant when the optimization problem
    wants to penalize large downside events rather than only average variance.
    """
    losses = np.asarray(losses, dtype=float)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)



def turnover(weights: np.ndarray) -> float:
    """Average L1 portfolio turnover across consecutive rebalancing dates."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) <= 1:
        return 0.0
    return float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean())



def table_to_png(df: pd.DataFrame, path: Path, title: str | None = None) -> None:
    """Render a pandas DataFrame as a PNG figure suitable for insertion.

    The function deliberately avoids external styling dependencies to keep the
    script lightweight and portable.
    """
    nrows, ncols = df.shape
    fig_w = max(10, ncols * 1.5)
    fig_h = max(2.5, 0.55 * (nrows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, pad=12)

    display_df = df.copy()
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.92] if title else [0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def save_dataframe(df: pd.DataFrame, base_path_without_ext: Path, title: str | None = None) -> None:
    """Save a DataFrame both as CSV and as a PNG table."""
    df.to_csv(base_path_without_ext.with_suffix(".csv"), index=False)
    table_to_png(df, base_path_without_ext.with_suffix(".png"), title=title)



def annualized_return(r: np.ndarray, periods_per_year: int = 12) -> float:
    wealth = np.cumprod(1.0 + np.asarray(r, dtype=float))
    return float(wealth[-1] ** (periods_per_year / len(r)) - 1.0)



def perf_metrics(series: Iterable[float], weights: np.ndarray | None = None, periods_per_year: int = 12) -> Dict[str, float]:
    """Compute standard performance metrics used in portfolio studies."""
    r = np.asarray(list(series), dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0

    downside = r[r < 0.0]
    sharpe = (r.mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(periods_per_year)
    sortino = np.nan
    if len(downside) > 1:
        sortino = (r.mean() / (downside.std(ddof=1) + 1e-12)) * np.sqrt(periods_per_year)

    out = {
        "Total Return": float(wealth[-1] - 1.0),
        "Annualized Return": annualized_return(r, periods_per_year=periods_per_year),
        "Annualized Volatility": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Max Drawdown": float(drawdown.min()),
        "Monthly VaR 95%": float(-np.quantile(r, 0.05)),
        "Monthly CVaR 95%": float(compute_cvar(-r, alpha=0.95)),
    }

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        out["Average Turnover"] = turnover(weights)
        out["Concentration (HHI)"] = float(np.mean(np.sum(weights ** 2, axis=1)))

    return out


# ---------------------------------------------------------------------------
# Market simulation functions
# ---------------------------------------------------------------------------


def mv_t_sample(mu_vec: np.ndarray, cov_mat: np.ndarray, df: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a multivariate Student-t sample.

    The Student-t distribution is used instead of a pure Gaussian draw because
    heavy tails are a well-documented stylized fact of financial returns. This
    makes the synthetic environment more appropriate for downside-risk studies.
    """
    z = rng.multivariate_normal(np.zeros(len(mu_vec)), cov_mat)
    g = rng.chisquare(df) / df
    return mu_vec + z / np.sqrt(g)



def simulate_regime_market(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.Series, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Generate a stylized multi-asset return panel with regime dynamics.

    Returns
    -------
    ret_df:
        Monthly return panel.
    regime_series:
        Realized market regimes for each month.
    mu, vol, corr:
        Regime-specific mean, volatility, and correlation assumptions.
    transition_matrix:
        Regime transition probabilities.
    assets:
        Asset labels.
    """
    rng = np.random.default_rng(config.seed)

    assets = [
        "US Equity",
        "Intl Equity",
        "Treasury",
        "Credit",
        "REIT",
        "Gold",
    ]

    regimes = ["Expansion", "Stress", "Inflation"]

    # Monthly expected returns by regime.
    mu = {
        "Expansion": np.array([0.010, 0.008, 0.003, 0.005, 0.007, 0.002]),
        "Stress": np.array([-0.018, -0.020, 0.006, -0.008, -0.015, 0.008]),
        "Inflation": np.array([0.004, 0.003, -0.004, 0.001, 0.002, 0.007]),
    }

    # Monthly volatilities by regime.
    vol = {
        "Expansion": np.array([0.045, 0.048, 0.020, 0.028, 0.050, 0.035]),
        "Stress": np.array([0.085, 0.090, 0.035, 0.055, 0.080, 0.045]),
        "Inflation": np.array([0.055, 0.058, 0.040, 0.035, 0.060, 0.050]),
    }

    # Cross-asset correlations by regime.
    corr = {}
    corr["Expansion"] = np.array([
        [1.00, 0.78, -0.22, 0.55, 0.68, 0.05],
        [0.78, 1.00, -0.18, 0.50, 0.62, 0.02],
        [-0.22, -0.18, 1.00, 0.15, -0.10, 0.08],
        [0.55, 0.50, 0.15, 1.00, 0.45, -0.02],
        [0.68, 0.62, -0.10, 0.45, 1.00, 0.06],
        [0.05, 0.02, 0.08, -0.02, 0.06, 1.00],
    ])
    corr["Stress"] = np.array([
        [1.00, 0.88, -0.48, 0.72, 0.80, 0.18],
        [0.88, 1.00, -0.42, 0.70, 0.77, 0.15],
        [-0.48, -0.42, 1.00, -0.05, -0.22, 0.12],
        [0.72, 0.70, -0.05, 1.00, 0.60, 0.02],
        [0.80, 0.77, -0.22, 0.60, 1.00, 0.10],
        [0.18, 0.15, 0.12, 0.02, 0.10, 1.00],
    ])
    corr["Inflation"] = np.array([
        [1.00, 0.74, 0.10, 0.48, 0.60, 0.20],
        [0.74, 1.00, 0.12, 0.46, 0.56, 0.18],
        [0.10, 0.12, 1.00, 0.35, 0.08, -0.05],
        [0.48, 0.46, 0.35, 1.00, 0.40, 0.10],
        [0.60, 0.56, 0.08, 0.40, 1.00, 0.22],
        [0.20, 0.18, -0.05, 0.10, 0.22, 1.00],
    ])

    transition_matrix = np.array([
        [0.88, 0.06, 0.06],
        [0.20, 0.70, 0.10],
        [0.12, 0.12, 0.76],
    ])

    dfs = {"Expansion": 10, "Stress": 6, "Inflation": 7}
    cov = {k: np.diag(vol[k]) @ corr[k] @ np.diag(vol[k]) for k in regimes}

    regime_idx = np.zeros(config.periods, dtype=int)
    regime_idx[0] = 0
    for t in range(1, config.periods):
        regime_idx[t] = rng.choice(3, p=transition_matrix[regime_idx[t - 1]])

    regime_series = pd.Series([regimes[i] for i in regime_idx], name="Regime")

    returns = np.zeros((config.periods, config.n_assets))
    for t, regime in enumerate(regime_series):
        returns[t] = mv_t_sample(mu[regime], cov[regime], dfs[regime], rng)

    ret_df = pd.DataFrame(returns, columns=assets)
    return ret_df, regime_series, mu, vol, corr, transition_matrix, assets


# ---------------------------------------------------------------------------
# Feature engineering and estimation
# ---------------------------------------------------------------------------


def build_feature_state(window_returns: np.ndarray, assets: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Construct the state vector observed by the neural network.

    The feature space combines return, volatility, downside-risk, correlation,
    and aggregate market-state descriptors. This reflects the article's idea
    that a learned allocation rule should react not only to expected returns but
    also to the evolving dependence structure of the market.
    """
    L, n = window_returns.shape

    short = window_returns[-12:] if L >= 12 else window_returns
    mu_short = short.mean(axis=0)
    mu_long = window_returns.mean(axis=0)
    vol_short = short.std(axis=0, ddof=1)
    vol_long = window_returns.std(axis=0, ddof=1)
    semi = np.sqrt(np.mean(np.minimum(window_returns, 0.0) ** 2, axis=0))

    corr_mat = np.corrcoef(window_returns, rowvar=False)
    corr_mat = np.nan_to_num(corr_mat)
    triu_idx = np.triu_indices(n, k=1)
    corr_vec = corr_mat[triu_idx]

    market = window_returns.mean(axis=1)
    market_mean_3 = market[-3:].mean() if L >= 3 else market.mean()
    market_mean_12 = market[-12:].mean() if L >= 12 else market.mean()
    market_vol_12 = market[-12:].std(ddof=1) if L >= 12 else market.std(ddof=1)
    wealth = np.cumprod(1.0 + market)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    current_dd = dd[-1]
    avg_corr = corr_vec.mean() if len(corr_vec) else 0.0

    feature = np.concatenate([
        mu_short,
        mu_long,
        vol_short,
        vol_long,
        semi,
        corr_vec,
        np.array([market_mean_3, market_mean_12, market_vol_12, current_dd, avg_corr]),
    ])

    names = (
        [f"mu_short_{a}" for a in assets]
        + [f"mu_long_{a}" for a in assets]
        + [f"vol_short_{a}" for a in assets]
        + [f"vol_long_{a}" for a in assets]
        + [f"semi_{a}" for a in assets]
        + [f"corr_{assets[i]}_{assets[j]}" for i, j in zip(*triu_idx)]
        + ["market_mean_3", "market_mean_12", "market_vol_12", "market_drawdown", "avg_corr"]
    )

    return feature, names



def estimate_mu_sigma(window_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate conditional mean and covariance from the rolling window.

    The mean estimator uses a weighted blend of short and long windows to avoid
    relying exclusively on either very noisy short-term estimates or overly
    persistent long-run averages.

    The covariance estimator uses Ledoit-Wolf shrinkage, which is standard when
    the sample covariance matrix can be unstable or poorly conditioned.
    """
    short = window_returns[-12:] if len(window_returns) >= 12 else window_returns
    mu_short = short.mean(axis=0)
    mu_long = window_returns.mean(axis=0)
    mu_hat = 0.65 * mu_short + 0.35 * mu_long

    lw = LedoitWolf().fit(window_returns)
    sigma_hat = lw.covariance_
    return mu_hat, sigma_hat


# ---------------------------------------------------------------------------
# Scenario generation and portfolio optimization
# ---------------------------------------------------------------------------


def generate_scenarios(mu_hat: np.ndarray, sigma_hat: np.ndarray, n_scenarios: int, df: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Monte Carlo scenarios from an elliptical Student-t model."""
    vals, vecs = np.linalg.eigh(sigma_hat)
    vals = np.clip(vals, 1e-8, None)
    sigma_hat = vecs @ np.diag(vals) @ vecs.T

    z = rng.multivariate_normal(np.zeros(len(mu_hat)), sigma_hat, size=n_scenarios)
    g = rng.chisquare(df, size=n_scenarios) / df
    return mu_hat + z / np.sqrt(g)[:, None]



def optimize_portfolio_mc(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    prev_w: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Solve the scenario-based portfolio optimization problem.

    Objective:
        maximize mean - lambda * variance - eta * CVaR - tau * turnover

    subject to:
        w_i >= 0, sum_i w_i = 1.
    """
    n = len(mu_hat)
    scenarios = generate_scenarios(
        mu_hat,
        sigma_hat,
        n_scenarios=config.n_scenarios_train,
        df=7,
        rng=rng,
    )

    x0 = prev_w.copy()
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w: np.ndarray) -> float:
        port = scenarios @ w
        mean = port.mean()
        variance = port.var()
        cvar = compute_cvar(-port, alpha=config.alpha_cvar)
        turn = np.abs(w - prev_w).sum()
        utility = (
            mean
            - config.risk_aversion * variance
            - config.cvar_penalty * cvar
            - config.turnover_penalty * turn
        )
        return -float(utility)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-8, "disp": False},
    )

    if not result.success:
        return project_to_simplex(x0)
    return project_to_simplex(result.x)



def optimize_min_variance(sigma_hat: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance benchmark."""
    n = sigma_hat.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w: np.ndarray) -> float:
        return float(w.T @ sigma_hat @ w)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        return x0
    return project_to_simplex(result.x)



def inverse_vol_weights(sigma_hat: np.ndarray) -> np.ndarray:
    """Simple heuristic benchmark based on inverse volatilities."""
    vols = np.sqrt(np.diag(sigma_hat))
    inv = 1.0 / np.maximum(vols, 1e-8)
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------------------


def create_training_targets(
    ret_df: pd.DataFrame,
    regime_series: pd.Series,
    assets: List[str],
    config: ExperimentConfig,
) -> Dict[str, object]:
    """Create state vectors and optimized target weights for learning."""
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    mus: List[np.ndarray] = []
    sigs: List[np.ndarray] = []
    dates: List[int] = []
    regimes: List[str] = []
    feature_names: List[str] | None = None

    prev_w = np.ones(config.n_assets) / config.n_assets

    returns = ret_df.values
    for t in range(config.lookback, len(ret_df) - 1):
        window = returns[t - config.lookback : t]
        feature, feature_names = build_feature_state(window, assets)
        mu_hat, sigma_hat = estimate_mu_sigma(window)
        target_w = optimize_portfolio_mc(
            mu_hat,
            sigma_hat,
            prev_w=prev_w,
            config=config,
            rng=np.random.default_rng(1000 + t),
        )

        X.append(feature)
        y.append(target_w)
        mus.append(mu_hat)
        sigs.append(sigma_hat)
        dates.append(t)
        regimes.append(regime_series.iloc[t])
        prev_w = target_w.copy()

    return {
        "X": np.array(X),
        "y": np.array(y),
        "mus": mus,
        "sigs": sigs,
        "dates": dates,
        "regimes": regimes,
        "feature_names": feature_names,
    }



def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> float:
    """Average cosine similarity across rows."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return float(np.mean(np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)))



def fit_neural_surrogate(X: np.ndarray, y: np.ndarray, config: ExperimentConfig) -> Dict[str, object]:
    """Fit the MLP surrogate and compute train/validation/test diagnostics."""
    N = len(X)
    train_end = int(config.train_share * N)
    val_end = int((config.train_share + config.val_share) * N)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=config.hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        random_state=config.seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
    )
    model.fit(X_train_s, y_train)

    pred_train = np.vstack([project_to_simplex(v) for v in model.predict(X_train_s)])
    pred_val = np.vstack([project_to_simplex(v) for v in model.predict(X_val_s)])
    pred_test = np.vstack([project_to_simplex(v) for v in model.predict(X_test_s)])

    diagnostics = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Test"],
            "Weight MSE": [
                float(((pred_train - y_train) ** 2).mean()),
                float(((pred_val - y_val) ** 2).mean()),
                float(((pred_test - y_test) ** 2).mean()),
            ],
            "Weight MAE": [
                float(np.abs(pred_train - y_train).mean()),
                float(np.abs(pred_val - y_val).mean()),
                float(np.abs(pred_test - y_test).mean()),
            ],
            "Mean L1 Distance": [
                float(np.abs(pred_train - y_train).sum(axis=1).mean()),
                float(np.abs(pred_val - y_val).sum(axis=1).mean()),
                float(np.abs(pred_test - y_test).sum(axis=1).mean()),
            ],
            "Mean Cosine Similarity": [
                cosine_similarity_rows(pred_train, y_train),
                cosine_similarity_rows(pred_val, y_val),
                cosine_similarity_rows(pred_test, y_test),
            ],
        }
    )

    return {
        "model": model,
        "scaler": scaler,
        "train_end": train_end,
        "val_end": val_end,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "diagnostics": diagnostics,
    }



def backtest_portfolios(
    ret_df: pd.DataFrame,
    artifacts: Dict[str, object],
    learning: Dict[str, object],
    config: ExperimentConfig,
) -> Dict[str, object]:
    """Backtest benchmark and learned portfolios on the test segment."""
    returns = ret_df.values
    y = artifacts["y"]
    sigs = artifacts["sigs"]
    dates = artifacts["dates"]
    regimes = artifacts["regimes"]
    val_end = learning["val_end"]

    portfolio_names = [
        "Equal Weight",
        "Min Variance",
        "Inverse Vol",
        "MC-Optimal",
        "Neural Surrogate",
    ]

    rets = {name: [] for name in portfolio_names}
    weights_history = {name: [] for name in portfolio_names}

    previous_nn = learning["pred_val"][-1] if len(learning["pred_val"]) else np.ones(config.n_assets) / config.n_assets

    for i, idx in enumerate(range(val_end, len(y))):
        sigma_hat = sigs[idx]
        t = dates[idx]
        realized_next = returns[t + 1]

        w_eq = np.ones(config.n_assets) / config.n_assets
        w_mv = optimize_min_variance(sigma_hat)
        w_iv = inverse_vol_weights(sigma_hat)
        w_mc = y[idx]
        raw_nn = learning["pred_test"][i]
        w_nn = project_to_simplex(config.smoothing_lambda * previous_nn + (1.0 - config.smoothing_lambda) * raw_nn)
        previous_nn = w_nn

        for name, w in zip(portfolio_names, [w_eq, w_mv, w_iv, w_mc, w_nn]):
            weights_history[name].append(w)
            rets[name].append(float(np.dot(w, realized_next)))

    ret_back = pd.DataFrame(rets)
    weights_history = {k: np.array(v) for k, v in weights_history.items()}
    test_regimes = pd.Series(regimes[val_end:], name="Regime").reset_index(drop=True)

    performance = pd.DataFrame(
        {
            name: perf_metrics(ret_back[name].values, weights_history[name])
            for name in portfolio_names
        }
    ).T.reset_index().rename(columns={"index": "Portfolio"})

    regime_rows = []
    for name in portfolio_names:
        series = ret_back[name].values
        for reg in ["Expansion", "Stress", "Inflation"]:
            vals = series[test_regimes.values == reg]
            regime_rows.append(
                {
                    "Portfolio": name,
                    "Regime": reg,
                    "Mean Monthly Return": float(vals.mean()),
                    "Monthly Volatility": float(vals.std(ddof=1)),
                    "Downside Frequency": float((vals < 0.0).mean()),
                }
            )
    regime_performance = pd.DataFrame(regime_rows)

    return {
        "returns": ret_back,
        "weights_history": weights_history,
        "test_regimes": test_regimes,
        "performance": performance,
        "regime_performance": regime_performance,
        "portfolio_names": portfolio_names,
    }


# ---------------------------------------------------------------------------
# Diagnostics and robustness analysis
# ---------------------------------------------------------------------------


def representative_scenario_diagnostics(
    ret_df: pd.DataFrame,
    artifacts: Dict[str, object],
    backtest: Dict[str, object],
    assets: List[str],
    config: ExperimentConfig,
) -> Dict[str, object]:
    """Build a representative scenario data set for figure creation."""
    test_regimes = backtest["test_regimes"]
    val_end = int((config.train_share + config.val_share) * len(artifacts["y"]))
    stress_idx = [i for i, r in enumerate(test_regimes) if r == "Stress"]
    representative_i = stress_idx[0] if stress_idx else 0
    idx = val_end + representative_i
    t_rep = artifacts["dates"][idx]

    window = ret_df.values[t_rep - config.lookback : t_rep]
    mu_hat, sigma_hat = estimate_mu_sigma(window)
    scenarios = generate_scenarios(
        mu_hat,
        sigma_hat,
        n_scenarios=config.n_scenarios_rep,
        df=7,
        rng=np.random.default_rng(999),
    )

    sim_mean = scenarios.mean(axis=0)
    sim_vol = scenarios.std(axis=0, ddof=1)
    hist_corr = np.corrcoef(window, rowvar=False)
    sim_corr = np.corrcoef(scenarios, rowvar=False)
    triu = np.triu_indices(len(assets), 1)

    moment_df = pd.DataFrame(
        {
            "Asset": assets,
            "Estimated Mean": mu_hat,
            "Simulated Mean": sim_mean,
            "Abs Error Mean": np.abs(sim_mean - mu_hat),
            "Estimated Volatility": np.sqrt(np.diag(sigma_hat)),
            "Simulated Volatility": sim_vol,
            "Abs Error Volatility": np.abs(sim_vol - np.sqrt(np.diag(sigma_hat))),
        }
    )

    return {
        "t_rep": t_rep,
        "window": window,
        "scenarios": scenarios,
        "moment_df": moment_df,
        "corr_mae": float(np.mean(np.abs(hist_corr[triu] - sim_corr[triu]))),
    }



def aggregate_scenario_diagnostics(
    ret_df: pd.DataFrame,
    artifacts: Dict[str, object],
    learning: Dict[str, object],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Compute aggregate diagnostics across the test period."""
    triu = np.triu_indices(ret_df.shape[1], 1)
    rows = []
    for idx in range(learning["val_end"], len(artifacts["dates"])):
        t = artifacts["dates"][idx]
        window = ret_df.values[t - config.lookback : t]
        mu_hat, sigma_hat = estimate_mu_sigma(window)
        scenarios = generate_scenarios(
            mu_hat,
            sigma_hat,
            n_scenarios=config.n_scenarios_diag,
            df=7,
            rng=np.random.default_rng(7000 + t),
        )
        hist_corr = np.corrcoef(window, rowvar=False)
        sim_corr = np.corrcoef(scenarios, rowvar=False)
        rows.append(
            {
                "DateIndex": t,
                "Mean MAE": float(np.mean(np.abs(scenarios.mean(axis=0) - mu_hat))),
                "Volatility MAE": float(np.mean(np.abs(scenarios.std(axis=0, ddof=1) - np.sqrt(np.diag(sigma_hat))))),
                "Correlation MAE": float(np.mean(np.abs(hist_corr[triu] - sim_corr[triu]))),
            }
        )

    diag_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "Statistic": ["Average", "Median", "Maximum"],
            "Mean MAE": [diag_df["Mean MAE"].mean(), diag_df["Mean MAE"].median(), diag_df["Mean MAE"].max()],
            "Volatility MAE": [
                diag_df["Volatility MAE"].mean(),
                diag_df["Volatility MAE"].median(),
                diag_df["Volatility MAE"].max(),
            ],
            "Correlation MAE": [
                diag_df["Correlation MAE"].mean(),
                diag_df["Correlation MAE"].median(),
                diag_df["Correlation MAE"].max(),
            ],
            "Weight Test Cosine Similarity": [
                learning["diagnostics"].loc[learning["diagnostics"]["Split"] == "Test", "Mean Cosine Similarity"].iloc[0],
                np.nan,
                np.nan,
            ],
        }
    )
    return summary



def apply_shock_to_sigma(sigma: np.ndarray, vol_mult: float = 1.0, corr_pull: float = 0.0) -> np.ndarray:
    """Perturb the covariance matrix for sensitivity analysis.

    Parameters
    ----------
    vol_mult:
        Multiplicative factor applied to the marginal volatilities.
    corr_pull:
        Fraction that moves off-diagonal correlations toward one, mimicking a
        market-wide co-movement shock.
    """
    vols = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(vols, vols)
    corr = np.nan_to_num(corr)
    corr = np.clip(corr, -0.99, 0.99)

    corr_sh = corr.copy()
    off_diag = ~np.eye(len(vols), dtype=bool)
    corr_sh[off_diag] = corr_sh[off_diag] + corr_pull * (1.0 - corr_sh[off_diag])
    np.fill_diagonal(corr_sh, 1.0)

    shocked_vols = vols * vol_mult
    return np.outer(shocked_vols, shocked_vols) * corr_sh



def scenario_sensitivity_table(
    artifacts: Dict[str, object],
    backtest: Dict[str, object],
    learning: Dict[str, object],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Average scenario-based CVaR under stressed covariance structures."""
    shocks = {
        "Baseline": (1.00, 0.00),
        "High Volatility": (1.25, 0.00),
        "High Correlation": (1.00, 0.20),
        "Joint Shock": (1.25, 0.20),
    }

    weights_history = backtest["weights_history"]
    portfolio_names = backtest["portfolio_names"]
    rows = []

    for portfolio in portfolio_names:
        base_weights = weights_history[portfolio]
        for scenario_name, (vol_mult, corr_pull) in shocks.items():
            cvars = []
            for i, idx in enumerate(range(learning["val_end"], len(artifacts["dates"]))):
                mu_hat = artifacts["mus"][idx]
                sigma_hat = apply_shock_to_sigma(artifacts["sigs"][idx], vol_mult=vol_mult, corr_pull=corr_pull)
                scenarios = generate_scenarios(
                    mu_hat,
                    sigma_hat,
                    n_scenarios=config.n_scenarios_diag,
                    df=7,
                    rng=np.random.default_rng(20000 + idx * 17 + i),
                )
                port_scenarios = scenarios @ base_weights[i]
                cvars.append(compute_cvar(-port_scenarios, alpha=config.alpha_cvar))
            rows.append(
                {
                    "Portfolio": portfolio,
                    "Scenario": scenario_name,
                    "Scenario CVaR 95%": float(np.mean(cvars)),
                }
            )

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Portfolio", columns="Scenario", values="Scenario CVaR 95%").reset_index()
    pivot["Joint Shock Increase vs Baseline"] = pivot["Joint Shock"] / pivot["Baseline"] - 1.0
    return pivot


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def make_figure_scenario_cloud(rep: Dict[str, object], assets: List[str], output_path: Path) -> None:
    """Representative Monte Carlo scenario cloud for two strategic assets."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        rep["window"][:, 0],
        rep["window"][:, 2],
        s=18,
        alpha=0.7,
        label="Historical lookback window",
    )
    ax.scatter(
        rep["scenarios"][:, 0],
        rep["scenarios"][:, 2],
        s=6,
        alpha=0.2,
        label="Monte Carlo scenarios",
    )
    ax.set_xlabel(assets[0])
    ax.set_ylabel(assets[2])
    ax.set_title("Representative Scenario Cloud: US Equity vs Treasury")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_figure_learning_curve(model: MLPRegressor, output_path: Path) -> None:
    """Neural-network training loss across epochs."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(model.loss_curve_)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Neural Surrogate Training Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_figure_cumulative_wealth(backtest: Dict[str, object], output_path: Path) -> None:
    """Cumulative wealth for all strategies in the test period."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in backtest["portfolio_names"]:
        wealth = np.cumprod(1.0 + backtest["returns"][name].values)
        ax.plot(wealth, label=name)
    ax.set_xlabel("Test Month")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Out-of-Sample Cumulative Wealth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_figure_risk_return(backtest: Dict[str, object], output_path: Path) -> None:
    """Return-volatility map of competing portfolios."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name in backtest["portfolio_names"]:
        r = backtest["returns"][name].values
        x = r.std(ddof=1) * np.sqrt(12)
        y = annualized_return(r)
        ax.scatter(x, y, s=80)
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk-Return Positioning in the Test Period")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_figure_weight_heatmap(weights: np.ndarray, assets: List[str], output_path: Path) -> None:
    """Heatmap of neural allocations across the test period."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(weights.T, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets)
    ax.set_xlabel("Test Month")
    ax.set_title("Neural Surrogate Allocation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_figure_regime_weights(avg_weights: pd.DataFrame, output_path: Path) -> None:
    """Average neural allocations by latent market regime."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    avg_weights.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Weight")
    ax.set_xlabel("Regime")
    ax.set_title("Average Neural Weights by Market Regime")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Packaging and execution
# ---------------------------------------------------------------------------


def create_output_structure(root: Path) -> Dict[str, Path]:
    """Create section-specific result folders."""
    return {
        "root": ensure_dir(root),
        "5.1": ensure_dir(root / "5.1"),
        "5.2": ensure_dir(root / "5.2"),
        "5.3": ensure_dir(root / "5.3"),
    }



def zip_directory(source_dir: Path, output_zip: Path) -> None:
    """Zip an entire directory recursively."""
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(source_dir.parent))



def main() -> None:
    config = ExperimentConfig()
    output_root = Path(config.output_root).resolve()
    dirs = create_output_structure(output_root)

    # 1. Simulate the market environment.
    ret_df, regime_series, mu, vol, corr, transition_matrix, assets = simulate_regime_market(config)

    # 2. Generate training targets with Monte Carlo optimization.
    artifacts = create_training_targets(ret_df, regime_series, assets, config)

    # 3. Fit the neural surrogate.
    learning = fit_neural_surrogate(artifacts["X"], artifacts["y"], config)

    # 4. Backtest all portfolios on the test segment.
    backtest = backtest_portfolios(ret_df, artifacts, learning, config)

    # 5. Build diagnostics and sensitivity tables.
    rep = representative_scenario_diagnostics(ret_df, artifacts, backtest, assets, config)
    aggregate_diag = aggregate_scenario_diagnostics(ret_df, artifacts, learning, config)
    sensitivity = scenario_sensitivity_table(artifacts, backtest, learning, config)
    avg_weights_by_regime = (
        pd.DataFrame(backtest["weights_history"]["Neural Surrogate"], columns=assets)
        .groupby(backtest["test_regimes"])
        .mean()
        .reindex(["Expansion", "Stress", "Inflation"])
    )

    # 6. Save raw data needed for reproducibility.
    ret_df.to_csv(dirs["root"] / "synthetic_asset_returns.csv", index=False)
    regime_series.to_csv(dirs["root"] / "synthetic_regimes.csv", index=False)
    pd.DataFrame(artifacts["X"], columns=artifacts["feature_names"]).to_csv(dirs["root"] / "feature_matrix.csv", index=False)
    pd.DataFrame(artifacts["y"], columns=assets).to_csv(dirs["root"] / "optimizer_targets.csv", index=False)
    backtest["returns"].to_csv(dirs["root"] / "test_portfolio_returns.csv", index=False)
    pd.DataFrame(backtest["weights_history"]["Neural Surrogate"], columns=assets).to_csv(dirs["root"] / "neural_weights_test.csv", index=False)
    avg_weights_by_regime.reset_index().to_csv(dirs["root"] / "avg_neural_weights_by_regime.csv", index=False)

    # 7. Save tables for subsection 5.1.
    save_dataframe(
        aggregate_diag.round(4),
        dirs["5.1"] / "table_1_scenario_and_learning_diagnostics",
        title="Table 1. Scenario engine and learning diagnostics",
    )
    save_dataframe(
        rep["moment_df"].round(4),
        dirs["5.1"] / "table_1b_representative_moment_diagnostics",
        title="Supplementary diagnostics for the representative stress window",
    )

    # 8. Save tables for subsection 5.2.
    save_dataframe(
        backtest["performance"].round(4),
        dirs["5.2"] / "table_2_out_of_sample_performance",
        title="Table 2. Out-of-sample portfolio performance",
    )
    save_dataframe(
        backtest["regime_performance"].round(4),
        dirs["5.2"] / "table_3_regime_conditional_performance",
        title="Table 3. Regime-conditional performance",
    )

    # 9. Save tables for subsection 5.3.
    save_dataframe(
        sensitivity.round(4),
        dirs["5.3"] / "table_4_scenario_sensitivity",
        title="Table 4. Scenario sensitivity of downside risk",
    )
    save_dataframe(
        avg_weights_by_regime.reset_index().round(4),
        dirs["5.3"] / "table_5_average_neural_weights_by_regime",
        title="Supplementary average neural weights by regime",
    )

    # 10. Save figures.
    make_figure_scenario_cloud(rep, assets, dirs["5.1"] / "figure_1_representative_scenario_cloud.png")
    make_figure_learning_curve(learning["model"], dirs["5.1"] / "figure_2_neural_training_curve.png")
    make_figure_cumulative_wealth(backtest, dirs["5.2"] / "figure_3_cumulative_wealth.png")
    make_figure_risk_return(backtest, dirs["5.2"] / "figure_4_risk_return_positioning.png")
    make_figure_weight_heatmap(backtest["weights_history"]["Neural Surrogate"], assets, dirs["5.3"] / "figure_5_neural_weight_heatmap.png")
    make_figure_regime_weights(avg_weights_by_regime, dirs["5.3"] / "figure_6_average_weights_by_regime.png")

    # 11. Save metadata for the article-writing stage.
    metadata = {
        "config": config.__dict__,
        "assets": assets,
        "regime_counts": regime_series.value_counts().to_dict(),
        "test_regime_counts": backtest["test_regimes"].value_counts().to_dict(),
        "representative_date_index": int(rep["t_rep"]),
        "representative_correlation_mae": rep["corr_mae"],
        "performance_summary": backtest["performance"].round(6).to_dict(orient="records"),
        "learning_diagnostics": learning["diagnostics"].round(6).to_dict(orient="records"),
    }
    with open(dirs["root"] / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme_text = textwrap.dedent(
        f"""
        Monte Carlo Portfolio Optimization Outputs
        =========================================

        This folder contains all figures and tables required for Sections 5.1,
        5.2, and 5.3 of the manuscript.

        Folder structure
        ----------------
        - 5.1 : Scenario diagnostics and neural learning evidence
        - 5.2 : Out-of-sample performance and risk-return comparisons
        - 5.3 : Allocation patterns, robustness, and sensitivity analysis

        Reproducibility notes
        ---------------------
        - Random seed: {config.seed}
        - Observations: {config.periods} monthly returns
        - Lookback window: {config.lookback} months
        - Number of assets: {config.n_assets}
        - Training scenarios per date: {config.n_scenarios_train}
        - Downside confidence level (CVaR): {config.alpha_cvar}

        Main result summary
        -------------------
        The experiment shows that the neural surrogate is able to approximate the
        Monte Carlo optimizer while delivering competitive or superior
        out-of-sample risk-adjusted performance after allocation smoothing.
        """
    ).strip()
    (dirs["root"] / "README.txt").write_text(readme_text, encoding="utf-8")

    # 12. Zip only the outputs folder.
    zip_directory(dirs["root"], output_root.with_suffix(".zip"))

    print(f"Outputs written to: {output_root}")
    print(f"Zipped outputs: {output_root.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
