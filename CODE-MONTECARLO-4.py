#!/usr/bin/env python3
"""
insurance_hybrid_simulation.py

A complete and reproducible Python implementation of a hybrid Monte Carlo–neural network
framework for insurance risk evaluation.

The script performs the following tasks in a single workflow:

1. Generates a synthetic but actuarially plausible motor-insurance portfolio.
2. Builds policy-level claim frequency and severity parameters.
3. Simulates annual losses via Monte Carlo using a negative-binomial frequency model
   and a gamma + generalized Pareto severity mixture.
4. Constructs three supervised-learning targets from the simulation outputs:
      - pure premium,
      - multiclass risk profile,
      - extreme-loss indicator.
5. Trains classical baselines and neural-network models for all three tasks.
6. Exports tables and publication-ready figures into three subfolders:
      results/5.1
      results/5.2
      results/5.3
   so that the outputs can be inserted directly into the corresponding subsections
   of an academic manuscript.
7. Writes summary metrics as JSON and creates a ZIP archive of the full results.

The code is intentionally documented in detail because it is meant to support both
research reproducibility and later manuscript interpretation.

Author: OpenAI
"""

from __future__ import annotations

import json
import math
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import genpareto
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def ensure_output_folders(base_dir: Path) -> Tuple[Path, Path]:
    """
    Create the root output directory and the three results subdirectories expected
    by the manuscript structure.

    Parameters
    ----------
    base_dir : Path
        Base directory where the project outputs will be stored.

    Returns
    -------
    Tuple[Path, Path]
        The project root and the results directory.
    """
    project_root = base_dir
    results_dir = project_root / "results"
    for sub in ["5.1", "5.2", "5.3"]:
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    return project_root, results_dir


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute percentage error expressed as a percentage.
    A small guard is used to avoid division-by-zero problems if the target contains zeros.
    """
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def portfolio_risk_stats(aggregate_losses: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute portfolio-level tail risk measures.

    Returns
    -------
    (VaR95, VaR99, TVaR95, TVaR99)
    """
    q95 = float(np.quantile(aggregate_losses, 0.95))
    q99 = float(np.quantile(aggregate_losses, 0.99))
    t95 = float(aggregate_losses[aggregate_losses >= q95].mean())
    t99 = float(aggregate_losses[aggregate_losses >= q99].mean())
    return q95, q99, t95, t99


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an empirical cumulative distribution function.
    """
    x_sorted = np.sort(np.asarray(x))
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y


def save_df_as_png(df: pd.DataFrame, path: Path, title: str | None = None, font_size: int = 9) -> None:
    """
    Render a pandas DataFrame as a PNG image using matplotlib's table functionality.

    This is useful when the researcher wants both the original CSV file and a quick visual
    version of the same table, for example when inserting tables into manuscripts or slides.
    """
    nrows, ncols = df.shape
    fig_w = max(8, min(16, ncols * 1.7))
    fig_h = max(2.2, min(12, 0.55 * (nrows + 2)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index if not isinstance(df.index, pd.RangeIndex) else None,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    # Make header cells bold to improve readability.
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    if title:
        ax.set_title(title, pad=12)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Portfolio generation
# --------------------------------------------------------------------------------------

def generate_portfolio(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic portfolio designed to resemble heterogeneous motor-insurance data.

    The portfolio contains a mix of demographic, contract, exposure, and environment variables.
    The goal is not to replicate a particular insurer's book exactly, but to create a dataset
    rich enough to support nonlinear pricing and tail-risk dynamics.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 81, size=n)
    experience = np.clip(age - 18 - rng.integers(0, 12, size=n), 0, None)
    vehicle_age = rng.integers(0, 16, size=n)
    vehicle_value = np.round(np.exp(rng.normal(10.0, 0.35, size=n)), 2)
    mileage = np.round(np.exp(rng.normal(9.4, 0.45, size=n)))
    prior_claims = rng.poisson(0.4, size=n)
    exposure = rng.uniform(0.75, 1.0, size=n).round(3)

    region = rng.choice(
        ["North", "Center", "South", "Metro"],
        size=n,
        p=[0.22, 0.28, 0.23, 0.27],
    )
    credit_tier = rng.choice(
        ["A", "B", "C", "D"],
        size=n,
        p=[0.25, 0.35, 0.27, 0.13],
    )
    coverage = rng.choice(
        ["Basic", "Standard", "Premium"],
        size=n,
        p=[0.30, 0.45, 0.25],
    )
    deductible = rng.choice([250, 500, 1000], size=n, p=[0.30, 0.45, 0.25])
    urban = rng.binomial(1, 0.56, size=n)

    # Telematics is modeled as a bounded behavioral score.
    telematics = np.clip(
        rng.beta(5, 2, size=n) - 0.04 * prior_claims + 0.02 * (experience / 10) - 0.03 * urban,
        0,
        1,
    )

    climate_zone = rng.choice(["Low", "Medium", "High"], size=n, p=[0.40, 0.40, 0.20])

    return pd.DataFrame(
        {
            "Age": age,
            "Experience": experience,
            "VehicleAge": vehicle_age,
            "VehicleValue": vehicle_value,
            "AnnualMileage": mileage,
            "PriorClaims": prior_claims,
            "Exposure": exposure,
            "Region": region,
            "CreditTier": credit_tier,
            "Coverage": coverage,
            "Deductible": deductible.astype(str),
            "Urban": urban,
            "TelematicsScore": telematics.round(4),
            "ClimateZone": climate_zone,
        }
    )


# --------------------------------------------------------------------------------------
# Actuarial parameter construction
# --------------------------------------------------------------------------------------

def compute_params(df: pd.DataFrame, stress: bool = False) -> Dict[str, np.ndarray]:
    """
    Translate policy features into frequency and severity parameters.

    The parameterization is intentionally nonlinear. This is crucial because the research goal
    is to test whether a neural model can recover complex response surfaces from policy-level
    covariates after a stochastic simulation layer has transformed those covariates into
    actuarially meaningful outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Policy dataset.
    stress : bool
        If True, apply an adverse scenario that worsens frequency, tail probability,
        and tail scale for selected risk profiles.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing arrays of policy-level parameters.
    """
    age = df["Age"].to_numpy()
    exp = df["Experience"].to_numpy()
    vage = df["VehicleAge"].to_numpy()
    value = df["VehicleValue"].to_numpy()
    mileage = df["AnnualMileage"].to_numpy()
    prior = df["PriorClaims"].to_numpy()
    exposure = df["Exposure"].to_numpy()
    urban = df["Urban"].to_numpy()
    telem = df["TelematicsScore"].to_numpy()

    region = df["Region"].map({"North": 0.03, "Center": 0.00, "South": 0.07, "Metro": 0.12}).to_numpy()
    credit = df["CreditTier"].map({"A": -0.12, "B": 0.00, "C": 0.10, "D": 0.22}).to_numpy()
    coverage = df["Coverage"].map({"Basic": 0.00, "Standard": 0.08, "Premium": 0.18}).to_numpy()
    deductible = df["Deductible"].map({"250": 0.08, "500": 0.00, "1000": -0.10}).to_numpy()
    climate = df["ClimateZone"].map({"Low": 0.00, "Medium": 0.08, "High": 0.18}).to_numpy()

    # Several nonlinear and interaction terms are used to mimic realistic underwriting structure.
    age_term = ((age - 45) / 18) ** 2
    high_mileage = np.maximum(0, (mileage - 14000) / 7000)
    premium_flag = (df["Coverage"] == "Premium").to_numpy().astype(float)
    metro_flag = (df["Region"] == "Metro").to_numpy().astype(float)
    high_climate = (df["ClimateZone"] == "High").to_numpy().astype(float)
    low_credit = (df["CreditTier"].isin(["C", "D"])).to_numpy().astype(float)

    risk_score = (
        0.40 * age_term
        + 0.18 * urban
        + 0.22 * prior
        + 0.16 * high_mileage
        + 0.15 * vage / 10
        + 0.25 * high_climate
        + 0.14 * low_credit
        + 0.30 * premium_flag * metro_flag
        + 0.18 * high_mileage * (1 - telem)
        - 0.35 * telem
        - 0.12 * np.log1p(exp)
    )

    # Frequency: exposure-adjusted negative-binomial mean.
    eta_f = (
        -2.1
        + 0.22 * np.sin(age / 10)
        + 0.55 * risk_score
        + 0.15 * prior
        + region
        + credit
        + coverage
        + deductible
        + 0.09 * (urban * (mileage / 15000))
        - 0.18 * telem
    )
    lam = exposure * np.exp(eta_f)
    lam = np.clip(lam, 0.02, 2.0)
    theta = 1.8 + 2.2 * telem  # dispersion-like control

    # Severity body: gamma mean and shape.
    eta_s = (
        8.0
        + 0.22 * np.cos(age / 13)
        + 0.45 * risk_score
        + 0.000012 * value
        + 0.18 * coverage
        - 0.12 * deductible
        + 0.10 * region
        + 0.16 * climate
        + 0.14 * urban * (1 - telem)
    )
    mean_body = np.exp(eta_s)
    mean_body = np.clip(mean_body, 800, 12000)

    gamma_shape = 3.2 - 0.6 * np.clip(risk_score, -1.0, 2.0) / 2.0
    gamma_shape = np.clip(gamma_shape, 1.2, 4.5)

    # Tail channel: probability and scale of extreme claims.
    tail_prob = expit(
        -3.6 + 0.9 * risk_score + 0.55 * premium_flag + 0.35 * high_climate + 0.2 * urban - 0.35 * telem
    )
    tail_prob = np.clip(tail_prob, 0.005, 0.20)

    gpd_scale = 6000 * np.exp(0.35 * risk_score + 0.25 * high_climate + 0.10 * urban)
    gpd_scale = np.clip(gpd_scale, 1500, 45000)

    xi = np.clip(0.22 + 0.06 * high_climate + 0.03 * premium_flag + 0.02 * urban, 0.18, 0.45)

    # Adverse scenario: worsen frequency and tail behavior for exposed segments.
    if stress:
        lam = np.clip(lam * (1.10 + 0.08 * high_climate + 0.04 * urban + 0.03 * premium_flag), 0.02, 3.0)
        tail_prob = np.clip(tail_prob * (1.30 + 0.05 * high_climate + 0.04 * urban), 0.01, 0.35)
        gpd_scale = np.clip(gpd_scale * (1.22 + 0.06 * high_climate + 0.03 * premium_flag), 1500, 80000)
        mean_body = np.clip(mean_body * (1.06 + 0.04 * high_climate + 0.02 * urban), 800, 25000)

    return {
        "lam": lam,
        "theta": theta,
        "mean_body": mean_body,
        "gamma_shape": gamma_shape,
        "tail_prob": tail_prob,
        "gpd_scale": gpd_scale,
        "xi": xi,
        "risk_score": risk_score,
    }


# --------------------------------------------------------------------------------------
# Monte Carlo simulation
# --------------------------------------------------------------------------------------

def simulate_losses(
    params: Dict[str, np.ndarray],
    n_rep: int = 300,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate annual aggregate losses for each policy.

    For each policy:
      1. draw annual claim counts from a negative binomial model,
      2. simulate individual claim severities from a gamma body + GPD tail mixture,
      3. aggregate claim amounts within each Monte Carlo replication.

    Returns
    -------
    losses : np.ndarray
        Matrix of shape (n_policies, n_rep) containing annual aggregate losses.
    counts : np.ndarray
        Matrix of shape (n_policies, n_rep) containing annual claim counts.
    claim_count_sum : np.ndarray
        Total number of simulated claims across all replications for each policy.
    severity_mean : np.ndarray
        Mean simulated claim severity for each policy.
    """
    rng = np.random.default_rng(seed)
    n = len(params["lam"])

    losses = np.zeros((n, n_rep), dtype=np.float32)
    counts = np.zeros((n, n_rep), dtype=np.int16)
    claim_count_sum = np.zeros(n, dtype=np.int64)
    severity_mean = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lam = params["lam"][i]
        theta = params["theta"][i]

        # Convert the desired mean/dispersion parameterization to the version expected
        # by NumPy's negative_binomial sampler.
        p = theta / (theta + lam)
        annual_counts = rng.negative_binomial(theta, p, size=n_rep)

        counts[i] = annual_counts
        total_claims = int(annual_counts.sum())
        claim_count_sum[i] = total_claims

        if total_claims == 0:
            continue

        # Repeat replication indices so that individual simulated claim severities
        # can be added back to the proper Monte Carlo replication.
        rep_index = np.repeat(np.arange(n_rep), annual_counts)

        # Determine which claims fall in the heavy tail.
        tail_flags = rng.random(total_claims) < params["tail_prob"][i]

        # Body claims are drawn from a gamma model.
        severities = rng.gamma(
            shape=params["gamma_shape"][i],
            scale=params["mean_body"][i] / params["gamma_shape"][i],
            size=total_claims,
        )

        # Tail claims are replaced by threshold-excess GPD draws.
        if tail_flags.any():
            tail_size = int(tail_flags.sum())
            tail_claims = 7000 + genpareto.rvs(
                c=params["xi"][i],
                scale=params["gpd_scale"][i],
                size=tail_size,
                random_state=rng,
            )
            severities[tail_flags] = tail_claims

        np.add.at(losses[i], rep_index, severities.astype(np.float32))
        severity_mean[i] = float(severities.mean())

    return losses, counts, claim_count_sum, severity_mean


def sample_claim_severities(
    params: Dict[str, np.ndarray],
    sample_size: int = 120000,
    seed: int = 999,
) -> np.ndarray:
    """
    Draw a claim-level severity sample for visualization purposes.

    The sample is weighted by claim frequency so that policies with higher expected counts
    contribute proportionally more claim observations.
    """
    rng = np.random.default_rng(seed)
    weights = params["lam"] / params["lam"].sum()
    idx = rng.choice(len(weights), size=sample_size, replace=True, p=weights)

    tail_flags = rng.random(sample_size) < params["tail_prob"][idx]

    severities = rng.gamma(
        shape=params["gamma_shape"][idx],
        scale=params["mean_body"][idx] / params["gamma_shape"][idx],
        size=sample_size,
    )

    if tail_flags.any():
        tail_idx = np.where(tail_flags)[0]
        severities[tail_idx] = 7000 + genpareto.rvs(
            c=params["xi"][idx[tail_idx]],
            scale=params["gpd_scale"][idx[tail_idx]],
            size=len(tail_idx),
            random_state=rng,
        )

    return severities


def classify_risk_from_scores(pure_premium: np.ndarray, tail_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a composite risk score and translate it into Low / Moderate / High classes.

    The score mixes expected loss cost and tail probability so that segmentation remains
    informative for both routine pricing and adverse-loss surveillance.
    """
    positive_tail = tail_prob[tail_prob > 0]
    tail_scale = np.median(positive_tail) if len(positive_tail) > 0 else 1.0

    risk_composite = 0.7 * (pure_premium / np.median(pure_premium)) + 0.3 * (tail_prob / tail_scale)
    q1, q2 = np.quantile(risk_composite, [1 / 3, 2 / 3])

    risk_class = np.where(
        risk_composite <= q1,
        "Low",
        np.where(risk_composite <= q2, "Moderate", "High"),
    )
    return risk_composite, risk_class


# --------------------------------------------------------------------------------------
# Table and figure generation
# --------------------------------------------------------------------------------------

def create_tables_and_figures(
    portfolio: pd.DataFrame,
    losses: np.ndarray,
    counts: np.ndarray,
    base_params: Dict[str, np.ndarray],
    agg_base: np.ndarray,
    agg_stress: np.ndarray,
    y_true_premium: np.ndarray,
    y_pred_premium_nn: np.ndarray,
    le_risk: LabelEncoder,
    pred_c_nn: np.ndarray,
    y_c_test: np.ndarray,
    y_t_test: np.ndarray,
    prob_t_baseline: np.ndarray,
    prob_t_nn: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    results_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Create all tables and figures used by the manuscript and save them to disk.
    """
    # -----------------------------
    # Tables for Section 5.1
    # -----------------------------
    table1 = pd.DataFrame(
        {
            "Variable": ["Age", "Experience", "VehicleAge", "VehicleValue", "AnnualMileage", "PriorClaims", "Exposure", "TelematicsScore"],
            "Mean": [portfolio[c].mean() for c in ["Age", "Experience", "VehicleAge", "VehicleValue", "AnnualMileage", "PriorClaims", "Exposure", "TelematicsScore"]],
            "SD": [portfolio[c].std() for c in ["Age", "Experience", "VehicleAge", "VehicleValue", "AnnualMileage", "PriorClaims", "Exposure", "TelematicsScore"]],
            "Min": [portfolio[c].min() for c in ["Age", "Experience", "VehicleAge", "VehicleValue", "AnnualMileage", "PriorClaims", "Exposure", "TelematicsScore"]],
            "Max": [portfolio[c].max() for c in ["Age", "Experience", "VehicleAge", "VehicleValue", "AnnualMileage", "PriorClaims", "Exposure", "TelematicsScore"]],
        }
    ).round(2)

    table2 = (
        portfolio.groupby("RiskClass")
        .agg(
            Policies=("RiskClass", "size"),
            MeanFrequency=("MeanFrequencyMC", "mean"),
            MeanSeverity=("MeanSeverityMC", "mean"),
            PurePremium=("PurePremiumMC", "mean"),
            TailProbability=("TailProbMC", "mean"),
            VaR95=("VaR95", "mean"),
            TVaR95=("TVaR95", "mean"),
        )
        .reindex(["Low", "Moderate", "High"])
        .reset_index()
    )
    for column in ["MeanFrequency", "MeanSeverity", "PurePremium", "TailProbability", "VaR95", "TVaR95"]:
        table2[column] = table2[column].round(3 if column in ["MeanFrequency", "TailProbability"] else 2)

    # -----------------------------
    # Tables for Section 5.2
    # -----------------------------
    table3 = pd.DataFrame(
        [
            ["Tweedie GLM baseline", metrics["premium_baseline"]["RMSE"], metrics["premium_baseline"]["MAE"], metrics["premium_baseline"]["R2"], metrics["premium_baseline"]["MAPE_pct"]],
            ["Neural network", metrics["premium_nn"]["RMSE"], metrics["premium_nn"]["MAE"], metrics["premium_nn"]["R2"], metrics["premium_nn"]["MAPE_pct"]],
        ],
        columns=["Model", "RMSE", "MAE", "R2", "MAPE (%)"],
    ).round(3)

    table4 = pd.DataFrame(
        [
            ["Multinomial logistic baseline", metrics["risk_baseline"]["Accuracy"], metrics["risk_baseline"]["MacroF1"]],
            ["Neural network", metrics["risk_nn"]["Accuracy"], metrics["risk_nn"]["MacroF1"]],
        ],
        columns=["Model", "Accuracy", "Macro F1"],
    ).round(3)

    class_order = ["Low", "Moderate", "High"]
    label_to_index = {label: idx for idx, label in enumerate(le_risk.classes_)}
    order_idx = [label_to_index[c] for c in class_order]
    cm_risk_nn = confusion_matrix(y_c_test, pred_c_nn, labels=order_idx)

    table5 = pd.DataFrame(
        cm_risk_nn,
        index=[f"True {x}" for x in class_order],
        columns=[f"Pred {x}" for x in class_order],
    )

    # -----------------------------
    # Tables for Section 5.3
    # -----------------------------
    table6 = pd.DataFrame(
        [
            ["Logistic baseline", metrics["tail_baseline"]["Accuracy"], metrics["tail_baseline"]["F1"], metrics["tail_baseline"]["AUC"]],
            ["Neural network", metrics["tail_nn"]["Accuracy"], metrics["tail_nn"]["F1"], metrics["tail_nn"]["AUC"]],
        ],
        columns=["Model", "Accuracy", "F1-score", "AUC"],
    ).round(3)

    base_stats = portfolio_risk_stats(agg_base)
    stress_stats = portfolio_risk_stats(agg_stress)

    table7 = pd.DataFrame(
        {
            "Metric": ["Mean aggregate loss", "Std. dev. aggregate loss", "VaR 95%", "VaR 99%", "TVaR 95%", "TVaR 99%"],
            "Baseline": [agg_base.mean(), agg_base.std(ddof=1), base_stats[0], base_stats[1], base_stats[2], base_stats[3]],
            "Stress": [agg_stress.mean(), agg_stress.std(ddof=1), stress_stats[0], stress_stats[1], stress_stats[2], stress_stats[3]],
        }
    )
    table7["Increase_pct"] = (table7["Stress"] / table7["Baseline"] - 1) * 100
    table7 = table7.round(2)

    # Save CSV and PNG versions of all tables.
    table_map = {
        "5.1": [
            ("Table_1_portfolio_descriptive_statistics", table1),
            ("Table_2_monte_carlo_risk_indicators_by_risk_class", table2),
        ],
        "5.2": [
            ("Table_3_pure_premium_model_performance", table3),
            ("Table_4_risk_class_metrics", table4),
            ("Table_5_risk_class_confusion_matrix_nn", table5),
        ],
        "5.3": [
            ("Table_6_extreme_loss_classifier_metrics", table6),
            ("Table_7_portfolio_stress_test_metrics", table7),
        ],
    }

    for subfolder, items in table_map.items():
        for stem, df in items:
            csv_path = results_dir / subfolder / f"{stem}.csv"
            png_path = results_dir / subfolder / f"{stem}.png"
            df.to_csv(csv_path, index=stem.endswith("confusion_matrix_nn"))
            save_df_as_png(df, png_path, title=stem.replace("_", " "))

    # -----------------------------
    # Figures
    # -----------------------------

    # Figure 1: distribution of Monte Carlo mean frequency.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(portfolio["MeanFrequencyMC"], bins=35)
    ax.set_xlabel("Monte Carlo mean claim frequency per policy-year")
    ax.set_ylabel("Number of policies")
    ax.set_title("Figure 1. Distribution of Monte Carlo mean claim frequency")
    fig.tight_layout()
    fig.savefig(results_dir / "5.1" / "Figure_1_mean_frequency_distribution.png", dpi=220)
    plt.close(fig)

    # Figure 2: heavy-tailed severity distribution on log-x scale.
    severity_sample = sample_claim_severities(base_params, sample_size=120000, seed=999)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(severity_sample, bins=60)
    ax.set_xscale("log")
    ax.set_xlabel("Claim severity (log scale)")
    ax.set_ylabel("Number of sampled claims")
    ax.set_title("Figure 2. Sampled claim severity distribution with heavy tail")
    fig.tight_layout()
    fig.savefig(results_dir / "5.1" / "Figure_2_claim_severity_distribution.png", dpi=220)
    plt.close(fig)

    # Figure 3: baseline vs stress aggregate-loss histogram.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(agg_base, bins=30, alpha=0.7, label="Baseline")
    ax.hist(agg_stress, bins=30, alpha=0.7, label="Stress")
    ax.set_xlabel("Aggregate portfolio loss per simulation")
    ax.set_ylabel("Frequency")
    ax.set_title("Figure 3. Aggregate loss distribution: baseline vs stress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "5.1" / "Figure_3_aggregate_loss_baseline_vs_stress.png", dpi=220)
    plt.close(fig)

    # Figure 4: premium scatterplot.
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(y_true_premium, y_pred_premium_nn, alpha=0.35, s=18)
    diag_min = min(y_true_premium.min(), y_pred_premium_nn.min())
    diag_max = max(y_true_premium.max(), y_pred_premium_nn.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], linewidth=1.5)
    ax.set_xlabel("Observed Monte Carlo pure premium")
    ax.set_ylabel("Neural-network predicted pure premium")
    ax.set_title("Figure 4. Observed vs predicted pure premium")
    fig.tight_layout()
    fig.savefig(results_dir / "5.2" / "Figure_4_actual_vs_predicted_pure_premium.png", dpi=220)
    plt.close(fig)

    # Figure 5: risk-class confusion matrix.
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm_risk_nn)
    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(class_order)
    ax.set_yticklabels(class_order)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Figure 5. Neural-network confusion matrix for risk class")
    for i in range(cm_risk_nn.shape[0]):
        for j in range(cm_risk_nn.shape[1]):
            ax.text(j, i, int(cm_risk_nn[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(results_dir / "5.2" / "Figure_5_risk_class_confusion_matrix.png", dpi=220)
    plt.close(fig)

    # Figure 6: ROC curves for extreme-loss detection.
    fpr_b, tpr_b, _ = roc_curve(y_t_test, prob_t_baseline)
    fpr_n, tpr_n, _ = roc_curve(y_t_test, prob_t_nn)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(fpr_b, tpr_b, label=f'Baseline logistic (AUC = {metrics["tail_baseline"]["AUC"]:.3f})')
    ax.plot(fpr_n, tpr_n, label=f'Neural network (AUC = {metrics["tail_nn"]["AUC"]:.3f})')
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 6. ROC curves for extreme-loss anticipation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "5.3" / "Figure_6_roc_extreme_loss_classifier.png", dpi=220)
    plt.close(fig)

    # Figure 7: baseline vs stress ECDF.
    xb, yb = ecdf(agg_base)
    xs, ys = ecdf(agg_stress)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xb, yb, label="Baseline")
    ax.plot(xs, ys, label="Stress")
    ax.set_xlabel("Aggregate portfolio loss")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Figure 7. Empirical distribution of aggregate losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "5.3" / "Figure_7_ecdf_aggregate_losses.png", dpi=220)
    plt.close(fig)

    return {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "table5": table5,
        "table6": table6,
        "table7": table7,
    }


# --------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------

def main(output_root: str = "/mnt/data/insurance_hybrid_article") -> None:
    """
    Execute the full hybrid Monte Carlo–neural network workflow and save all outputs.
    """
    project_root, results_dir = ensure_output_folders(Path(output_root))

    # Step 1. Create the synthetic policy portfolio.
    portfolio_raw = generate_portfolio(n=4000, seed=42)

    # Step 2. Build base and stressed risk parameters.
    base_params = compute_params(portfolio_raw, stress=False)
    stress_params = compute_params(portfolio_raw, stress=True)

    # Step 3. Simulate annual losses under both scenarios.
    losses, counts, claim_count_sum, _severity_mean = simulate_losses(base_params, n_rep=300, seed=123)
    losses_stress, _counts_stress, _claim_count_sum_stress, _severity_mean_stress = simulate_losses(stress_params, n_rep=300, seed=321)

    # Step 4. Construct Monte Carlo targets.
    threshold_policy = float(np.quantile(losses.ravel(), 0.99))
    pure_premium = losses.mean(axis=1)
    mean_freq_mc = counts.mean(axis=1)
    mean_sev_mc = np.divide(losses.sum(axis=1), counts.sum(axis=1), out=np.zeros(losses.shape[0]), where=counts.sum(axis=1) > 0)
    tail_prob_mc = (losses > threshold_policy).mean(axis=1)
    var95_policy = np.quantile(losses, 0.95, axis=1)
    tvar95_policy = np.array(
        [
            row[row >= np.quantile(row, 0.95)].mean() if (row >= np.quantile(row, 0.95)).any() else 0
            for row in losses
        ],
        dtype=float,
    )
    risk_composite, risk_class = classify_risk_from_scores(pure_premium, tail_prob_mc)
    tail_label = (tail_prob_mc >= np.quantile(tail_prob_mc, 0.8)).astype(int)

    portfolio = portfolio_raw.copy()
    portfolio["PurePremiumMC"] = pure_premium
    portfolio["MeanFrequencyMC"] = mean_freq_mc
    portfolio["MeanSeverityMC"] = mean_sev_mc
    portfolio["TailProbMC"] = tail_prob_mc
    portfolio["VaR95"] = var95_policy
    portfolio["TVaR95"] = tvar95_policy
    portfolio["RiskComposite"] = risk_composite
    portfolio["RiskClass"] = risk_class
    portfolio["TailLabel"] = tail_label

    # Step 5. Prepare train/test sets.
    features = [
        "Age",
        "Experience",
        "VehicleAge",
        "VehicleValue",
        "AnnualMileage",
        "PriorClaims",
        "Exposure",
        "Region",
        "CreditTier",
        "Coverage",
        "Deductible",
        "Urban",
        "TelematicsScore",
        "ClimateZone",
    ]
    X = portfolio[features]
    y_p = np.log1p(portfolio["PurePremiumMC"].to_numpy())
    y_c = portfolio["RiskClass"].to_numpy()
    y_t = portfolio["TailLabel"].to_numpy()

    le_risk = LabelEncoder()
    y_c_num = le_risk.fit_transform(y_c)

    X_train, X_test, y_p_train, y_p_test, y_c_train, y_c_test, y_t_train, y_t_test = train_test_split(
        X,
        y_p,
        y_c_num,
        y_t,
        test_size=0.25,
        random_state=7,
        stratify=y_c_num,
    )

    num_features = ["Age", "Experience", "VehicleAge", "VehicleValue", "AnnualMileage", "PriorClaims", "Exposure", "Urban", "TelematicsScore"]
    cat_features = ["Region", "CreditTier", "Coverage", "Deductible", "ClimateZone"]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("scaler", StandardScaler())]), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    # Step 6. Define baselines and neural models.
    premium_baseline = Pipeline(
        [
            ("pre", preprocessor),
            ("model", TweedieRegressor(power=1.5, alpha=0.3, link="log", max_iter=1000)),
        ]
    )

    premium_nn = Pipeline(
        [
            ("pre", preprocessor),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.0008,
                    learning_rate_init=0.005,
                    max_iter=500,
                    early_stopping=True,
                    random_state=7,
                ),
            ),
        ]
    )

    risk_baseline = Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=3000, solver="lbfgs"))])
    risk_nn = Pipeline(
        [
            ("pre", preprocessor),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.0008,
                    learning_rate_init=0.005,
                    max_iter=500,
                    early_stopping=False,
                    random_state=7,
                ),
            ),
        ]
    )

    tail_baseline = Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=3000, solver="lbfgs"))])
    tail_nn = Pipeline(
        [
            ("pre", preprocessor),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.0008,
                    learning_rate_init=0.005,
                    max_iter=500,
                    early_stopping=False,
                    random_state=7,
                ),
            ),
        ]
    )

    # Step 7. Fit all models.
    premium_baseline.fit(X_train, np.expm1(y_p_train))
    premium_nn.fit(X_train, y_p_train)
    risk_baseline.fit(X_train, y_c_train)
    risk_nn.fit(X_train, y_c_train)
    tail_baseline.fit(X_train, y_t_train)
    tail_nn.fit(X_train, y_t_train)

    # Step 8. Generate predictions.
    pred_p_baseline = premium_baseline.predict(X_test)
    pred_p_nn = np.expm1(premium_nn.predict(X_test))
    y_p_true = np.expm1(y_p_test)

    pred_c_baseline = risk_baseline.predict(X_test)
    pred_c_nn = risk_nn.predict(X_test)

    prob_t_baseline = tail_baseline.predict_proba(X_test)[:, 1]
    pred_t_baseline = (prob_t_baseline >= 0.5).astype(int)

    prob_t_nn = tail_nn.predict_proba(X_test)[:, 1]
    pred_t_nn = (prob_t_nn >= 0.5).astype(int)

    # Step 9. Collect metrics.
    metrics = {
        "premium_baseline": {
            "RMSE": float(math.sqrt(mean_squared_error(y_p_true, pred_p_baseline))),
            "MAE": float(mean_absolute_error(y_p_true, pred_p_baseline)),
            "R2": float(r2_score(y_p_true, pred_p_baseline)),
            "MAPE_pct": float(mape(y_p_true, pred_p_baseline)),
        },
        "premium_nn": {
            "RMSE": float(math.sqrt(mean_squared_error(y_p_true, pred_p_nn))),
            "MAE": float(mean_absolute_error(y_p_true, pred_p_nn)),
            "R2": float(r2_score(y_p_true, pred_p_nn)),
            "MAPE_pct": float(mape(y_p_true, pred_p_nn)),
        },
        "risk_baseline": {
            "Accuracy": float(accuracy_score(y_c_test, pred_c_baseline)),
            "MacroF1": float(f1_score(y_c_test, pred_c_baseline, average="macro")),
        },
        "risk_nn": {
            "Accuracy": float(accuracy_score(y_c_test, pred_c_nn)),
            "MacroF1": float(f1_score(y_c_test, pred_c_nn, average="macro")),
        },
        "tail_baseline": {
            "Accuracy": float(accuracy_score(y_t_test, pred_t_baseline)),
            "F1": float(f1_score(y_t_test, pred_t_baseline)),
            "AUC": float(roc_auc_score(y_t_test, prob_t_baseline)),
        },
        "tail_nn": {
            "Accuracy": float(accuracy_score(y_t_test, pred_t_nn)),
            "F1": float(f1_score(y_t_test, pred_t_nn)),
            "AUC": float(roc_auc_score(y_t_test, prob_t_nn)),
        },
    }

    # Step 10. Portfolio-level aggregate loss metrics.
    agg_base = losses.sum(axis=0)
    agg_stress = losses_stress.sum(axis=0)

    summary = {
        "n_policies": int(len(portfolio)),
        "n_simulations": int(losses.shape[1]),
        "overall_mean_frequency": float(portfolio["MeanFrequencyMC"].mean()),
        "overall_mean_severity": float(portfolio["MeanSeverityMC"][portfolio["MeanSeverityMC"] > 0].mean()),
        "overall_mean_pure_premium": float(portfolio["PurePremiumMC"].mean()),
        "threshold_policy_99": threshold_policy,
        "tail_label_positive_rate": float(portfolio["TailLabel"].mean()),
        "premium_baseline": metrics["premium_baseline"],
        "premium_nn": metrics["premium_nn"],
        "risk_baseline": metrics["risk_baseline"],
        "risk_nn": metrics["risk_nn"],
        "tail_baseline": metrics["tail_baseline"],
        "tail_nn": metrics["tail_nn"],
        "aggregate_baseline": {
            "mean": float(agg_base.mean()),
            "std": float(agg_base.std(ddof=1)),
            "VaR95": float(portfolio_risk_stats(agg_base)[0]),
            "VaR99": float(portfolio_risk_stats(agg_base)[1]),
            "TVaR95": float(portfolio_risk_stats(agg_base)[2]),
            "TVaR99": float(portfolio_risk_stats(agg_base)[3]),
        },
        "aggregate_stress": {
            "mean": float(agg_stress.mean()),
            "std": float(agg_stress.std(ddof=1)),
            "VaR95": float(portfolio_risk_stats(agg_stress)[0]),
            "VaR99": float(portfolio_risk_stats(agg_stress)[1]),
            "TVaR95": float(portfolio_risk_stats(agg_stress)[2]),
            "TVaR99": float(portfolio_risk_stats(agg_stress)[3]),
        },
    }

    # Step 11. Export intermediate core data for reproducibility.
    portfolio.to_csv(project_root / "synthetic_portfolio_with_targets.csv", index=False)
    with open(project_root / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Step 12. Create tables and figures.
    create_tables_and_figures(
        portfolio=portfolio,
        losses=losses,
        counts=counts,
        base_params=base_params,
        agg_base=agg_base,
        agg_stress=agg_stress,
        y_true_premium=y_p_true,
        y_pred_premium_nn=pred_p_nn,
        le_risk=le_risk,
        pred_c_nn=pred_c_nn,
        y_c_test=y_c_test,
        y_t_test=y_t_test,
        prob_t_baseline=prob_t_baseline,
        prob_t_nn=prob_t_nn,
        metrics=metrics,
        results_dir=results_dir,
    )

    # Step 13. Package a ZIP archive for convenient transport.
    zip_path = project_root / "insurance_hybrid_results_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in project_root.rglob("*"):
            if file_path == zip_path:
                continue
            archive.write(file_path, arcname=file_path.relative_to(project_root))

    print(f"Project written to: {project_root}")
    print(f"ZIP archive written to: {zip_path}")


if __name__ == "__main__":
    main()
