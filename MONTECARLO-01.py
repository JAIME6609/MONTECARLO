# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 21:41:01 2025

@author: Asus.S510UNR
"""

# ============================================================
# MÉTODO MONTE CARLO PARA ESTIMAR π (CON VARIAS VARIANTES)
# ============================================================
# Métodos:
#   1) Naïve Monte Carlo (dardos en cuadrado)
#   2) Muestreo Estratificado / Latin Hypercube (LHS)
#   3) Control Variate (usando R²)
#   4) Buffon’s Needle
#
# El script genera:
#   - Tablas con errores, IC, coberturas, RMSE, sesgo, SD
#   - Gráficas de convergencia, RMSE vs N, histogramas, boxplots, cobertura
#   - Archivo Excel con todas las métricas
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List

# Configuración inicial
RNG = np.random.default_rng(20251022)
OUT_DIR = Path("./resultados_montecarlo_pi")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PI_TRUE = math.pi

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def ci_normal(mean: float, var: float, n: int, alpha: float = 0.05):
    """Intervalo de confianza (IC) normal aproximado (95%)"""
    z = 1.959963984540054
    se = math.sqrt(var)
    return mean - z * se, mean + z * se

# ============================================================
# ESTIMADORES
# ============================================================

# ---- 1. Estimador Naïve (dardos en cuadrado)
def estimate_pi_naive(n: int, rng=RNG):
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    inside = (xy[:, 0]**2 + xy[:, 1]**2) <= 1.0
    p_hat = inside.mean()
    pi_hat = 4.0 * p_hat
    var_est = (16.0 / n) * p_hat * (1.0 - p_hat)
    return pi_hat, var_est

# ---- 2. Estratificado / Latin Hypercube
def estimate_pi_stratified(n: int, rng=RNG):
    m = int(math.sqrt(n))
    n_eff = m*m
    xs = (np.arange(m) + rng.uniform(0, 1, size=m)) / m
    ys = (np.arange(m) + rng.uniform(0, 1, size=m)) / m
    rng.shuffle(xs); rng.shuffle(ys)
    xs_rep = np.repeat(xs, m)
    ys_perm = np.tile(ys, m)
    inside = (xs_rep**2 + ys_perm**2) <= 1.0
    p_hat = inside.mean()
    pi_hat = 4.0 * p_hat
    var_est = (16.0 / n_eff) * p_hat * (1.0 - p_hat)
    return pi_hat, var_est

# ---- 3. Control Variate (R² con media conocida 2/3)
def estimate_pi_control_variates(n: int, rng=RNG):
    xy = rng.uniform(0.0, 1.0, size=(n, 2))
    r2 = xy[:, 0]**2 + xy[:, 1]**2
    I = (r2 <= 1.0).astype(float)
    p_hat = I.mean()
    r2_bar = r2.mean()
    mu_r2 = 2.0 / 3.0
    cov = np.cov(I, r2, bias=True)[0, 1]
    var_r2 = np.var(r2)
    beta = 0.0 if var_r2 == 0 else cov / var_r2
    p_cv = p_hat - beta * (r2_bar - mu_r2)
    pi_hat = 4.0 * p_cv
    adj_terms = I - beta * (r2 - mu_r2)
    var_est = 16.0 * np.var(adj_terms, ddof=1) / n
    return pi_hat, var_est

# ---- 4. Buffon’s Needle
def estimate_pi_buffon(n: int, L: float = 1.0, D: float = 1.0, rng=RNG):
    U = rng.uniform(0.0, D/2.0, size=n)
    Theta = rng.uniform(0.0, math.pi/2.0, size=n)
    cross = (U <= (L/2.0) * np.sin(Theta))
    p_hat = cross.mean()
    eps = 1e-12
    pi_hat = (2.0 * L) / (D * max(p_hat, eps))
    if p_hat > 0:
        deriv = (2.0 * L / D) * (1.0 / (p_hat**2))
        var_p = p_hat * (1.0 - p_hat) / n
        var_est = (deriv**2) * var_p
    else:
        var_est = float("inf")
    return pi_hat, var_est

# ============================================================
# EXPERIMENTOS Y TABLAS
# ============================================================

@dataclass
class EstimationResult:
    method: str; N: int; pi_hat: float; abs_error: float
    ci_low: float; ci_high: float; ci_halfwidth: float; covered: bool; var_hat: float

def single_run_summary(Ns: List[int], rng=RNG) -> pd.DataFrame:
    rows = []
    for N in Ns:
        phat, varh = estimate_pi_naive(N, rng=rng)
        ci_l, ci_h = ci_normal(phat, varh, N)
        rows.append(EstimationResult("Naive", N, phat, abs(phat-PI_TRUE),
                                     ci_l, ci_h, (ci_h-ci_l)/2.0, (ci_l<=PI_TRUE<=ci_h), varh))
        m = int(math.sqrt(N))
        phat, varh = estimate_pi_stratified(m*m, rng=rng)
        ci_l, ci_h = ci_normal(phat, varh, m*m)
        rows.append(EstimationResult("Stratified(LHS)", m*m, phat, abs(phat-PI_TRUE),
                                     ci_l, ci_h, (ci_h-ci_l)/2.0, (ci_l<=PI_TRUE<=ci_h), varh))
        phat, varh = estimate_pi_control_variates(N, rng=rng)
        ci_l, ci_h = ci_normal(phat, varh, N)
        rows.append(EstimationResult("ControlVariate(R2)", N, phat, abs(phat-PI_TRUE),
                                     ci_l, ci_h, (ci_h-ci_l)/2.0, (ci_l<=PI_TRUE<=ci_h), varh))
        phat, varh = estimate_pi_buffon(N, L=1.0, D=1.0, rng=rng)
        ci_l, ci_h = ci_normal(phat, varh, N)
        rows.append(EstimationResult("Buffon", N, phat, abs(phat-PI_TRUE),
                                     ci_l, ci_h, (ci_h-ci_l)/2.0, (ci_l<=PI_TRUE<=ci_h), varh))
    return pd.DataFrame([r.__dict__ for r in rows])

# ============================================================
# FUNCIONES DE ANÁLISIS
# ============================================================

def rmse_vs_N(method_fn_map: Dict[str, callable], Ns: List[int], R: int = 200, rng=RNG) -> pd.DataFrame:
    results = []
    for method, fn in method_fn_map.items():
        for N in Ns:
            ests = [fn(N, rng=rng)[0] for _ in range(R)]
            ests = np.array(ests)
            rmse = math.sqrt(np.mean((ests - PI_TRUE)**2))
            bias = float(np.mean(ests) - PI_TRUE)
            std = float(np.std(ests, ddof=1))
            results.append({"method": method, "N": N, "RMSE": rmse, "Bias": bias, "SD": std})
    return pd.DataFrame(results)

def replicate_distributions(method_fn_map: Dict[str, callable], N: int, R: int = 500, rng=RNG) -> pd.DataFrame:
    rows = []
    for method, fn in method_fn_map.items():
        for _ in range(R):
            phat, _ = fn(N, rng=rng)
            rows.append({"method": method, "N": N, "pi_hat": phat, "abs_error": abs(phat-PI_TRUE)})
    return pd.DataFrame(rows)

def summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["method", "N"])
    rows = []
    for m, grp in df_sorted.groupby("method"):
        row = grp.iloc[-1]
        rows.append({
            "method": m, "N": int(row["N"]), "pi_hat": row["pi_hat"],
            "abs_error": row["abs_error"], "ci_halfwidth": row["ci_halfwidth"],
            "covered": bool(row["covered"])
        })
    return pd.DataFrame(rows)

# ============================================================
# EJECUCIÓN DE EXPERIMENTOS
# ============================================================

Ns_single = [100, 300, 1000, 3000, 10000, 30000, 100000]
df_single = single_run_summary(Ns_single, rng=RNG)

method_fn_map = {
    "Naive": estimate_pi_naive,
    "Stratified(LHS)": lambda n, rng: estimate_pi_stratified(max(1, int(math.sqrt(n)))**2, rng=rng),
    "ControlVariate(R2)": estimate_pi_control_variates,
    "Buffon": estimate_pi_buffon,
}

Ns_rmse = [500, 1000, 2000, 5000, 10000]
df_rmse = rmse_vs_N(method_fn_map, Ns_rmse, R=200, rng=RNG)
df_reps = replicate_distributions(method_fn_map, N=5000, R=500, rng=RNG)
df_summary_method = summary_by_method(df_single)

# ============================================================
# GUARDAR RESULTADOS EN EXCEL
# ============================================================

excel_path = OUT_DIR / "MonteCarlo_Pi_Results.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_single.to_excel(writer, sheet_name="SingleRun_Summary", index=False)
    df_rmse.to_excel(writer, sheet_name="RMSE_vs_N", index=False)
    df_reps.to_excel(writer, sheet_name="Replicate_Distributions", index=False)
    df_summary_method.to_excel(writer, sheet_name="Summary_By_Method", index=False)

# ============================================================
# GRÁFICAS
# ============================================================

def running_convergence_plot(N: int = 50000, rng=RNG, save_path: Path = OUT_DIR / "Fig_Convergence_Naive.png"):
    xy = rng.uniform(-1.0, 1.0, size=(N, 2))
    inside = (xy[:, 0]**2 + xy[:, 1]**2) <= 1.0
    cum = np.cumsum(inside.astype(float))
    idx = np.arange(1, N+1)
    p_hat_seq = cum / idx
    pi_hat_seq = 4.0 * p_hat_seq
    var_seq = (16.0 / idx) * p_hat_seq * (1.0 - p_hat_seq)
    z = 1.959963984540054
    halfwidth = z * np.sqrt(var_seq)
    plt.figure(figsize=(8, 5))
    plt.plot(idx, pi_hat_seq, label="Estimador Naïve (corrida única)")
    plt.axhline(PI_TRUE, linestyle="--", label="π real")
    plt.plot(idx, pi_hat_seq - halfwidth, linewidth=1.0, label="Límite inferior 95%")
    plt.plot(idx, pi_hat_seq + halfwidth, linewidth=1.0, label="Límite superior 95%")
    plt.xlabel("n (muestras acumuladas)")
    plt.ylabel("Estimación de π")
    plt.title("Convergencia por corrida única (Naïve)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def histograms_by_method(df: pd.DataFrame, save_prefix: Path = OUT_DIR / "Fig_Hist_"):
    methods = df["method"].unique()
    for m in methods:
        sub = df[df["method"] == m]
        plt.figure(figsize=(7, 5))
        plt.hist(sub["pi_hat"].values, bins=25, density=True)
        plt.axvline(PI_TRUE, linestyle="--")
        plt.xlabel("Estimación de π")
        plt.ylabel("Densidad")
        plt.title(f"Distribución de estimaciones (N={int(sub['N'].iloc[0])}) – {m}")
        plt.tight_layout()
        fname = f"{save_prefix}{m.replace(' ','_').replace('(','').replace(')','')}.png"
        plt.savefig(fname)
        plt.close()

def rmse_plot(df: pd.DataFrame, save_path: Path = OUT_DIR / "Fig_RMSE_loglog.png"):
    plt.figure(figsize=(7, 5))
    for m in df["method"].unique():
        sub = df[df["method"] == m].sort_values("N")
        plt.loglog(sub["N"].values, sub["RMSE"].values, marker="o", label=m)
    plt.xlabel("N (escala log)")
    plt.ylabel("RMSE (escala log)")
    plt.title("Decaimiento del RMSE vs N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def boxplot_abs_errors(df: pd.DataFrame, save_path: Path = OUT_DIR / "Fig_Box_AbsError.png"):
    methods = df["method"].unique()
    data = [df[df["method"] == m]["abs_error"].values for m in methods]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=list(methods), showmeans=True)
    plt.ylabel("|Error|")
    plt.title("Errores absolutos por método (N fijo)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def coverage_plot(df: pd.DataFrame, save_path: Path = OUT_DIR / "Fig_Coverage_Single.png"):
    y_map = {"Naive": 3, "Stratified(LHS)": 2, "ControlVariate(R2)": 1, "Buffon": 0}
    plt.figure(figsize=(7, 5))
    for _, row in df.iterrows():
        y = y_map.get(row["method"], -1)
        if y >= 0:
            plt.scatter(row["N"], y, marker="o")
            plt.plot([row["ci_low"], row["ci_high"]], [y, y])
    plt.axvline(PI_TRUE, linestyle=":")
    plt.yticks(list(y_map.values()), list(y_map.keys()))
    plt.xlabel("Intervalo de confianza (horizontal) y N (puntos discretos)")
    plt.title("Cobertura puntual de IC (corridas únicas por N y método)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Generar todas las figuras
running_convergence_plot()
histograms_by_method(df_reps)
rmse_plot(df_rmse)
boxplot_abs_errors(df_reps)
coverage_plot(df_single)

print(f"\nResultados guardados en: {OUT_DIR.resolve()}")
