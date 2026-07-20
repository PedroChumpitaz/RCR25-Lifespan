#!/usr/bin/env python3
"""
Reproduce the appliance-lifespan estimates reported for Peru's ENAHO survey,
2008–2024.

The script performs two tasks:

1. Fits a survey-weighted two-component Weibull mixture for each appliance-year.
2. Computes survey-weighted appliance ownership percentages.

The computational convention is:
- component 1: shorter-lived component, interpreted as second use;
- component 2: longer-lived component, interpreted as first use;
- internal pi: weight of component 2 (first use);
- reported second-use proportion: 1 - pi.

The standard fit uses the historical deterministic initialization. If its
component optimizer fails, the case is refitted with the recovered robust
implementation.

Run from the repository root:

    python src/estimate_lifespans.py

Outputs are written to reproduced/ by default. Existing reported results in
results/ are never overwritten.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar
from scipy.special import gamma


YEARS = tuple(range(2008, 2025))

ARTIFACTS: Dict[int, str] = {
    1: "Radio",
    2: "Color TV",
    4: "Sound System",
    5: "DVD",
    6: "VCR",
    7: "PC",
    8: "Iron",
    9: "Blender",
    10: "Gas Stove",
    12: "Refrigerator",
    13: "Washer",
    14: "Microwave Oven",
}

REFERENCE_LIFETIMES: Dict[int, float] = {
    1: 10.000,
    2: 12.080,
    4: 10.000,
    5: 8.330,
    6: 8.330,
    7: 9.545,
    8: 7.970,
    9: 7.970,
    10: 19.350,
    12: 16.710,
    13: 13.320,
    14: 17.990,
}

SENTINELS = (9, 99, 999, 9999, 99999, 999999, 999999.99)

# Appliance-year fits robustly re-estimated in the reported analysis.
HISTORICAL_ROBUST_REFITS = {
    (1, 2010),
    (1, 2012),
    (1, 2014),
    (1, 2021),
    (2, 2011),
    (2, 2015),
    (4, 2024),
    (5, 2015),
    (5, 2020),
    (8, 2010),
    (8, 2012),
    (9, 2017),
    (13, 2010),
    (13, 2023),
}

REPORTED_COLUMNS = [
    "artifact_code",
    "artifact",
    "year",
    "n_observations",
    "fit_status",
    "second_use_weight",
    "first_use_weight",
    "second_use_scale",
    "second_use_shape",
    "first_use_scale",
    "first_use_shape",
    "second_use_mean",
    "first_use_mean",
    "mixture_mean",
    "reference_lifetime",
    "relative_error_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Weibull-mixture and ownership results."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/enaho_module_18"),
        help="Directory containing mod18_YYYY.csv.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduced"),
        help="Directory for recomputed outputs.",
    )
    parser.add_argument(
        "--reported-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the reported CSV files.",
    )
    parser.add_argument(
        "--task",
        choices=("all", "fit", "ownership"),
        default="all",
        help="Task to run.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum EM iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="EM convergence tolerance.",
    )
    return parser.parse_args()


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    target = percentile / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target))
    index = min(index, len(values_sorted) - 1)
    return float(values_sorted[index])


def weibull_logpdf(
    values: np.ndarray,
    scale: float,
    shape: float,
) -> np.ndarray:
    if (
        scale <= 0
        or shape <= 0
        or not np.isfinite(scale)
        or not np.isfinite(shape)
    ):
        return np.full_like(values, -np.inf, dtype=float)

    values = np.maximum(values, 1e-12)
    log_ratio = np.log(values / scale)
    exponent = np.clip(shape * log_ratio, -700, 700)

    return (
        np.log(shape / scale)
        + (shape - 1.0) * log_ratio
        - np.exp(exponent)
    )


def negative_weighted_loglik_linear(
    parameters: Iterable[float],
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    scale, shape = parameters

    if scale <= 0 or shape <= 0:
        return np.inf

    objective = -np.sum(
        weights * weibull_logpdf(values, scale, shape)
    )

    return float(objective) if np.isfinite(objective) else np.inf


def negative_weighted_loglik_log(
    log_parameters: Iterable[float],
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    log_scale, log_shape = log_parameters
    scale, shape = np.exp(log_scale), np.exp(log_shape)

    objective = -np.sum(
        weights * weibull_logpdf(values, scale, shape)
    )

    return float(objective) if np.isfinite(objective) else np.inf


def standard_component_fit(
    values: np.ndarray,
    weights: np.ndarray,
    scale0: float,
    shape0: float,
) -> Tuple[float, float]:
    """
    Reproduce the standard component update.

    A numerical failure activates a local robust update for that
    component without restarting the complete mixture fit.
    """
    result = minimize(
        negative_weighted_loglik_linear,
        x0=[scale0, shape0],
        args=(values, weights),
        method="L-BFGS-B",
        bounds=[(1e-3, None), (1e-3, None)],
    )

    if result.success and np.all(np.isfinite(result.x)):
        scale, shape = result.x
        return float(scale), float(shape)

    return robust_component_fit(
        values,
        weights,
        scale0,
        shape0,
    )


def root_component_fit(
    values: np.ndarray,
    weights: np.ndarray,
    shape0: float = 1.5,
) -> Tuple[float, float]:
    total_weight = float(np.sum(weights))

    if total_weight < 1e-6:
        raise RuntimeError("Component has negligible survey weight.")

    log_values = np.log(values)

    def score(shape: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            powered = np.power(values, shape)
            weighted_powered = weights * powered
            denominator = np.sum(weighted_powered)

            if not np.isfinite(denominator) or denominator <= 0:
                return np.nan

            numerator = np.sum(weighted_powered * log_values)

            return float(
                numerator / denominator
                - 1.0 / shape
                - np.sum(weights * log_values) / total_weight
            )

    solution = root_scalar(
        score,
        x0=shape0,
        bracket=[0.05, 20.0],
    )

    if not solution.converged:
        raise RuntimeError("Weighted Weibull score equation did not converge.")

    shape = float(solution.root)
    scale = float(
        (
            np.sum(weights * np.power(values, shape))
            / total_weight
        )
        ** (1.0 / shape)
    )

    return scale, shape


def robust_component_fit(
    values: np.ndarray,
    weights: np.ndarray,
    scale0: float,
    shape0: float,
) -> Tuple[float, float]:
    if float(np.sum(weights)) < 1e-6:
        return float(scale0), float(shape0)

    result = minimize(
        negative_weighted_loglik_log,
        x0=np.log([scale0, shape0]),
        args=(values, weights),
        method="L-BFGS-B",
        options={"maxiter": 800},
    )

    if result.success and np.all(np.isfinite(result.x)):
        scale, shape = np.exp(result.x)
        return float(scale), float(shape)

    return root_component_fit(values, weights, shape0)


def run_em(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    robust: bool,
    max_iter: int,
    tol: float,
) -> Dict[str, float]:
    """
    Fit the recovered two-component weighted Weibull mixture.

    Both fitting paths use the same weighted-quantile initialization.
    The robust path changes only the component optimizer.
    """
    scale1 = max(
        weighted_percentile(
            values,
            weights,
            25,
        ),
        0.5,
    )

    scale2 = max(
        weighted_percentile(
            values,
            weights,
            75,
        ),
        scale1 * 1.2,
    )

    shape1 = 1.0
    shape2 = 1.8
    pi = 0.4

    component_fit = (
        robust_component_fit
        if robust
        else standard_component_fit
    )

    converged = False

    for iteration in range(max_iter):
        density1 = np.exp(
            weibull_logpdf(
                values,
                scale1,
                shape1,
            )
        )

        density2 = np.exp(
            weibull_logpdf(
                values,
                scale2,
                shape2,
            )
        )

        denominator = (
            (1.0 - pi) * density1
            + pi * density2
        )

        denominator[denominator == 0] = 1e-12

        responsibility2 = (
            pi * density2 / denominator
        )

        pi_new = float(
            np.sum(weights * responsibility2)
            / np.sum(weights)
        )

        weights1 = weights * (
            1.0 - responsibility2
        )
        weights2 = weights * responsibility2

        scale1_new, shape1_new = component_fit(
            values,
            weights1,
            scale1,
            shape1,
        )

        scale2_new, shape2_new = component_fit(
            values,
            weights2,
            scale2,
            shape2,
        )

        mean1_new = (
            scale1_new
            * gamma(1.0 + 1.0 / shape1_new)
        )

        mean2_new = (
            scale2_new
            * gamma(1.0 + 1.0 / shape2_new)
        )

        if mean1_new > mean2_new:
            scale1_new, scale2_new = (
                scale2_new,
                scale1_new,
            )

            shape1_new, shape2_new = (
                shape2_new,
                shape1_new,
            )

            responsibility2 = (
                1.0 - responsibility2
            )

            pi_new = float(
                np.sum(
                    weights * responsibility2
                )
                / np.sum(weights)
            )

        delta = max(
            abs(pi_new - pi),
            abs(scale1_new - scale1),
            abs(shape1_new - shape1),
            abs(scale2_new - scale2),
            abs(shape2_new - shape2),
        )

        pi = pi_new
        scale1 = scale1_new
        shape1 = shape1_new
        scale2 = scale2_new
        shape2 = shape2_new

        if delta < tol:
            converged = True
            break

    mean1 = float(
        scale1 * gamma(
            1.0 + 1.0 / shape1
        )
    )

    mean2 = float(
        scale2 * gamma(
            1.0 + 1.0 / shape2
        )
    )

    mixture_mean = float(
        (1.0 - pi) * mean1
        + pi * mean2
    )

    return {
        "pi": float(pi),
        "scale1": float(scale1),
        "shape1": float(shape1),
        "scale2": float(scale2),
        "shape2": float(shape2),
        "mean1": mean1,
        "mean2": mean2,
        "mixture_mean": mixture_mean,
        "iterations": iteration + 1,
        "converged": converged,
    }



def read_year(data_dir: Path, year: int) -> pd.DataFrame:
    path = data_dir / f"mod18_{year}.csv.gz"

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    data = pd.read_csv(
        path,
        encoding="latin1",
        low_memory=False,
    )

    data.rename(
        columns={data.columns[0]: "ANO"},
        inplace=True,
    )

    required = ["ANO", "P612", "P612N", "P612C", "FACTOR07"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    data.replace(
        {
            "P612C": SENTINELS,
            "FACTOR07": SENTINELS,
        },
        np.nan,
        inplace=True,
    )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data


def lifespan_sample(
    data: pd.DataFrame,
    artifact_code: int,
    year: int,
) -> pd.DataFrame:
    sample = data.loc[
        (data["P612"] == 1)
        & (data["P612N"] == artifact_code)
        & (data["ANO"] == year)
        & data["P612C"].notna()
        & data["FACTOR07"].notna()
    ].copy()

    sample["lifetime"] = sample["ANO"] - sample["P612C"]

    return sample.loc[
        (sample["lifetime"] > 0)
        & (sample["lifetime"] <= 80)
    ].copy()


def fit_all_mixtures(
    data_dir: Path,
    max_iter: int,
    tol: float,
) -> pd.DataFrame:
    """
    Recompute all 204 appliance-year estimates.

    Standard rows retain the historical stored precision:
    pi to three decimals and the remaining mixture quantities
    to two decimals. Robust refits retain full precision.
    """
    records = []
    total = len(YEARS) * len(ARTIFACTS)
    completed = 0

    for year in YEARS:
        data = read_year(data_dir, year)

        for (
            artifact_code,
            artifact_name,
        ) in ARTIFACTS.items():
            completed += 1

            sample = lifespan_sample(
                data,
                artifact_code,
                year,
            )

            if len(sample) < 20:
                raise RuntimeError(
                    "Insufficient observations for "
                    f"artifact={artifact_code}, "
                    f"year={year}: "
                    f"n={len(sample)}"
                )

            values = sample[
                "lifetime"
            ].to_numpy(dtype=float)

            weights = sample[
                "FACTOR07"
            ].to_numpy(dtype=float)

            robust_refit = (
                artifact_code,
                year,
            ) in HISTORICAL_ROBUST_REFITS

            fit_status = (
                "robust_refit"
                if robust_refit
                else "standard"
            )

            fit = run_em(
                values,
                weights,
                robust=robust_refit,
                max_iter=max_iter,
                tol=tol,
            )

            reference_lifetime = (
                REFERENCE_LIFETIMES[
                    artifact_code
                ]
            )

            if robust_refit:
                pi = fit["pi"]
                scale1 = fit["scale1"]
                shape1 = fit["shape1"]
                scale2 = fit["scale2"]
                shape2 = fit["shape2"]
                mean1 = fit["mean1"]
                mean2 = fit["mean2"]
                mixture_mean = (
                    fit["mixture_mean"]
                )

                second_use_weight = (
                    1.0 - pi
                )
                first_use_weight = pi

                relative_error = (
                    abs(
                        mixture_mean
                        - reference_lifetime
                    )
                    / reference_lifetime
                    * 100.0
                )

            else:
                pi = round(
                    fit["pi"],
                    3,
                )

                scale1 = round(
                    fit["scale1"],
                    2,
                )

                shape1 = round(
                    fit["shape1"],
                    2,
                )

                scale2 = round(
                    fit["scale2"],
                    2,
                )

                shape2 = round(
                    fit["shape2"],
                    2,
                )

                mean1 = round(
                    fit["mean1"],
                    2,
                )

                mean2 = round(
                    fit["mean2"],
                    2,
                )

                mixture_mean = round(
                    fit["mixture_mean"],
                    2,
                )

                second_use_weight = round(
                    1.0 - pi,
                    3,
                )

                first_use_weight = pi

                relative_error = round(
                    abs(
                        mixture_mean
                        - reference_lifetime
                    )
                    / reference_lifetime
                    * 100.0,
                    2,
                )

            records.append(
                {
                    "artifact_code":
                        artifact_code,
                    "artifact":
                        artifact_name,
                    "year":
                        year,
                    "n_observations":
                        len(sample),
                    "fit_status":
                        fit_status,
                    "second_use_weight":
                        second_use_weight,
                    "first_use_weight":
                        first_use_weight,
                    "second_use_scale":
                        scale1,
                    "second_use_shape":
                        shape1,
                    "first_use_scale":
                        scale2,
                    "first_use_shape":
                        shape2,
                    "second_use_mean":
                        mean1,
                    "first_use_mean":
                        mean2,
                    "mixture_mean":
                        mixture_mean,
                    "reference_lifetime":
                        reference_lifetime,
                    "relative_error_percent":
                        relative_error,
                }
            )

            print(
                f"[{completed:3d}/{total}] "
                f"{artifact_code:2d}-{year}: "
                f"{fit_status}",
                flush=True,
            )

    result = pd.DataFrame(
        records
    )[REPORTED_COLUMNS]

    return (
        result.sort_values(
            [
                "artifact_code",
                "year",
            ]
        )
        .reset_index(drop=True)
    )



def ownership_yes_column(columns: Iterable[object]) -> object:
    candidates = list(columns)

    for candidate in candidates:
        if candidate == 1:
            return candidate

        normalized = str(candidate).strip().lower()

        if normalized in {"1", "si", "sí", "yes"}:
            return candidate

    raise KeyError(
        f"Could not identify the ownership='yes' column among {candidates}"
    )


def compute_ownership(data_dir: Path) -> pd.DataFrame:
    records = []

    for year in YEARS:
        data = read_year(data_dir, year)

        sample = data.loc[
            data["P612N"].isin(ARTIFACTS)
            & data["P612"].notna()
            & data["FACTOR07"].notna(),
            ["P612N", "P612", "FACTOR07"],
        ].copy()

        weighted_totals = (
            sample.groupby(["P612N", "P612"], as_index=False)["FACTOR07"]
            .sum()
        )

        pivot = weighted_totals.pivot(
            index="P612N",
            columns="P612",
            values="FACTOR07",
        ).fillna(0.0)

        percentages = pivot.div(
            pivot.sum(axis=1),
            axis=0,
        ) * 100.0

        yes_column = ownership_yes_column(percentages.columns)

        for artifact_code, row in percentages.iterrows():
            artifact_code = int(artifact_code)

            if artifact_code not in ARTIFACTS:
                continue

            records.append(
                {
                    "artifact_code": artifact_code,
                    "artifact": ARTIFACTS[artifact_code],
                    "year": year,
                    "ownership_percent": round(
                        float(row[yes_column]),
                        1,
                    ),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(["artifact_code", "year"])
        .reset_index(drop=True)
    )


def compare_mixture_results(
    recomputed: pd.DataFrame,
    reported_path: Path,
) -> str:
    if not reported_path.exists():
        return f"Reported mixture file not found: {reported_path}"

    reported = pd.read_csv(reported_path)

    required = set(REPORTED_COLUMNS)
    missing = sorted(required - set(reported.columns))

    if missing:
        return (
            "Reported mixture file uses an unexpected schema. "
            f"Missing columns: {missing}"
        )

    merged = reported.merge(
        recomputed,
        on=["artifact_code", "year"],
        suffixes=("_reported", "_recomputed"),
        validate="one_to_one",
    )

    numeric_columns = [
        "n_observations",
        "second_use_weight",
        "first_use_weight",
        "second_use_scale",
        "second_use_shape",
        "first_use_scale",
        "first_use_shape",
        "second_use_mean",
        "first_use_mean",
        "mixture_mean",
        "reference_lifetime",
        "relative_error_percent",
    ]

    lines = [
        "WEIBULL-MIXTURE COMPARISON",
        f"Reported rows:   {len(reported)}",
        f"Recomputed rows: {len(recomputed)}",
        f"Matched rows:    {len(merged)}",
        "",
        "Maximum absolute differences:",
    ]

    for column in numeric_columns:
        reported_values = merged[f"{column}_reported"]
        recomputed_values = merged[f"{column}_recomputed"]

        difference = np.abs(reported_values - recomputed_values)

        if "shape" in column:
            degenerate = np.maximum(
                np.abs(reported_values),
                np.abs(recomputed_values),
            ) > 1000

            difference = difference.mask(degenerate)

        maximum = difference.max(skipna=True)
        lines.append(f"  {column:24s} {maximum:.12g}")

    observation_mismatches = int(
        (
            merged["n_observations_reported"]
            != merged["n_observations_recomputed"]
        ).sum()
    )

    lines.extend(
        [
            "",
            f"Observation-count mismatches: {observation_mismatches}",
            (
                "Shape differences above are omitted for components with "
                "shape > 1000 because those near-degenerate estimates are "
                "numerically weakly identified."
            ),
        ]
    )

    return "\n".join(lines)


def compare_ownership_results(
    recomputed: pd.DataFrame,
    reported_path: Path,
) -> str:
    if not reported_path.exists():
        return f"Reported ownership file not found: {reported_path}"

    reported = pd.read_csv(reported_path)

    required = {
        "artifact_code",
        "artifact",
        "year",
        "ownership_percent",
    }

    missing = sorted(required - set(reported.columns))

    if missing:
        return (
            "Reported ownership file uses an unexpected schema. "
            f"Missing columns: {missing}"
        )

    merged = reported.merge(
        recomputed,
        on=["artifact_code", "year"],
        suffixes=("_reported", "_recomputed"),
        validate="one_to_one",
    )

    difference = np.abs(
        merged["ownership_percent_reported"]
        - merged["ownership_percent_recomputed"]
    )

    return "\n".join(
        [
            "OWNERSHIP COMPARISON",
            f"Reported rows:   {len(reported)}",
            f"Recomputed rows: {len(recomputed)}",
            f"Matched rows:    {len(merged)}",
            (
                "Maximum absolute percentage-point difference: "
                f"{difference.max():.12g}"
            ),
        ]
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="overflow encountered",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered",
        category=RuntimeWarning,
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_sections = []

    if args.task in {"all", "fit"}:
        mixture_results = fit_all_mixtures(
            args.data_dir,
            max_iter=args.max_iter,
            tol=args.tol,
        )

        mixture_output = (
            args.output_dir / "weibull_mixture_estimates.csv"
        )

        mixture_results.to_csv(
            mixture_output,
            index=False,
        )

        comparison_sections.append(
            compare_mixture_results(
                mixture_results,
                args.reported_dir / "weibull_mixture_estimates.csv",
            )
        )

        print(f"Saved: {mixture_output}")

    if args.task in {"all", "ownership"}:
        ownership_results = compute_ownership(args.data_dir)

        ownership_output = (
            args.output_dir / "ownership_penetration_2008_2024.csv"
        )

        ownership_results.to_csv(
            ownership_output,
            index=False,
        )

        comparison_sections.append(
            compare_ownership_results(
                ownership_results,
                (
                    args.reported_dir
                    / "ownership_penetration_2008_2024.csv"
                ),
            )
        )

        print(f"Saved: {ownership_output}")

    report = "\n\n".join(comparison_sections) + "\n"
    report_path = args.output_dir / "comparison_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
