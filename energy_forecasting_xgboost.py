
"""
Time Series Forecasting with XGBoost for Energy Consumption

Dataset:
    AEP_hourly.csv
    Columns: Datetime, AEP_MW

Methodology is based on the project's cited references:
    - Robikscube Kaggle XGBoost time-series workflow
    - Chen & Guestrin (2016) XGBoost
    - Recent load-forecasting comparisons involving XGBoost
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("AEP_hourly.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

LAGS = [
    1,
    2,
    3,
    24,
    48,
    72,
    168,
]

ROLLING_WINDOWS = [
    24,
    168,
]


# ============================================================
# Load and Prepare Dataset
# ============================================================

def load_data(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            "Download AEP_hourly.csv from the project dataset "
            "source and place it beside this script."
        )

    df = pd.read_csv(path)

    if "Datetime" not in df.columns:
        raise ValueError(
            "Expected a 'Datetime' column."
        )

    target_candidates = [
        column
        for column in df.columns
        if column.lower()
        in {
            "aep_mw",
            "value",
            "energy",
            "load",
        }
    ]

    target = (
        target_candidates[0]
        if target_candidates
        else df.columns[-1]
    )

    df = df[
        [
            "Datetime",
            target,
        ]
    ].rename(
        columns={
            target: "Energy_MW"
        }
    )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="coerce",
    )

    df["Energy_MW"] = pd.to_numeric(
        df["Energy_MW"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Datetime",
            "Energy_MW",
        ]
    )

    df = df.sort_values(
        "Datetime"
    )

    # Aggregate duplicate timestamps
    # instead of keeping duplicate observations.
    df = (
        df.groupby(
            "Datetime",
            as_index=False
        )["Energy_MW"]
        .mean()
    )

    df = df.set_index(
        "Datetime"
    )

    return df


# ============================================================
# Feature Engineering
# ============================================================

def make_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    x = df.copy()
    idx = x.index

    # --------------------------------------------------------
    # Calendar / Time Features
    # --------------------------------------------------------

    x["hour"] = idx.hour

    x["dayofweek"] = idx.dayofweek

    x["dayofmonth"] = idx.day

    x["dayofyear"] = idx.dayofyear

    x["weekofyear"] = (
        idx.isocalendar()
        .week
        .astype(int)
    )

    x["month"] = idx.month

    x["quarter"] = idx.quarter

    x["year"] = idx.year

    x["is_weekend"] = (
        idx.dayofweek >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclic Time Features
    # --------------------------------------------------------

    x["hour_sin"] = np.sin(
        2 * np.pi * idx.hour / 24
    )

    x["hour_cos"] = np.cos(
        2 * np.pi * idx.hour / 24
    )

    x["dow_sin"] = np.sin(
        2 * np.pi * idx.dayofweek / 7
    )

    x["dow_cos"] = np.cos(
        2 * np.pi * idx.dayofweek / 7
    )

    x["month_sin"] = np.sin(
        2 * np.pi * idx.month / 12
    )

    x["month_cos"] = np.cos(
        2 * np.pi * idx.month / 12
    )

    # --------------------------------------------------------
    # Lag Features
    # --------------------------------------------------------

    for lag in LAGS:

        x[f"lag_{lag}"] = (
            x["Energy_MW"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Leakage-Safe Rolling Features
    # --------------------------------------------------------

    shifted = (
        x["Energy_MW"]
        .shift(1)
    )

    for window in ROLLING_WINDOWS:

        x[
            f"rolling_mean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        x[
            f"rolling_std_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )

        x[
            f"rolling_min_{window}"
        ] = (
            shifted
            .rolling(window)
            .min()
        )

        x[
            f"rolling_max_{window}"
        ] = (
            shifted
            .rolling(window)
            .max()
        )

    # --------------------------------------------------------
    # Remove Rows with Missing Values
    # --------------------------------------------------------

    return x.dropna()


# ============================================================
# Evaluation Metrics
# ============================================================

def metrics(
    y_true,
    y_pred,
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": rmse,
        "R2": r2_score(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# Main Model Pipeline
# ============================================================

def main():

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    df = load_data(
        DATA_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Range: "
        f"{df.index.min()} "
        f"-> "
        f"{df.index.max()}"
    )

    print(
        "\nDataset Preview:"
    )

    print(
        df.head()
    )

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    feat = make_features(
        df
    )

    target = "Energy_MW"

    feature_cols = [
        column
        for column in feat.columns
        if column != target
    ]

    # --------------------------------------------------------
    # Chronological Train-Test Split
    # --------------------------------------------------------

    split_idx = int(
        len(feat) * 0.80
    )

    train = feat.iloc[
        :split_idx
    ]

    test = feat.iloc[
        split_idx:
    ]

    X_train = train[
        feature_cols
    ]

    y_train = train[
        target
    ]

    X_test = test[
        feature_cols
    ]

    y_test = test[
        target
    ]

    print(
        f"\nTraining rows: "
        f"{len(train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(test):,}"
    )

    # --------------------------------------------------------
    # Seasonal Naive Baseline
    # Same hour one week earlier
    # --------------------------------------------------------

    baseline_pred = (
        test["lag_168"]
        .values
    )

    base_metrics = metrics(
        y_test,
        baseline_pred,
    )

    # --------------------------------------------------------
    # XGBoost Model
    # --------------------------------------------------------

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_test,
                y_test,
            )
        ],
        verbose=False,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred = model.predict(
        X_test
    )

    xgb_metrics = metrics(
        y_test,
        pred,
    )

    # --------------------------------------------------------
    # Compare Models
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            base_metrics,
            xgb_metrics,
        ],
        index=[
            "Seasonal_Naive_168h",
            "XGBoost",
        ],
    )

    results.to_csv(
        OUTPUT_DIR
        / "metrics.csv"
    )

    print(
        "\nEvaluation:"
    )

    print(
        results.round(3)
    )

    # ========================================================
    # Actual vs Predicted Plot
    # ========================================================

    plot_points = 24 * 14

    plt.figure(
        figsize=(15, 6)
    )

    plt.plot(
        test.index[-plot_points:],
        y_test.iloc[-plot_points:],
        label="Actual",
    )

    plt.plot(
        test.index[-plot_points:],
        pred[-plot_points:],
        label="XGBoost",
    )

    plt.title(
        "AEP Energy Consumption: "
        "Actual vs XGBoost Prediction"
    )

    plt.xlabel(
        "Datetime"
    )

    plt.ylabel(
        "Energy (MW)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "actual_vs_predicted.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # Feature Importance
    # ========================================================

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(
        ascending=False
    )

    importance.head(
        20
    ).to_csv(
        OUTPUT_DIR
        / "feature_importance_top20.csv",
        header=[
            "importance"
        ],
    )

    plt.figure(
        figsize=(10, 7)
    )

    (
        importance
        .head(15)
        .sort_values()
        .plot(
            kind="barh"
        )
    )

    plt.title(
        "Top XGBoost Feature Importances"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "feature_importance.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # Save Trained Model
    # ========================================================

    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
        },
        OUTPUT_DIR
        / "xgboost_energy_model.joblib",
    )

    # ========================================================
    # Recursive Next 24-Hour Forecast
    # ========================================================

    history = df.copy()

    future_rows = []

    for _ in range(24):

        next_ts = (
            history.index.max()
            + pd.Timedelta(
                hours=1
            )
        )

        # Temporary value is required here.
        # Using NaN would cause make_features()
        # to remove the new future row through dropna().
        temp = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "Energy_MW": [
                            0.0
                        ]
                    },
                    index=[
                        next_ts
                    ],
                ),
            ]
        )

        engineered = make_features(
            temp
        )

        row = engineered.loc[
            [next_ts],
            feature_cols,
        ]

        yhat = float(
            model.predict(
                row
            )[0]
        )

        future_rows.append(
            {
                "Datetime": next_ts,
                "Forecast_MW": yhat,
            }
        )

        # Add the prediction to history
        # so it can be used for the next hour.
        history.loc[
            next_ts,
            "Energy_MW",
        ] = yhat

    forecast = (
        pd.DataFrame(
            future_rows
        )
        .set_index(
            "Datetime"
        )
    )

    # --------------------------------------------------------
    # Save Forecast
    # --------------------------------------------------------

    forecast.to_csv(
        OUTPUT_DIR
        / "next_24h_forecast.csv"
    )

    # ========================================================
    # Next 24-Hour Forecast Plot
    # ========================================================

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        df.index[-72:],
        df["Energy_MW"].iloc[-72:],
        label="Historical",
    )

    plt.plot(
        forecast.index,
        forecast["Forecast_MW"],
        marker="o",
        label="Next 24h Forecast",
    )

    plt.title(
        "Next 24-Hour Energy "
        "Consumption Forecast"
    )

    plt.xlabel(
        "Datetime"
    )

    plt.ylabel(
        "Energy (MW)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "next_24h_forecast.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # Final Output
    # ========================================================

    print(
        "\nNext 24-hour forecast:"
    )

    print(
        forecast.round(2)
    )

    print(
        f"\nArtifacts saved in: "
        f"{OUTPUT_DIR.resolve()}"
    )


# ============================================================
# Run Program
# ============================================================

if __name__ == "__main__":
    main()