"""
feature_engineering.py
=======================
Constructs the feature matrix for the price regime prediction module.

Feature groups:
  1. Log-returns        — daily momentum signal
  2. Rolling statistics — multi-scale trend (5d / 21d / 63d windows)
  3. Technical indicators — RSI, Bollinger Band Width, MACD signal line
  4. Cross-commodity correlation — 21-day rolling WTI × Copper correlation
  5. PCA compression — retains 95% of cumulative variance

All features are computed on log-price series to ensure variance-stationarity
across the different price-level regimes present in the 2000–2023 window.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = [5, 21, 63]   # weekly, monthly, quarterly
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
CORR_WINDOW = 21
PCA_VARIANCE_THRESHOLD = 0.95


def log_returns(series: pd.Series, name: str) -> pd.Series:
    """Compute daily log-returns from a log-price series."""
    return series.diff().rename(f"{name}_return")


def rolling_stats(series: pd.Series, name: str) -> pd.DataFrame:
    """
    Compute rolling mean and standard deviation at three time scales.
    Windows: 5d (weekly), 21d (monthly), 63d (quarterly).
    """
    frames = {}
    for w in ROLLING_WINDOWS:
        frames[f"{name}_ma{w}"] = series.rolling(w).mean()
        frames[f"{name}_vol{w}"] = series.rolling(w).std()
    return pd.DataFrame(frames)


def rsi(series: pd.Series, period: int = RSI_PERIOD, name: str = "RSI") -> pd.Series:
    """Relative Strength Index on log-price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"{name}_rsi")


def bollinger_band_width(
    series: pd.Series,
    window: int = BB_WINDOW,
    num_std: float = 2.0,
    name: str = "BBW",
) -> pd.Series:
    """
    Bollinger Band Width: (upper - lower) / middle band.
    A measure of volatility that normalises bandwidth by price level.
    """
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    bbw = (upper - lower) / ma
    return bbw.rename(f"{name}_bbw")


def macd_signal(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
    name: str = "MACD",
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.
    Captures medium-term trend dynamics beyond raw returns.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        f"{name}_line": macd_line,
        f"{name}_signal": signal_line,
        f"{name}_hist": histogram,
    })


def cross_commodity_correlation(
    wti_log: pd.Series,
    copper_log: pd.Series,
    window: int = CORR_WINDOW,
) -> pd.Series:
    """
    21-day rolling Pearson correlation between WTI and copper log-prices.
    Acts as a macro-economic context feature: high correlation signals
    broad commodity market movement rather than commodity-specific dynamics.
    """
    return wti_log.rolling(window).corr(copper_log).rename("WTI_Copper_corr21d")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the full feature matrix from a preprocessed price DataFrame.

    Args:
        df: DataFrame with columns WTI_log, Copper_log (output of data_loader).

    Returns:
        Feature DataFrame with all engineered features. NaN rows (due to
        rolling window initialisation) are dropped.
    """
    features = pd.DataFrame(index=df.index)

    for asset, col in [("WTI", "WTI_log"), ("Copper", "Copper_log")]:
        features[f"{asset}_return"] = log_returns(df[col], asset)
        features = features.join(rolling_stats(df[col], asset))
        features[f"{asset}_rsi"] = rsi(df[col], name=asset)
        features[f"{asset}_bbw"] = bollinger_band_width(df[col], name=asset)
        features = features.join(macd_signal(df[col], name=asset))

    features["WTI_Copper_corr21d"] = cross_commodity_correlation(
        df["WTI_log"], df["Copper_log"]
    )

    n_raw = len(features)
    features = features.dropna()
    logger.info(
        "Feature matrix built: %d features, %d observations (dropped %d NaN rows)",
        features.shape[1], len(features), n_raw - len(features),
    )
    return features


def apply_pca(
    feature_matrix: pd.DataFrame,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
    scaler: StandardScaler = None,
    pca: PCA = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler, PCA]:
    """
    Standardise and compress the feature matrix using PCA.

    Retains components explaining `variance_threshold` of cumulative variance
    (default 95%). Separation of fit/transform supports the rolling-window
    train/inference pattern: fit on training window, transform on new data.

    Args:
        feature_matrix: Output of build_feature_matrix().
        variance_threshold: Cumulative variance to retain (0–1).
        scaler: Pre-fitted StandardScaler (pass None to fit new one).
        pca: Pre-fitted PCA (pass None to fit new one).
        fit: If True, fit scaler and PCA. If False, transform only.

    Returns:
        Tuple of (compressed DataFrame, fitted scaler, fitted PCA).
    """
    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)
        pca = PCA(n_components=variance_threshold, svd_solver="full")
        compressed = pca.fit_transform(scaled)
        logger.info(
            "PCA: reduced %d → %d components (%.1f%% variance retained)",
            feature_matrix.shape[1],
            pca.n_components_,
            pca.explained_variance_ratio_.sum() * 100,
        )
    else:
        if scaler is None or pca is None:
            raise ValueError("scaler and pca must be provided when fit=False")
        scaled = scaler.transform(feature_matrix)
        compressed = pca.transform(scaled)

    col_names = [f"PC{i+1}" for i in range(compressed.shape[1])]
    return (
        pd.DataFrame(compressed, index=feature_matrix.index, columns=col_names),
        scaler,
        pca,
    )
