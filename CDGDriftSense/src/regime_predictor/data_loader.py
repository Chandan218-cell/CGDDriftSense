"""
data_loader.py
==============
Fetches and preprocesses WTI crude oil and LME copper daily price series.

Data sources:
  - WTI crude: U.S. Energy Information Administration (EIA) public API
  - LME copper: Quandl/Nasdaq Data Link API

Preprocessing applied:
  - Forward-fill missing observations (exchange holidays)
  - Log transformation for variance stabilisation across price levels
  - Alignment to a common date index
"""

import os
import logging
import pandas as pd
import numpy as np
import requests
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_wti_crude(start_date: str = "2000-01-01", end_date: str = "2023-12-31") -> pd.Series:
    """
    Fetch WTI crude oil daily spot prices from the EIA API.

    Args:
        start_date: ISO date string for series start.
        end_date: ISO date string for series end.

    Returns:
        pd.Series with DatetimeIndex, named 'WTI_close'.

    Raises:
        ValueError: if EIA_API_KEY environment variable is not set.
        requests.HTTPError: on API failure.
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError(
            "EIA_API_KEY environment variable not set. "
            "Get a free key at https://www.eia.gov/opendata/"
        )

    url = (
        f"https://api.eia.gov/v2/petroleum/pri/spt/data/"
        f"?api_key={api_key}&frequency=daily"
        f"&data[0]=value&facets[product][]=EPCWTI"
        f"&start={start_date}&end={end_date}&sort[0][column]=period"
        f"&sort[0][direction]=asc&length=5000"
    )
    logger.info("Fetching WTI crude prices from EIA API (%s to %s)", start_date, end_date)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    records = response.json()["response"]["data"]
    series = pd.Series(
        {r["period"]: float(r["value"]) for r in records},
        name="WTI_close",
    )
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def fetch_lme_copper(start_date: str = "2000-01-01", end_date: str = "2023-12-31") -> pd.Series:
    """
    Fetch LME copper daily settlement prices from Quandl/Nasdaq Data Link.

    Args:
        start_date: ISO date string.
        end_date: ISO date string.

    Returns:
        pd.Series with DatetimeIndex, named 'Copper_close'.
    """
    api_key = os.getenv("QUANDL_API_KEY")
    if not api_key:
        raise ValueError(
            "QUANDL_API_KEY environment variable not set. "
            "Get a free key at https://data.nasdaq.com/"
        )

    url = (
        f"https://data.nasdaq.com/api/v3/datasets/LME/PR_CU.json"
        f"?api_key={api_key}&start_date={start_date}&end_date={end_date}"
        f"&order=asc"
    )
    logger.info("Fetching LME copper prices from Quandl (%s to %s)", start_date, end_date)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()["dataset"]
    col_names = [c.lower() for c in data["column_names"]]
    df = pd.DataFrame(data["data"], columns=col_names)
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["settle"].rename("Copper_close")
    return series.sort_index()


def build_price_dataframe(
    wti: pd.Series,
    copper: pd.Series,
    start_date: str = "2000-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame:
    """
    Align WTI and copper series to a common business-day index, forward-fill
    missing observations (exchange holidays), and apply log transformation.

    Forward-filling methodology note: for regime identification tasks where the
    relevant time scale is weeks to months, forward-filling introduces no
    material distortion. It would be inappropriate for high-frequency or
    event-study analysis.

    Args:
        wti: WTI daily close price series.
        copper: LME copper daily settlement series.
        start_date: ISO date string for output series start.
        end_date: ISO date string for output series end.

    Returns:
        DataFrame with columns: WTI_close, Copper_close, WTI_log, Copper_log.
        Index is a business-day DatetimeIndex with no NaN values.
    """
    bday_index = pd.bdate_range(start=start_date, end=end_date)
    df = pd.DataFrame(index=bday_index)
    df["WTI_close"] = wti.reindex(bday_index)
    df["Copper_close"] = copper.reindex(bday_index)

    # Forward-fill holiday gaps
    df = df.ffill()

    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Remaining NaN values after forward-fill: %s", missing[missing > 0].to_dict())
        df = df.dropna()

    # Log transformation: converts multiplicative price dynamics to additive
    # return dynamics; stabilises variance across price level regimes
    df["WTI_log"] = np.log(df["WTI_close"])
    df["Copper_log"] = np.log(df["Copper_close"])

    logger.info(
        "Price DataFrame built: %d observations, %s to %s",
        len(df), df.index[0].date(), df.index[-1].date()
    )
    return df


def load_price_data(
    start_date: str = "2000-01-01",
    end_date: str = "2023-12-31",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full ingestion pipeline: fetch, align, transform, and optionally cache.

    Args:
        start_date: Start of the price series.
        end_date: End of the price series.
        cache_path: If provided and file exists, load from disk instead of API.
                    If provided and file does not exist, save result to disk.

    Returns:
        Preprocessed price DataFrame ready for feature engineering.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached price data from %s", cache_path)
        return pd.read_parquet(cache_path)

    wti = fetch_wti_crude(start_date, end_date)
    copper = fetch_lme_copper(start_date, end_date)
    df = build_price_dataframe(wti, copper, start_date, end_date)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Price data cached to %s", cache_path)

    return df
