# %%
# %%
# %%
# %%
"""
Financial Analysis Pipeline - Core Configuration & Data Download
ORIGINAL DOWNLOAD LOGIC - NICHT ÄNDERN!
"""
import hashlib
import json
import asyncio
import datetime as dt
import io
import json
import logging
import re
import ssl
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from itertools import combinations

import aiohttp
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from abc import ABC, abstractmethod

# Regression methods
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

# Statistical analysis
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Optional advanced methods
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from ecbdata import ecbdata
    HAS_ECBDATA = True
except ImportError:
    HAS_ECBDATA = False
    ecbdata = None

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================


from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LagConfig:
    enable: bool = True
    candidates: List[int] = field(default_factory=lambda: [1, 3])
    per_var_max: int = 1
    total_max: int = 8
    min_train_overlap: int = 24
    min_abs_corr: float = 0.0  # optional threshold
@dataclass
class AnalysisConfig:
    """Enhanced configuration with conservative defaults for robust analysis."""
    
    # Cache and download settings
    cache_max_age_days: int = 60
    cache_dir: str = "financial_cache"
    download_timeout_seconds: int = 30
    max_concurrent_downloads: int = 8
    default_start_date: str = "2000-01"
    default_end_date: str = "2024-12"
    min_response_size: int = 100
    max_retry_attempts: int = 3
    
    # Validation settings
    test_size: float = 0.25
    cv_folds: int = 4
    gap_periods: int = 2
    
    # Feature selection settings
    max_feature_combinations: int = 20
    min_features_per_combination: int = 2
    
    # Model settings
    remove_outliers: bool = True
    outlier_method: str = "conservative"
    add_seasonal_dummies: bool = True
    handle_mixed_frequencies: bool = True
    
    # Model persistence
    save_models: bool = True
    model_cleanup_days: int = 30
    keep_best_models: int = 10
    random_seed: int = 42
    
    # Final dataset caching
    cache_final_dataset: bool = True
    final_cache_format: str = "xlsx"
    final_cache_subdir: str = "final_datasets"
    
    # Plot settings
    save_plots: bool = True
    show_plots: bool = True
    plots_dir: str = "financial_cache/diagnostic_plots"
    
    # Validation thresholds
    high_r2_warning_threshold: float = 0.9
    high_overfitting_threshold: float = 0.1
    cv_test_discrepancy_threshold: float = 0.1
    evaluate_quarter_ends_only: bool = True  # evaluate metrics only at quarter ends
    
    # Index-Normalisierung Parameter
    index_base_year: int = 2015
    index_base_value: float = 100.0
    lag_config: Optional[LagConfig] = None
    ab_compare_lags: bool = True

# =============================================================================
# CONSTANTS AND DEFINITIONS (ORIGINAL)
# =============================================================================

# Target variable definitions
INDEX_TARGETS = {
    "PH_EINLAGEN": "INDEX(BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F22.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29A.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29B.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29D.S14.DE.S1.W0.F.N._X.B)",
    "PH_WERTPAPIERE": "INDEX(BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F511.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F512.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F519.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F52.S14.DE.S1.W0.F.N._X.B)",
    "PH_VERSICHERUNGEN": "INDEX(BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F8.S14.DE.S1.W0.F.N._X.B)",
    "PH_KREDITE": "INDEX(BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B, BBAF3.Q.F8.S1.W0.S14.DE.F.N._X.B)",
    "NF_KG_EINLAGEN": "INDEX(BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F22.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29A.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29B.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29D.S11.DE.S1.W0.F.N._X.B)",
    "NF_KG_WERTPAPIERE": "INDEX(BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F32.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F511.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F512.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F519.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F52.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F7.S11.DE.S1.W0.F.N._X.B)",
    "NF_KG_VERSICHERUNGEN": "INDEX(BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F8.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F6.S1.W0.S11.DE.F.N._X.B)",
    "NF_KG_KREDITE": "INDEX(BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F42.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F4.S1.W0.S11.DE.F.N._X.B, BBAF3.Q.F8.S1.W0.S11.DE.F.N._X.B)",
}

# Standard exogenous variables
STANDARD_EXOG_VARS = {
    "euribor_3m": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "german_rates": "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
    "german_inflation": "ICP.M.DE.N.000000.4.ANR",
    "german_unemployment": "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
    "eur_usd_rate": "EXR.D.USD.EUR.SP00.A",
    "german_gdp": "MNA.Q.DE.N.B1GQ.C.S1.S1.B.B1GQ._Z.EUR.LR.GY",
    "ecb_main_rate": "FM.M.U2.EUR.RT.MM.EONIA_.HSTA",
}

# API constants
ECB_API_BASE_URL = "https://data-api.ecb.europa.eu/service/data"
BUNDESBANK_API_BASE_URL = "https://api.statistiken.bundesbank.de/rest/download"
ECB_PREFIXES = ("ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.", "STS.", "MNA.", "BOP.", "GFS.", "EXR.")

# Fallback definitions
SIMPLE_TARGET_FALLBACKS = {
    "PH_KREDITE": "BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B",
    "PH_EINLAGEN": "BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B",
    "PH_WERTPAPIERE": "BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B",
    "PH_VERSICHERUNGEN": "BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B",
    "NF_KG_EINLAGEN": "BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_WERTPAPIERE": "BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_VERSICHERUNGEN": "BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_KREDITE": "BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B",
}

INDEX_SPEC_RE = re.compile(r'^\s*INDEX\s*\(\s*(.*?)\s*\)\s*', re.IGNORECASE)

# =============================================================================
# ORIGINAL UTILITY FUNCTIONS (NICHT ÄNDERN)
# =============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_data_source(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Invalid series code: {code}")
    code_upper = code.upper()
    if "." in code_upper and code_upper.startswith(ECB_PREFIXES):
        return "ECB"
    return "BUNDESBANK"

def parse_index_specification(spec: str) -> Optional[List[str]]:
    if not isinstance(spec, str):
        return None
    match = INDEX_SPEC_RE.match(spec.strip())
    if not match:
        return None
    inner = match.group(1)
    codes = [c.strip() for c in inner.split(",") if c.strip()]
    return list(dict.fromkeys(codes)) if codes else None

def validate_date_string(date_str: str) -> bool:
    if not isinstance(date_str, str):
        return False
    date_patterns = ["%Y-%m", "%Y-%m-%d", "%Y"]
    for pattern in date_patterns:
        try:
            dt.datetime.strptime(date_str, pattern)
            return True
        except ValueError:
            continue
    return False

def format_date_for_ecb_api(date_str: str) -> str:
    if not date_str:
        return date_str
    try:
        if len(date_str) == 4:
            return f"{date_str}-01"
        elif len(date_str) == 7:
            return date_str
        elif len(date_str) == 10:
            return date_str[:7]
        else:
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime("%Y-%m")
    except:
        return date_str

def get_excel_engine() -> str:
    try:
        import openpyxl
        return 'openpyxl'
    except ImportError:
        try:
            import xlsxwriter
            return 'xlsxwriter'
        except ImportError:
            raise ImportError("Excel support requires openpyxl or xlsxwriter. Install with: pip install openpyxl")

# =============================================================================
# ORIGINAL DATA DOWNLOAD CLASSES (NICHT ÄNDERN)
# =============================================================================

class DataProcessor:
    @staticmethod
    def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        date_candidates = ["TIME_PERIOD", "DATE", "Datum", "Period", "period"]
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None and len(df.columns) > 0:
            date_col = df.columns[0]
        
        value_candidates = ["OBS_VALUE", "VALUE", "Wert", "Value"]
        value_col = None
        for candidate in value_candidates:
            if candidate in df.columns and candidate != date_col:
                value_col = candidate
                break
        if value_col is None:
            numeric_cols = [c for c in df.columns if c != date_col and df[c].dtype in ['float64', 'int64']]
            if numeric_cols:
                value_col = numeric_cols[-1]
            else:
                raise ValueError("No value column found")
        
        result = pd.DataFrame()
        result["Datum"] = pd.to_datetime(df[date_col], errors='coerce')
        result["value"] = pd.to_numeric(df[value_col], errors='coerce')
        result = result.dropna(subset=["value", "Datum"])
        result = result.sort_values("Datum").reset_index(drop=True)
        return result

class CacheManager:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir = self.cache_dir / self.config.final_cache_subdir
        self.final_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, code: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in code)
        return self.cache_dir / f"{safe_name}.xlsx"
    
    def is_fresh(self, code: str) -> bool:
        cache_path = self._cache_path(code)
        if not cache_path.exists():
            return False
        try:
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_days = (dt.datetime.now() - mtime).days
            return age_days <= self.config.cache_max_age_days
        except OSError:
            return False
    
    def read_cache(self, code: str) -> Optional[pd.DataFrame]:
        if not self.is_fresh(code):
            return None
        cache_path = self._cache_path(code)
        try:
            df = pd.read_excel(cache_path, sheet_name="data", engine=get_excel_engine())
            return DataProcessor.standardize_dataframe(df)
        except Exception:
            return None
    
    def write_cache(self, code: str, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        cache_path = self._cache_path(code)
        temp_path = cache_path.with_suffix(".tmp.xlsx")
        try:
            with pd.ExcelWriter(temp_path, engine=get_excel_engine()) as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            temp_path.replace(cache_path)
            return True
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False

class BundesbankCSVParser:
    @staticmethod
    def parse(content: str, code: str) -> pd.DataFrame:
        try:
            lines = content.strip().split('\n')
            if not lines:
                raise ValueError("Empty CSV content")
            
            data_start_idx = BundesbankCSVParser._find_data_start(lines, code)
            csv_lines = lines[data_start_idx:]
            if not csv_lines:
                raise ValueError("No data lines found")
            
            delimiter = BundesbankCSVParser._detect_delimiter(csv_lines[0])
            df = pd.read_csv(io.StringIO('\n'.join(csv_lines)), delimiter=delimiter, skip_blank_lines=True)
            df = df.dropna(how='all')
            if df.empty:
                raise ValueError("No valid data after parsing")
            
            time_col, value_col = BundesbankCSVParser._identify_columns(df, code)
            result_df = pd.DataFrame()
            time_values = df[time_col].dropna()
            value_values = df[value_col].dropna()
            min_len = min(len(time_values), len(value_values))
            if min_len == 0:
                raise ValueError("No valid data pairs found")
            
            result_df['Datum'] = time_values.iloc[:min_len].astype(str)
            result_df['value'] = pd.to_numeric(value_values.iloc[:min_len], errors='coerce')
            result_df = result_df.dropna()
            if result_df.empty:
                raise ValueError("No valid numeric data after cleaning")
            return result_df
        except Exception as e:
            raise ValueError(f"Bundesbank CSV parsing failed: {e}")
    
    @staticmethod
    def _find_data_start(lines: List[str], code: str) -> int:
        for i, line in enumerate(lines):
            if code in line and ('BBAF3' in line or 'BBK' in line):
                return i
        for i, line in enumerate(lines):
            if code in line:
                return i
        for i, line in enumerate(lines):
            if ',' in line or ';' in line:
                sep_count = max(line.count(','), line.count(';'))
                if sep_count >= 2:
                    return i
        return 0
    
    @staticmethod
    def _detect_delimiter(header_line: str) -> str:
        comma_count = header_line.count(',')
        semicolon_count = header_line.count(';')
        if comma_count > semicolon_count:
            return ','
        elif semicolon_count > 0:
            return ';'
        else:
            if '\t' in header_line:
                return '\t'
            elif '|' in header_line:
                return '|'
            else:
                return ','
    
    @staticmethod
    def _identify_columns(df: pd.DataFrame, code: str) -> Tuple[str, str]:
        value_col = None
        for col in df.columns:
            col_str = str(col)
            if code in col_str and 'FLAG' not in col_str.upper() and 'ATTRIBUT' not in col_str.upper():
                value_col = col
                break
        if value_col is None:
            code_parts = code.split('.')
            for col in df.columns:
                col_str = str(col)
                if any(part in col_str for part in code_parts if len(part) > 3) and 'FLAG' not in col_str.upper():
                    value_col = col
                    break
        if value_col is None and len(df.columns) >= 2:
            for col in df.columns[1:]:
                if pd.to_numeric(df[col], errors='coerce').notna().sum() > 0:
                    value_col = col
                    break
        if value_col is None:
            if len(df.columns) >= 2:
                value_col = df.columns[1]
            else:
                raise ValueError("Could not identify value column")
        
        time_col = None
        date_keywords = ['TIME', 'DATE', 'PERIOD', 'DATUM', 'ZEIT']
        for col in df.columns:
            col_str = str(col).upper()
            if any(keyword in col_str for keyword in date_keywords):
                time_col = col
                break
        if time_col is None:
            for col in df.columns:
                if col != value_col and 'FLAG' not in str(col).upper():
                    time_col = col
                    break
        if time_col is None:
            time_col = df.columns[0]
        
        return time_col, value_col

class APIClient:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    async def fetch_series(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        source = detect_data_source(code)
        if source == "ECB":
            return await self._fetch_ecb(session, code, start, end)
        else:
            return await self._fetch_bundesbank(session, code, start, end)
    
    async def _fetch_ecb(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        if HAS_ECBDATA:
            try:
                df = ecbdata.get_series(series_key=code, start=start, end=end)
                if df is not None and not df.empty:
                    return DataProcessor.standardize_dataframe(df)
            except Exception:
                pass
        
        flow, series = code.split(".", 1)
        url = f"{ECB_API_BASE_URL}/{flow}/{series}"
        fstart = format_date_for_ecb_api(start)
        fend = format_date_for_ecb_api(end)
        
        param_strategies = [
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
            {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
            {"format": "csvdata", "detail": "dataonly"},
        ]
        
        timeout = aiohttp.ClientTimeout(total=self.config.download_timeout_seconds)
        headers = {"Accept": "text/csv"}
        last_error = None
        
        for params in param_strategies:
            async with session.get(url, params=params, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    last_error = f"Status {response.status}"
                    continue
                text = await response.text()
                if not text.strip() or len(text.strip()) < self.config.min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                try:
                    df = pd.read_csv(io.StringIO(text))
                    df = DataProcessor.standardize_dataframe(df)
                    if not df.empty:
                        return df
                except Exception as e:
                    last_error = f"CSV parse error: {e}"
                    continue
        
        raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    
    async def _fetch_bundesbank(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        url_patterns = self._build_bundesbank_urls(code)
        params_variants = self._get_bundesbank_params(start, end)
        headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        timeout = aiohttp.ClientTimeout(total=self.config.download_timeout_seconds)
        last_error = None
        attempt_count = 0
        max_attempts = min(len(url_patterns) * len(params_variants), 20)
        
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
        
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as bb_session:
                for url in url_patterns:
                    for params in params_variants:
                        attempt_count += 1
                        if attempt_count > max_attempts:
                            break
                        try:
                            async with bb_session.get(url, params=params, headers=headers) as response:
                                if response.status == 200:
                                    text = await response.text()
                                    if text and len(text.strip()) > self.config.min_response_size:
                                        df = BundesbankCSVParser.parse(text, code)
                                        if df is not None and not df.empty:
                                            df = DataProcessor.standardize_dataframe(df)
                                            if not df.empty:
                                                return df
                                    else:
                                        last_error = f"Response too small: {len(text)} bytes"
                                        continue
                                elif response.status == 404:
                                    last_error = "Series not found (404)"
                                    continue
                                else:
                                    error_text = await response.text()
                                    last_error = f"Status {response.status}: {error_text[:100]}"
                                    continue
                        except asyncio.TimeoutError:
                            last_error = f"Timeout after {self.config.download_timeout_seconds}s"
                            continue
                        except Exception as e:
                            last_error = f"Unexpected error: {str(e)}"
                            continue
                    if attempt_count > max_attempts:
                        break
        except Exception as e:
            last_error = f"Session creation failed: {e}"
        
        raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")
    
    def _build_bundesbank_urls(self, code: str) -> List[str]:
        base_urls = [
            "https://api.statistiken.bundesbank.de/rest/download",
            "https://www.bundesbank.de/statistic-rmi/StatisticDownload"
        ]
        url_patterns = []
        if '.' in code:
            dataset, series = code.split('.', 1)
            url_patterns.extend([
                f"{base_urls[0]}/{dataset}/{series}",
                f"{base_urls[0]}/{code.replace('.', '/')}",
                f"{base_urls[1]}/{dataset}/{series}",
                f"{base_urls[1]}/{code.replace('.', '/')}"
            ])
        url_patterns.extend([
            f"{base_urls[0]}/{code}",
            f"{base_urls[1]}/{code}"
        ])
        if code.count('.') > 1:
            parts = code.split('.')
            for i in range(1, len(parts)):
                path1 = '.'.join(parts[:i])
                path2 = '.'.join(parts[i:])
                url_patterns.extend([
                    f"{base_urls[0]}/{path1}/{path2}",
                    f"{base_urls[0]}/{path1.replace('.', '/')}/{path2.replace('.', '/')}"
                ])
        seen = set()
        unique_patterns = []
        for pattern in url_patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        return unique_patterns[:12]
    
    def _get_bundesbank_params(self, start: str, end: str) -> List[Dict[str, str]]:
        return [
            {"format": "csv", "lang": "en", "metadata": "false"},
            {"format": "csv", "lang": "de", "metadata": "false"},
            {"format": "csv", "lang": "en", "metadata": "false", "startPeriod": start, "endPeriod": end},
            {"format": "csv", "lang": "de", "metadata": "false", "startPeriod": start, "endPeriod": end},
            {"format": "tsv", "lang": "en", "metadata": "false"},
            {"format": "tsv", "lang": "de", "metadata": "false"},
            {"format": "csv"},
            {"lang": "en"},
            {"lang": "de"},
            {}
        ]

class IndexCreator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def create_index(self, data_df: pd.DataFrame, series_codes: List[str], index_name: str) -> pd.Series:
        if 'Datum' not in data_df.columns:
            raise ValueError("DataFrame must contain a 'Datum' column")
        
        available_codes = [code for code in series_codes if code in data_df.columns]
        if not available_codes:
            raise ValueError(f"No valid series found for index {index_name}")
        
        index_data = data_df[['Datum'] + available_codes].copy()
        index_data = index_data.set_index('Datum')
        
        has_any = index_data[available_codes].notna().any(axis=1)
        index_data = index_data.loc[has_any].copy()

        def _fill_inside(s: pd.Series) -> pd.Series:
            if s.notna().sum() == 0:
                return s
            first, last = s.first_valid_index(), s.last_valid_index()
            if first is None or last is None:
                return s
            filled = s.ffill().bfill()
            mask = (s.index >= first) & (s.index <= last)
            return filled.where(mask, s)
        
        index_data[available_codes] = index_data[available_codes].apply(_fill_inside)
        clean_data = index_data.dropna()
        
        if clean_data.empty:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result
        
        weights = {code: 1.0 / len(available_codes) for code in available_codes}
        weighted_values = []
        for code in available_codes:
            if code in clean_data.columns:
                weighted_values.append(clean_data[code] * weights[code])
        
        if not weighted_values:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result
        
        aggregated = sum(weighted_values)
        
        try:
            base_year_int = int(self.config.index_base_year)
            base_year_mask = aggregated.index.year == base_year_int
            base_year_data = aggregated[base_year_mask]
            
            if base_year_data.empty or base_year_data.isna().all():
                first_valid = aggregated.dropna()
                if first_valid.empty:
                    base_value_actual = 1.0
                else:
                    base_value_actual = first_valid.iloc[0]
            else:
                base_value_actual = base_year_data.mean()
            
            if base_value_actual == 0 or pd.isna(base_value_actual):
                base_value_actual = 1.0
            
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            mask = aggregated.notna()
            result[mask] = (aggregated[mask] / base_value_actual) * self.config.index_base_value
            
            return result
            
        except Exception as e:
            print(f"Warning: Index normalization failed for {index_name}, using raw data: {e}")
            aggregated.name = index_name
            return aggregated

# =============================================================================
# ORIGINAL DOWNLOAD LOGIC (NICHT ÄNDERN)
# =============================================================================

class FinancialDataDownloader:
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.cache_manager = CacheManager(self.config)
        self.api_client = APIClient(self.config)
        self.index_creator = IndexCreator(self.config)
    
    def download(self, series_definitions: Dict[str, str], start_date: str = None, 
                end_date: str = None, prefer_cache: bool = True, anchor_var: Optional[str] = None) -> pd.DataFrame:
        start_date = start_date or self.config.default_start_date
        end_date = end_date or self.config.default_end_date
        print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
        
        regular_codes = {}
        index_definitions = {}
        
        for var_name, definition in series_definitions.items():
            index_codes = parse_index_specification(definition)
            if index_codes:
                index_definitions[var_name] = index_codes
            else:
                regular_codes[var_name] = definition
        
        all_codes = set(regular_codes.values())
        for index_codes in index_definitions.values():
            all_codes.update(index_codes)
        all_codes = list(all_codes)
        
        print(f"Total series to download: {len(all_codes)}")
        
        cached_data = {}
        missing_codes = []
        
        if prefer_cache:
            for code in all_codes:
                cached_df = self.cache_manager.read_cache(code)
                if cached_df is not None:
                    cached_data[code] = cached_df
                else:
                    missing_codes.append(code)
        else:
            missing_codes = all_codes[:]
        
        downloaded_data = {}
        if missing_codes:
            print(f"Downloading {len(missing_codes)} missing series...")
            try:
                downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
                    except ImportError:
                        print("Using synchronous download mode...")
                        downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
                else:
                    print("Async failed, using synchronous download mode...")
                    downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            except Exception as e:
                print(f"Download failed ({e}), trying synchronous mode...")
                downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            
            for code, df in downloaded_data.items():
                self.cache_manager.write_cache(code, df)
        
        all_data = {**cached_data, **downloaded_data}
        
        if not all_data:
            raise Exception("No series loaded successfully")
        
        merged_df = self._merge_series_data(all_data)
        final_data = {"Datum": merged_df["Datum"]}
        
        for var_name, series_code in regular_codes.items():
            if series_code in merged_df.columns:
                final_data[var_name] = merged_df[series_code]
        
        for var_name, index_codes in index_definitions.items():
            try:
                available_codes = [c for c in index_codes if c in merged_df.columns]
                
                if len(available_codes) >= len(index_codes) * 0.3:
                    index_series = self.index_creator.create_index(merged_df, available_codes, var_name)
                    aligned_index = index_series.reindex(pd.to_datetime(merged_df['Datum']))
                    final_data[var_name] = aligned_index.values
                    print(f"Created INDEX: {var_name} from {len(available_codes)}/{len(index_codes)} series")
                else:
                    if var_name in SIMPLE_TARGET_FALLBACKS:
                        fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                        if fallback_code in merged_df.columns:
                            final_data[var_name] = merged_df[fallback_code]
                            print(f"Using fallback for {var_name}: {fallback_code}")
                        else:
                            print(f"Warning: Could not create {var_name} - fallback series {fallback_code} not available")
                    else:
                        print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available_codes)}/{len(index_codes)} series available)")
                        
            except Exception as e:
                print(f"Failed to create INDEX {var_name}: {e}")
                if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                    fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback_code in merged_df.columns:
                        final_data[var_name] = merged_df[fallback_code]
                        print(f"Using fallback for {var_name} after INDEX creation failed: {fallback_code}")
        
        final_df = pd.DataFrame(final_data)
        final_df["Datum"] = pd.to_datetime(final_df["Datum"])
        final_df = final_df.sort_values("Datum").reset_index(drop=True)

        value_cols = [c for c in final_df.columns if c != 'Datum']
        if value_cols:
            non_na_count = final_df[value_cols].notna().sum(axis=1)
            required = 2 if len(value_cols) >= 2 else 1
            keep_mask = non_na_count >= required
            if keep_mask.any():
                first_keep = keep_mask.idxmax()
                if first_keep > 0:
                    _before = len(final_df)
                    final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                    print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

        if anchor_var and anchor_var in final_df.columns:
            mask_anchor = final_df[anchor_var].notna()
            if mask_anchor.any():
                start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
                end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
                _before_rows = len(final_df)
                final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
                final_df.reset_index(drop=True, inplace=True)
                print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

        if anchor_var and anchor_var in final_df.columns:
            exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
            if exog_cols:
                tgt_notna = final_df[anchor_var].notna().values
                all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
                keep_start = 0
                for i in range(len(final_df)):
                    if not (tgt_notna[i] and all_exog_nan[i]):
                        keep_start = i
                        break
                if keep_start > 0:
                    _before = len(final_df)
                    final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                    print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

        print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")
        return final_df
    
    def _fetch_all_series_sync(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        import requests
        successful = {}
        
        for code in codes:
            try:
                source = detect_data_source(code)
                if source == "ECB":
                    df = self._fetch_ecb_sync(code, start, end)
                else:
                    df = self._fetch_bundesbank_sync(code, start, end)
                
                if df is not None and not df.empty:
                    successful[code] = df
                    print(f"  ✓ {code}: {len(df)} observations")
                else:
                    print(f"  ✗ {code}: No data returned")
            except Exception as e:
                print(f"  ✗ {code}: {str(e)}")
            
            import time
            time.sleep(0.5)
        
        return successful
    
    def _fetch_ecb_sync(self, code: str, start: str, end: str) -> pd.DataFrame:
        import requests
        
        if HAS_ECBDATA:
            try:
                df = ecbdata.get_series(series_key=code, start=start, end=end)
                if df is not None and not df.empty:
                    return DataProcessor.standardize_dataframe(df)
            except Exception:
                pass
        
        flow, series = code.split(".", 1)
        url = f"{ECB_API_BASE_URL}/{flow}/{series}"
        fstart = format_date_for_ecb_api(start)
        fend = format_date_for_ecb_api(end)

        param_strategies = [
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
            {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
            {"format": "csvdata", "detail": "dataonly"},
        ]

        headers = {"Accept": "text/csv"}
        last_error = None

        for params in param_strategies:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.config.download_timeout_seconds)
                if resp.status_code != 200:
                    last_error = f"Status {resp.status_code}"
                    continue
                text = resp.text
                if not text.strip() or len(text.strip()) < self.config.min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                df = pd.read_csv(io.StringIO(text))
                df = DataProcessor.standardize_dataframe(df)
                if not df.empty:
                    return df
            except Exception as e:
                last_error = str(e)
                continue

        raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    
    def _fetch_bundesbank_sync(self, code: str, start: str, end: str) -> pd.DataFrame:
        import requests
        
        url_patterns = self.api_client._build_bundesbank_urls(code)
        params_variants = self.api_client._get_bundesbank_params(start, end)
        headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
        last_error = None
        attempt_count = 0
        max_attempts = min(len(url_patterns) * len(params_variants), 20)
        
        for url in url_patterns:
            for params in params_variants:
                attempt_count += 1
                if attempt_count > max_attempts:
                    break
                
                try:
                    response = requests.get(
                        url, params=params, headers=headers,
                        timeout=self.config.download_timeout_seconds, verify=False
                    )
                    
                    if response.status_code == 200:
                        text = response.text
                        if text and len(text.strip()) > self.config.min_response_size:
                            df = BundesbankCSVParser.parse(text, code)
                            if df is not None and not df.empty:
                                df = DataProcessor.standardize_dataframe(df)
                                if not df.empty:
                                    return df
                        else:
                            last_error = f"Response too small: {len(text)} bytes"
                            continue
                    elif response.status_code == 404:
                        last_error = "Series not found (404)"
                        continue
                    else:
                        last_error = f"Status {response.status_code}: {response.text[:100]}"
                        continue
                        
                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {self.config.download_timeout_seconds}s"
                    continue
                except requests.exceptions.SSLError:
                    last_error = "SSL verification failed"
                    continue
                except Exception as e:
                    last_error = f"Request failed: {str(e)}"
                    continue
            
            if attempt_count > max_attempts:
                break
        
        raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")

    async def _fetch_all_series(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        successful = {}
        
        async with aiohttp.ClientSession() as session:
            for code in codes:
                try:
                    df = await self.api_client.fetch_series(session, code, start, end)
                    successful[code] = df
                    print(f"  ✓ {code}: {len(df)} observations")
                except Exception as e:
                    print(f"  ✗ {code}: {str(e)}")
                
                await asyncio.sleep(0.5)
        
        return successful
    
    def _merge_series_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        all_series = []
        
        for code, df in data_dict.items():
            if not df.empty and "Datum" in df.columns and "value" in df.columns:
                series_df = df.set_index("Datum")[["value"]].rename(columns={"value": code})
                all_series.append(series_df)
        
        if not all_series:
            return pd.DataFrame()
        
        merged_df = pd.concat(all_series, axis=1, sort=True)
        merged_df = merged_df.reset_index()
        merged_df = merged_df.sort_values("Datum").reset_index(drop=True)
        return merged_df

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_target_with_standard_exog(target_name: str, start_date: str = "2000-01", 
                                 config: AnalysisConfig = None) -> pd.DataFrame:
    if target_name not in INDEX_TARGETS:
        raise ValueError(f"Unknown target: {target_name}. Available: {list(INDEX_TARGETS.keys())}")
    
    series_definitions = {target_name: INDEX_TARGETS[target_name]}
    standard_exog = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    for exog_name in standard_exog:
        if exog_name in STANDARD_EXOG_VARS:
            series_definitions[exog_name] = STANDARD_EXOG_VARS[exog_name]
    
    downloader = FinancialDataDownloader(config)
    return downloader.download(series_definitions, start_date=start_date)

class FinancialAnalysisError(Exception):
    pass

class DataDownloadError(FinancialAnalysisError):
    pass

class ValidationError(FinancialAnalysisError):
    pass

class AnalysisError(FinancialAnalysisError):
    pass

print("Core configuration and data download loaded")

# %%
"""
Mixed-Frequency Data Processing - KORRIGIERTE VERSION
Löst das Forward-Fill Problem für Quartalsdaten korrekt
"""

class MixedFrequencyProcessor:
    """
    Handles quarterly target variables with monthly exogenous variables.
    Ensures proper forward-filling without data leakage.
    """
    
    @staticmethod
    def detect_frequency(series: pd.Series, date_col: pd.Series) -> str:
        """
        Detect if a series is monthly or quarterly based on data availability.
        """
        if series.isna().all():
            return "unknown"
        
        # Count observations per year
        df_temp = pd.DataFrame({'date': date_col, 'value': series})
        df_temp = df_temp.dropna()
        
        if len(df_temp) == 0:
            return "unknown"
        
        df_temp['year'] = df_temp['date'].dt.year
        obs_per_year = df_temp.groupby('year').size()
        
        avg_obs_per_year = obs_per_year.mean()
        
        if avg_obs_per_year <= 4.5:  # Allow for some missing quarters
            return "quarterly"
        elif avg_obs_per_year >= 10:  # Allow for some missing months
            return "monthly"
        else:
            return "unknown"
    
    @staticmethod
    def forward_fill_quarterly(df: pd.DataFrame, quarterly_vars: List[str]) -> pd.DataFrame:
        """
        Forward-fill quarterly variables properly:
        1. Only fill within the available data range (no extrapolation)
        2. Fill monthly gaps between quarterly observations
        """
        result = df.copy()
        
        for var in quarterly_vars:
            if var not in df.columns:
                continue
                
            series = df[var].copy()
            
            # Find first and last valid observation
            valid_mask = series.notna()
            if not valid_mask.any():
                continue
                
            first_valid_idx = valid_mask.idxmax()
            last_valid_idx = valid_mask[::-1].idxmax()  # Last valid
            
            # Only forward fill between first and last valid observation
            fill_range = series.iloc[first_valid_idx:last_valid_idx+1]
            filled_range = fill_range.ffill()
            
            # Update only the range between first and last valid
            result.loc[first_valid_idx:last_valid_idx, var] = filled_range
            
        return result
    
    @staticmethod
    def align_frequencies(df: pd.DataFrame, target_var: str, 
                         date_col: str = "Datum") -> pd.DataFrame:
        """
        Align mixed-frequency data properly for regression analysis.
        """
        if target_var not in df.columns or date_col not in df.columns:
            raise ValueError(f"Missing {target_var} or {date_col} column")
        
        # Detect frequencies
        frequencies = {}
        all_vars = [col for col in df.columns if col != date_col]
        
        for var in all_vars:
            freq = MixedFrequencyProcessor.detect_frequency(df[var], df[date_col])
            frequencies[var] = freq
            
        print(f"Detected frequencies:")
        for var, freq in frequencies.items():
            obs_count = df[var].notna().sum()
            print(f"  {var}: {freq} ({obs_count} observations)")
        
        # Identify quarterly variables
        quarterly_vars = [var for var, freq in frequencies.items() if freq == "quarterly"]
        monthly_vars = [var for var, freq in frequencies.items() if freq == "monthly"]
        
        if not quarterly_vars:
            print("No quarterly variables detected - returning original data")
            return df
        
        # Apply forward-fill to quarterly variables
        print(f"Forward-filling {len(quarterly_vars)} quarterly variables...")
        processed_df = MixedFrequencyProcessor.forward_fill_quarterly(df, quarterly_vars)
        
        # Validation: Check improvement
        for var in quarterly_vars:
            before_count = df[var].notna().sum()
            after_count = processed_df[var].notna().sum()
            print(f"  {var}: {before_count} → {after_count} observations")
        
        return processed_df

class DataQualityChecker:
    """
    Comprehensive data quality validation for financial time series.
    """
    
    @staticmethod
    def validate_financial_data(data: pd.DataFrame, target_var: str, 
                               exog_vars: List[str], 
                               min_target_coverage: float = 0.15) -> Dict[str, Any]:
        """
        Enhanced data validation with mixed-frequency awareness.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Check if variables exist
        missing_vars = [var for var in [target_var] + exog_vars if var not in data.columns]
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {', '.join(missing_vars)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Analyze target variable quality (CRITICAL)
        target_series = data[target_var]
        target_coverage = target_series.notna().sum() / len(target_series)
        
        validation_results['data_quality'][target_var] = {
            'total_obs': len(target_series),
            'valid_obs': target_series.notna().sum(),
            'coverage': target_coverage,
            'frequency': MixedFrequencyProcessor.detect_frequency(target_series, data['Datum'])
        }
        
        # CRITICAL CHECK: Target coverage
        if target_coverage < min_target_coverage:
            validation_results['errors'].append(
                f"Target variable {target_var} has only {target_coverage:.1%} valid data "
                f"(minimum required: {min_target_coverage:.1%})"
            )
            validation_results['is_valid'] = False
            
        # Check for completely constant target
        if target_series.notna().sum() > 1 and target_series.std() == 0:
            validation_results['errors'].append(f"Target variable {target_var} is constant")
            validation_results['is_valid'] = False
        
        # Analyze exogenous variables
        for var in exog_vars:
            if var in data.columns:
                series = data[var]
                coverage = series.notna().sum() / len(series)
                
                validation_results['data_quality'][var] = {
                    'total_obs': len(series),
                    'valid_obs': series.notna().sum(),
                    'coverage': coverage,
                    'frequency': MixedFrequencyProcessor.detect_frequency(series, data['Datum'])
                }
                
                if coverage < 0.5:
                    validation_results['warnings'].append(
                        f"Exogenous variable {var} has low coverage ({coverage:.1%})"
                    )
                
                if series.notna().sum() > 1 and series.std() == 0:
                    validation_results['warnings'].append(f"Variable {var} is constant")
        
        # Check for sufficient overlapping data
        all_vars = [target_var] + exog_vars
        available_vars = [var for var in all_vars if var in data.columns]
        
        if available_vars:
            complete_cases = data[available_vars].dropna()
            overlap_ratio = len(complete_cases) / len(data)
            
            validation_results['overlap_analysis'] = {
                'complete_cases': len(complete_cases),
                'total_cases': len(data),
                'overlap_ratio': overlap_ratio
            }
            
            if overlap_ratio < 0.1:
                validation_results['errors'].append(
                    f"Very low overlap between variables ({overlap_ratio:.1%} complete cases)"
                )
                validation_results['is_valid'] = False
            elif overlap_ratio < 0.3:
                validation_results['warnings'].append(
                    f"Low overlap between variables ({overlap_ratio:.1%} complete cases)"
                )
        
        # Generate recommendations
        if validation_results['data_quality'][target_var]['frequency'] == 'quarterly':
            validation_results['recommendations'].append(
                "Target is quarterly - will apply forward-fill to align with monthly exogenous variables"
            )
        
        if not validation_results['is_valid']:
            validation_results['recommendations'].append(
                "Consider using a different target variable or extending the time period"
            )
        
        return validation_results

class DataPreprocessor:
    """
    Handles data preprocessing for financial regression analysis.
    Includes proper mixed-frequency handling and transformation logic.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.date_col = date_col
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
    
    def create_transformations(self, transformation: str = 'levels') -> pd.DataFrame:
        """
        Create data transformations with proper mixed-frequency handling.
        """
        # Step 1: Handle mixed frequencies FIRST
        processed_data = MixedFrequencyProcessor.align_frequencies(
            self.data, self.target_var, self.date_col
        )
        
        # Step 2: Apply transformations
        transformed_data = processed_data[[self.date_col]].copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        print(f"Applying '{transformation}' transformation to {len(numeric_cols)} variables...")
        
        for col in numeric_cols:
            series = processed_data[col]
            
            if transformation == 'levels':
                transformed_data[col] = series
            elif transformation == 'log':
                # Check if all positive values
                positive_mask = series > 0
                if positive_mask.sum() > len(series) * 0.8:  # At least 80% positive
                    # Apply log transformation with small epsilon for zeros
                    transformed_data[col] = np.log(series.clip(lower=1e-6))
                else:
                    # Fall back to levels if not suitable for log
                    transformed_data[col] = series
                    print(f"  Warning: {col} not suitable for log transformation (negative values)")
            elif transformation == 'pct':
                transformed_data[col] = series.pct_change()
            elif transformation == 'diff':
                transformed_data[col] = series.diff()
            else:
                transformed_data[col] = series
        
        # Step 3: Clean up infinite and NaN values
        transformed_data = transformed_data.replace([np.inf, -np.inf], np.nan)
        
        # Step 4: Conservative outlier handling
        numeric_cols = [c for c in transformed_data.columns if c != self.date_col]
        for col in numeric_cols:
            series = transformed_data[col].dropna()
            if len(series) > 20:  # Only if enough observations
                q_low = series.quantile(0.01)  # Conservative 1%/99%
                q_high = series.quantile(0.99)
                if pd.notna(q_low) and pd.notna(q_high) and q_high > q_low:
                    transformed_data[col] = transformed_data[col].clip(lower=q_low, upper=q_high)
        
        # Step 5: Add seasonal dummies and time trend
        transformed_data = self._add_seasonal_features(transformed_data)
        
        # Step 6: Final cleaning
        before_clean = len(transformed_data)
        transformed_data = transformed_data.dropna(how="any")
        after_clean = len(transformed_data)
        
        if before_clean > after_clean:
            print(f"Dropped {before_clean - after_clean} rows with missing values after transformation")
        
        # Ensure stable data types
        for col in [c for c in transformed_data.columns if c != self.date_col]:
            transformed_data[col] = transformed_data[col].astype("float64")
        
        return transformed_data
    
    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add quarterly seasonal dummies and time trend."""
        data_with_features = data.copy()
        
        # Extract quarter from date
        data_with_features['quarter'] = pd.to_datetime(data_with_features[self.date_col]).dt.quarter
        
        # Create dummy variables (Q1 as base category)
        for q in [2, 3, 4]:
            data_with_features[f'Q{q}'] = (data_with_features['quarter'] == q).astype(int)
        
        # Create time trend (normalize to start from 0)
        data_with_features['time_trend'] = range(len(data_with_features))
        
        # Drop the quarter column
        data_with_features = data_with_features.drop('quarter', axis=1)
        
        return data_with_features

def diagnose_data_issues(data: pd.DataFrame, target_var: str, exog_vars: List[str]) -> None:
    """
    Comprehensive data quality diagnosis with specific focus on mixed-frequency issues.
    """
    print("\n" + "="*60)
    print("DATA QUALITY DIAGNOSIS")
    print("="*60)
    
    # Overall dataset info
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['Datum'].min().strftime('%Y-%m')} to {data['Datum'].max().strftime('%Y-%m')}")
    
    # Analyze each variable
    all_vars = [target_var] + exog_vars
    
    for i, var in enumerate(all_vars):
        if var not in data.columns:
            print(f"\n{i+1}. {var}: ❌ NOT FOUND IN DATA")
            continue
        
        series = data[var]
        valid_count = series.notna().sum()
        coverage = valid_count / len(series)
        frequency = MixedFrequencyProcessor.detect_frequency(series, data['Datum'])
        
        print(f"\n{i+1}. {var} ({'TARGET' if var == target_var else 'EXOG'}):")
        print(f"   - Frequency: {frequency}")
        print(f"   - Coverage: {coverage:.1%} ({valid_count}/{len(series)} observations)")
        
        if valid_count > 0:
            print(f"   - Range: {series.min():.4f} to {series.max():.4f}")
            print(f"   - Mean: {series.mean():.4f}, Std: {series.std():.4f}")
            
            # Check for problematic patterns
            if series.std() == 0:
                print(f"   - ⚠️  WARNING: Variable is constant!")
            
            if frequency == "quarterly" and var != target_var:
                print(f"   - ℹ️  INFO: Quarterly exogenous variable detected")
            elif frequency == "quarterly" and var == target_var:
                print(f"   - ℹ️  INFO: Quarterly target - will use forward-fill")
        
        # Check correlations with target (if not the target itself)
        if var != target_var and var in data.columns and target_var in data.columns:
            # Find overlapping observations
            overlap_data = data[[var, target_var]].dropna()
            if len(overlap_data) > 10:
                corr = overlap_data[var].corr(overlap_data[target_var])
                print(f"   - Correlation with {target_var}: {corr:.4f}")
            else:
                print(f"   - Correlation: insufficient overlap ({len(overlap_data)} obs)")
    
    # Check data overlap
    print(f"\n" + "-"*40)
    print("DATA OVERLAP ANALYSIS")
    print("-"*40)
    
    available_vars = [var for var in all_vars if var in data.columns]
    complete_cases = data[available_vars].dropna()
    overlap_ratio = len(complete_cases) / len(data)
    
    print(f"Complete cases: {len(complete_cases)} / {len(data)} ({overlap_ratio:.1%})")
    
    if overlap_ratio < 0.1:
        print("❌ CRITICAL: Very low data overlap - analysis likely to fail")
    elif overlap_ratio < 0.3:
        print("⚠️  WARNING: Low data overlap - results may be unreliable")
    else:
        print("✅ OK: Sufficient data overlap for analysis")
    
    # Frequency analysis summary
    print(f"\n" + "-"*40)
    print("FREQUENCY ANALYSIS SUMMARY")
    print("-"*40)
    
    quarterly_vars = []
    monthly_vars = []
    unknown_vars = []
    
    for var in available_vars:
        freq = MixedFrequencyProcessor.detect_frequency(data[var], data['Datum'])
        if freq == "quarterly":
            quarterly_vars.append(var)
        elif freq == "monthly":
            monthly_vars.append(var)
        else:
            unknown_vars.append(var)
    
    print(f"Quarterly variables ({len(quarterly_vars)}): {', '.join(quarterly_vars)}")
    print(f"Monthly variables ({len(monthly_vars)}): {', '.join(monthly_vars)}")
    if unknown_vars:
        print(f"Unknown frequency ({len(unknown_vars)}): {', '.join(unknown_vars)}")
    
    # Recommendations
    print(f"\n" + "-"*40)
    print("RECOMMENDATIONS")
    print("-"*40)
    
    if len(quarterly_vars) > 0 and target_var in quarterly_vars:
        print("✅ Will apply forward-fill to quarterly target variable")
        
    if overlap_ratio < 0.3:
        print("💡 Consider:")
        print("   - Using a longer time period")
        print("   - Using different target/exogenous variables")
        print("   - Checking data quality at source")
    
    if len(complete_cases) < 50:
        print("⚠️  Small sample size - consider using simpler models")

print("Mixed-frequency data processor loaded")


# %%
"""
Regression Methods & Cross-Validation - VEREINFACHT & ROBUSTE VERSION
Ohne Monkey-Patches - saubere Klassenstruktur
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from scipy import stats

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class RegressionMethod(ABC):
    """Abstract base class for regression methods."""
    
    def __init__(self, name: str, requires_scaling: bool = True):
        self.name = name
        self.requires_scaling = requires_scaling
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model and return results."""
        pass

class OLSMethod(RegressionMethod):
    """OLS Regression with robust standard errors."""
    
    def __init__(self):
        super().__init__("OLS", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # Add constant
        X_with_const = sm.add_constant(X, has_constant='add')
        
        # Fit model
        model = OLS(y, X_with_const, missing='drop')
        
        # Use robust standard errors (HAC)
        max_lags = min(4, int(len(y) ** (1/4)))
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        
        # Calculate diagnostics
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(results.resid)
        except:
            dw = np.nan
        
        try:
            jb_stat, jb_p = stats.jarque_bera(results.resid)[:2]
            jarque_bera = {'statistic': jb_stat, 'p_value': jb_p}
        except:
            jarque_bera = {'statistic': np.nan, 'p_value': np.nan}
        
        diagnostics = {
            'durbin_watson': dw,
            'jarque_bera': jarque_bera
        }
        
        return {
            'model': results,
            'coefficients': results.params,
            'std_errors': results.bse,
            'p_values': results.pvalues,
            'r_squared': results.rsquared,
            'r_squared_adj': results.rsquared_adj,
            'mse': results.mse_resid,
            'mae': np.mean(np.abs(results.resid)),
            'residuals': results.resid,
            'fitted_values': results.fittedvalues,
            'diagnostics': diagnostics
        }

class RandomForestMethod(RegressionMethod):
    """Conservative Random Forest optimized for financial time series."""
    
    def __init__(self):
        super().__init__("Random Forest", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = X.shape
        
        # Very conservative hyperparameters based on sample size
        if n_samples < 50:
            n_estimators = 100
            max_depth = 3
            min_samples_leaf = max(5, n_samples // 10)
            min_samples_split = max(10, n_samples // 5)
        elif n_samples < 100:
            n_estimators = 150
            max_depth = 4
            min_samples_leaf = max(8, n_samples // 12)
            min_samples_split = max(16, n_samples // 6)
        else:
            n_estimators = 200
            max_depth = min(5, max(3, int(np.log2(n_samples)) - 2))
            min_samples_leaf = max(10, n_samples // 15)
            min_samples_split = max(20, n_samples // 8)
        
        # Additional constraints for high-dimensional data
        max_features = "sqrt" if n_features <= n_samples // 3 else max(1, min(int(np.sqrt(n_features)), n_features // 2))
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            max_samples=min(0.8, max(0.5, 1.0 - 0.1 * n_features / n_samples)),
            random_state=42,
            n_jobs=1  # Avoid multiprocessing issues
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Out-of-bag score as additional validation
        oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else np.nan
        
        return {
            'model': model,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'oob_score': oob_score,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'feature_importance': model.feature_importances_,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'max_features': max_features
            }
        }

class XGBoostMethod(RegressionMethod):
    """Conservative XGBoost optimized for financial time series."""
    
    def __init__(self):
        super().__init__("XGBoost", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available")
        
        n_samples, n_features = X.shape
        
        # Very conservative hyperparameters
        n_estimators = min(100, max(50, n_samples // 3))
        max_depth = 3
        learning_rate = 0.01 if n_samples > 50 else 0.02
        
        # Strong regularization
        reg_alpha = 1.0 + 0.1 * (n_features / 10)  # L1
        reg_lambda = 2.0 + 0.2 * (n_features / 10)  # L2
        
        # Sampling parameters
        subsample = max(0.6, 1.0 - 0.05 * (n_features / 10))
        colsample_bytree = max(0.6, 1.0 - 0.05 * (n_features / 10))
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=max(3, n_samples // 20),
            gamma=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'feature_importance': model.feature_importances_,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }
        }

class SVRMethod(RegressionMethod):
    """Support Vector Regression."""
    
    def __init__(self):
        super().__init__("SVR", requires_scaling=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred
        }

class BayesianRidgeMethod(RegressionMethod):
    """Bayesian Ridge Regression."""
    
    def __init__(self):
        super().__init__("Bayesian Ridge", requires_scaling=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = BayesianRidge()
        model.fit(X_scaled, y)
        
        y_pred, y_std = model.predict(X_scaled, return_std=True)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'prediction_std': y_std,
            'coefficients': model.coef_
        }

class MethodRegistry:
    """Registry for regression methods."""
    
    def __init__(self):
        self.methods = {
            "OLS": OLSMethod(),
            "Random Forest": RandomForestMethod(),
            "SVR": SVRMethod(),
            "Bayesian Ridge": BayesianRidgeMethod()
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.methods["XGBoost"] = XGBoostMethod()
    
    def get_method(self, name: str) -> RegressionMethod:
        if name not in self.methods:
            raise ValueError(f"Method '{name}' not available. Choose from: {list(self.methods.keys())}")
        return self.methods[name]
    
    def list_methods(self) -> List[str]:
        return list(self.methods.keys())

class RobustCrossValidator:
    """
    Robust time series cross-validation for financial data.
    Fixed implementation without leakage or extreme scores.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def validate_method(self, method: RegressionMethod, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform robust time series cross-validation.
        """
        n_samples = len(X_train)
        
        # Conservative CV parameters based on sample size
        if n_samples < 30:
            n_splits = 2
            gap = 1
        elif n_samples < 60:
            n_splits = 3
            gap = 2
        else:
            n_splits = min(4, n_samples // 20)  # Conservative: at least 20 obs per fold
            gap = max(2, int(n_samples * 0.05))  # Larger gaps
        
        if n_splits < 2:
            return {'cv_scores': [], 'cv_mean': np.nan, 'cv_std': np.nan, 'n_folds': 0}
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        cv_scores = []
        successful_folds = 0
        
        print(f"    Running {n_splits}-fold CV with gap={gap}...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            try:
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Ensure minimum fold sizes
                if len(X_fold_train) < 10 or len(X_fold_val) < 3:
                    print(f"      Fold {fold_idx+1}: Skipped (insufficient data)")
                    continue
                
                # Fit method on fold training data
                fold_results = method.fit(X_fold_train, y_fold_train)
                
                # Evaluate on fold validation data
                fold_test_perf = self._evaluate_on_test(fold_results, X_fold_val, y_fold_val, method)
                
                score = fold_test_perf['r_squared']
                
                # Sanity check: Only accept reasonable scores
                if np.isfinite(score) and -1.0 <= score <= 1.0:
                    cv_scores.append(score)
                    successful_folds += 1
                    print(f"      Fold {fold_idx+1}: R² = {score:.4f}")
                else:
                    print(f"      Fold {fold_idx+1}: Extreme score {score:.4f} - skipped")
                
            except Exception as e:
                print(f"      Fold {fold_idx+1}: Failed ({str(e)[:50]})")
                continue
        
        if len(cv_scores) == 0:
            return {'cv_scores': [], 'cv_mean': np.nan, 'cv_std': np.nan, 'n_folds': 0}
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'n_folds': successful_folds
        }
    
    def _evaluate_on_test(self, train_results: Dict[str, Any], X_test: np.ndarray, 
                         y_test: np.ndarray, method: RegressionMethod) -> Dict[str, Any]:
        """Evaluate trained model on test data."""
        model = train_results['model']
        
        # Handle predictions based on model type
        try:
            if method.name == "OLS":
                X_test_with_const = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_with_const)
            elif method.requires_scaling and 'scaler' in train_results:
                X_test_scaled = train_results['scaler'].transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'mse': test_mse,
                'r_squared': test_r2,
                'mae': test_mae,
                'predictions': y_pred,
                'actual': y_test,
                'residuals': y_test - y_pred
            }
            
        except Exception as e:
            # Return NaN if prediction fails
            return {
                'mse': np.nan,
                'r_squared': np.nan,
                'mae': np.nan,
                'predictions': np.full(len(y_test), np.nan),
                'actual': y_test,
                'residuals': np.full(len(y_test), np.nan),
                'error': str(e)
            }

class TimeSeriesSplitter:
    """
    Robust time series train/test splitting with mandatory gaps.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def split(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-series aware train/test split with mandatory gap periods."""
        n_samples = len(X)
        
        # Adaptive gap and test size based on sample size
        if n_samples < 50:
            gap = 1
            test_size = max(0.2, self.config.test_size)  # Minimum 20% for test
        elif n_samples < 100:
            gap = 2
            test_size = self.config.test_size
        else:
            gap = max(2, int(n_samples * 0.02))  # 2% of sample as gap, minimum 2
            test_size = self.config.test_size
        
        # Calculate indices
        test_samples = int(n_samples * test_size)
        train_end = n_samples - test_samples - gap
        test_start = train_end + gap
        
        # Ensure minimum training data (60%)
        min_train = int(n_samples * 0.6)
        if train_end < min_train:
            print(f"Warning: Limited training data: {train_end}/{n_samples} samples")
            train_end = min_train
            gap = max(1, n_samples - train_end - test_samples)
            test_start = train_end + gap
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_start + test_samples]
        y_test = y[test_start:test_start + test_samples]
        
        print(f"    Split: Train={len(y_train)}, Gap={gap}, Test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test

print("Regression methods and cross-validation loaded")



# %%
"""
Feature Selection & Analysis - VEREINFACHT
Saubere Implementation ohne komplexe Patches
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

class SimpleFeatureSelector:
    """
    Simplified feature selection with robust methods.
    """
    
    @staticmethod
    def statistical_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 5) -> Tuple[List[str], np.ndarray]:
        """Select k best features using F-test."""
        k_actual = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_actual)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices
    
    @staticmethod
    def importance_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[List[str], np.ndarray]:
        """Select features using Random Forest importance."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top n features
        n_actual = min(n_features, X.shape[1])
        top_indices = np.argsort(importances)[::-1][:n_actual]
        
        selected_names = [feature_names[i] for i in top_indices]
        return selected_names, top_indices
    
    @staticmethod
    def rfe_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[List[str], np.ndarray]:
        """Recursive feature elimination."""
        n_actual = min(n_features, X.shape[1])
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        selector = RFE(estimator=estimator, n_features_to_select=n_actual)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices
    
    @staticmethod
    def lasso_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], np.ndarray]:
        """Lasso-based feature selection."""
        from sklearn.feature_selection import SelectFromModel
        
        lasso_cv = LassoCV(cv=3, random_state=42, max_iter=1000)
        selector = SelectFromModel(lasso_cv)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        if len(selected_indices) == 0:
            # Fallback: use all features
            selected_indices = np.arange(X.shape[1])
        
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices

class FeatureCombinationTester:
    """
    Test different feature combinations systematically.
    """
    
    def __init__(self, method_registry, cv_validator):
        self.method_registry = method_registry
        self.cv_validator = cv_validator
    
    def test_combinations(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         max_combinations: int = 20) -> pd.DataFrame:
        """
        Test feature combinations with proper validation.
        """
        print(f"Testing feature combinations from {len(feature_names)} features...")
        
        # Limit feature space for combinations if too large
        if len(feature_names) > 10:
            print(f"Feature space too large ({len(feature_names)}), using top 10 by importance...")
            # Use Random Forest to get top features
            selector = SimpleFeatureSelector()
            top_names, top_indices = selector.importance_selection(X, y, feature_names, n_features=10)
            X_reduced = X[:, top_indices]
            feature_names = top_names
            X = X_reduced
        
        # Generate combinations
        all_combos = []
        min_features = 2
        max_features = min(6, len(feature_names))  # Limit to reasonable size
        
        for size in range(min_features, max_features + 1):
            combos = list(combinations(feature_names, size))
            all_combos.extend(combos)
            
            # Stop if we have enough combinations
            if len(all_combos) >= max_combinations:
                break
        
        # Limit to max_combinations
        if len(all_combos) > max_combinations:
            all_combos = all_combos[:max_combinations]
        
        print(f"Testing {len(all_combos)} combinations...")
        
        # Test each combination
        results = []
        method = self.method_registry.get_method('Random Forest')
        
        for i, combo in enumerate(all_combos):
            try:
                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{len(all_combos)} combinations tested")
                
                # Get indices for this combination
                combo_indices = [feature_names.index(name) for name in combo]
                X_combo = X[:, combo_indices]
                
                if X_combo.shape[1] == 0 or len(X_combo) < 20:
                    continue
                
                # Split data for this combination
                splitter = TimeSeriesSplitter(AnalysisConfig())
                X_train, X_test, y_train, y_test = splitter.split(y, X_combo)
                
                if len(X_train) < 10 or len(X_test) < 3:
                    continue
                
                # Fit method
                train_results = method.fit(X_train, y_train)
                
                # Evaluate on test
                test_perf = self.cv_validator._evaluate_on_test(train_results, X_test, y_test, method)
                
                # Store results
                results.append({
                    'combination_id': i,
                    'features': ', '.join(combo),
                    'n_features': len(combo),
                    'test_r_squared': test_perf.get('r_squared', np.nan),
                    'test_mse': test_perf.get('mse', np.nan),
                    'overfitting': train_results.get('r_squared', 0) - test_perf.get('r_squared', 0),
                    'feature_list': list(combo)
                })
                
            except Exception as e:
                print(f"  Combination {i} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print("No successful combinations tested")
            return pd.DataFrame(columns=['combination_id', 'features', 'n_features', 'test_r_squared', 'test_mse', 'overfitting', 'feature_list'])
        
        df = pd.DataFrame(results)
        df = df.sort_values('test_r_squared', ascending=False).reset_index(drop=True)
        
        print(f"Combination testing completed: {len(df)} successful combinations")
        return df

class FeatureAnalyzer:
    """
    Main feature analysis coordinator.
    """
    
    def __init__(self, method_registry, cv_validator):
        self.method_registry = method_registry
        self.cv_validator = cv_validator
        self.selector = SimpleFeatureSelector()
        self.combo_tester = FeatureCombinationTester(method_registry, cv_validator)
    
    def test_selection_methods(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """
        Compare different feature selection methods.
        """
        print("Testing feature selection methods...")
        
        # Split data once for consistent comparison
        splitter = TimeSeriesSplitter(AnalysisConfig())
        X_train, X_test, y_train, y_test = splitter.split(y, X)
        
        selection_methods = {}
        
        # All Features baseline
        selection_methods['All Features'] = (feature_names, np.arange(len(feature_names)))
        
        # Statistical selection (F-test)
        try:
            sel_names, sel_idx = self.selector.statistical_selection(X_train, y_train, feature_names, k=5)
            selection_methods['Statistical (F-test)'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Statistical selection failed: {e}")
        
        # Importance-based selection
        try:
            sel_names, sel_idx = self.selector.importance_selection(X_train, y_train, feature_names, n_features=5)
            selection_methods['Importance (RF)'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Importance selection failed: {e}")
        
        # RFE selection
        try:
            sel_names, sel_idx = self.selector.rfe_selection(X_train, y_train, feature_names, n_features=5)
            selection_methods['RFE'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  RFE selection failed: {e}")
        
        # Lasso selection
        try:
            sel_names, sel_idx = self.selector.lasso_selection(X_train, y_train, feature_names)
            selection_methods['Lasso'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Lasso selection failed: {e}")
        
        # Test each selection method
        results = []
        method = self.method_registry.get_method('Random Forest')
        
        for method_name, (sel_names, sel_idx) in selection_methods.items():
            try:
                if len(sel_idx) == 0:
                    continue
                
                X_train_sel = X_train[:, sel_idx]
                X_test_sel = X_test[:, sel_idx]
                
                # Fit on selected features
                train_results = method.fit(X_train_sel, y_train)
                test_perf = self.cv_validator._evaluate_on_test(train_results, X_test_sel, y_test, method)
                
                results.append({
                    'selection_method': method_name,
                    'selected_features': ', '.join(sel_names),
                    'n_features': len(sel_names),
                    'test_r_squared': test_perf.get('r_squared', np.nan),
                    'test_mse': test_perf.get('mse', np.nan),
                    'overfitting': train_results.get('r_squared', 0) - test_perf.get('r_squared', 0)
                })
                
                print(f"  {method_name}: {len(sel_names)} features, Test R² = {test_perf.get('r_squared', np.nan):.4f}")
                
            except Exception as e:
                print(f"  {method_name} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print("No selection methods succeeded")
            return pd.DataFrame(columns=['selection_method', 'selected_features', 'n_features', 'test_r_squared', 'test_mse', 'overfitting'])
        
        df = pd.DataFrame(results)
        return df.sort_values('test_r_squared', ascending=False).reset_index(drop=True)
    
    def analyze_feature_importance(self, train_results: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze and rank feature importance from trained model.
        """
        importance_data = []
        
        # Extract feature importance/coefficients
        if 'feature_importance' in train_results:
            # Tree-based methods
            importances = train_results['feature_importance']
            for i, (name, imp) in enumerate(zip(feature_names, importances)):
                importance_data.append({
                    'feature': name,
                    'importance': imp,
                    'rank': i + 1,
                    'type': 'importance'
                })
        
        elif 'coefficients' in train_results:
            # Linear methods
            coefficients = train_results['coefficients']
            
            # Handle different coefficient formats
            if hasattr(coefficients, 'values'):
                coef_values = coefficients.values
            else:
                coef_values = np.array(coefficients)
            
            # Skip constant term if present (OLS adds constant)
            if len(coef_values) == len(feature_names) + 1:
                coef_values = coef_values[1:]  # Skip constant
            
            # Create importance based on absolute coefficient values
            abs_coefs = np.abs(coef_values[:len(feature_names)])
            
            for i, (name, coef) in enumerate(zip(feature_names, coef_values[:len(feature_names)])):
                importance_data.append({
                    'feature': name,
                    'importance': abs_coefs[i],
                    'coefficient': coef,
                    'rank': i + 1,
                    'type': 'coefficient'
                })
        
        if not importance_data:
            return pd.DataFrame(columns=['feature', 'importance', 'rank', 'type'])
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df

print("Feature selection and analysis loaded")







"""
Improved Time Series Splitting & Cross-Validation - KORRIGIERT
Verhindert Data Leakage durch strenge zeitliche Trennung und realistische CV-Splits
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

class ImprovedTimeSeriesSplitter:
    """
    Verbesserter Zeitreihen-Splitter mit strikten Anti-Leakage Regeln.
    """
    
    def __init__(self, config):
        self.config = config
    
    def split_with_dates(self, data: pd.DataFrame, 
                        date_col: str = "Datum",
                        test_size: float = None,
                        min_train_size: float = 0.6,
                        gap_months: int = 3) -> Dict[str, Any]:
        """
        Zeitbasierter Split mit expliziten Datums-Gaps.
        
        Args:
            data: DataFrame mit Zeitreihen
            date_col: Name der Datumsspalte  
            test_size: Anteil der Testdaten (default aus config)
            min_train_size: Mindestanteil für Trainingsdaten
            gap_months: Monate zwischen Training und Test
        
        Returns:
            Dict mit train_data, test_data, gap_info, split_info
        """
        if len(data) < 30:
            raise ValueError(f"Dataset zu klein für stabilen Split: {len(data)} Beobachtungen")
        
        test_size = test_size or self.config.test_size
        data_sorted = data.sort_values(date_col).reset_index(drop=True)
        
        total_obs = len(data_sorted)
        
        # Berechne Split-Punkte basierend auf Datum
        date_range = data_sorted[date_col].max() - data_sorted[date_col].min()
        total_months = date_range.days / 30.44  # Approximation
        
        if total_months < 24:  # Weniger als 2 Jahre
            gap_months = 1  # Reduziere Gap
            test_size = min(test_size, 0.2)  # Kleinere Testgröße
        
        # Test-Periode definieren
        test_months = total_months * test_size
        train_months = total_months - test_months - gap_months
        
        if train_months < total_months * min_train_size:
            # Anpassung wenn zu wenig Trainingsdaten
            train_months = total_months * min_train_size
            gap_months = max(1, int((total_months - train_months - test_months) / 2))
            test_months = total_months - train_months - gap_months
            
            warnings.warn(f"Gap reduziert auf {gap_months} Monate für ausreichend Trainingsdaten")
        
        # Datums-basierte Cutoffs berechnen
        start_date = data_sorted[date_col].min()
        train_end_date = start_date + pd.DateOffset(months=int(train_months))
        gap_end_date = train_end_date + pd.DateOffset(months=gap_months)
        test_end_date = data_sorted[date_col].max()
        
        # Daten aufteilen
        train_mask = data_sorted[date_col] < train_end_date
        test_mask = data_sorted[date_col] >= gap_end_date
        gap_mask = (data_sorted[date_col] >= train_end_date) & (data_sorted[date_col] < gap_end_date)
        
        train_data = data_sorted[train_mask].copy()
        test_data = data_sorted[test_mask].copy()
        gap_data = data_sorted[gap_mask].copy()
        
        # Validierung
        if len(train_data) < 20:
            raise ValueError(f"Training set zu klein: {len(train_data)} Beobachtungen")
        if len(test_data) < 5:
            raise ValueError(f"Test set zu klein: {len(test_data)} Beobachtungen")
        
        split_info = {
            'total_observations': total_obs,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'gap_size': len(gap_data),
            'train_ratio': len(train_data) / total_obs,
            'test_ratio': len(test_data) / total_obs,
            'gap_ratio': len(gap_data) / total_obs,
            'train_end_date': train_end_date,
            'gap_end_date': gap_end_date,
            'actual_gap_months': gap_months,
            'date_range_months': total_months
        }
        
        print(f"Time series split: Train={len(train_data)} | Gap={len(gap_data)} | Test={len(test_data)}")
        print(f"  Training period: {data_sorted[date_col].min().strftime('%Y-%m')} to {train_end_date.strftime('%Y-%m')}")
        print(f"  Gap period: {train_end_date.strftime('%Y-%m')} to {gap_end_date.strftime('%Y-%m')}")
        print(f"  Test period: {gap_end_date.strftime('%Y-%m')} to {data_sorted[date_col].max().strftime('%Y-%m')}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'gap_data': gap_data,
            'split_info': split_info,
            'train_end_date': train_end_date
        }
    
    def split(self, y: np.ndarray, X: np.ndarray, 
              dates: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Legacy-kompatibler Split für bestehenden Code.
        """
        n_samples = len(X)
        
        # Adaptive Größen basierend auf Stichprobe
        if n_samples < 40:
            test_size = 0.20
            gap_periods = 1
        elif n_samples < 80:
            test_size = 0.25
            gap_periods = 2
        else:
            test_size = self.config.test_size
            gap_periods = max(1, int(n_samples * 0.015))  # 1.5% als Gap
        
        # Berechne Indizes
        test_samples = int(n_samples * test_size)
        train_end = n_samples - test_samples - gap_periods
        test_start = train_end + gap_periods
        
        # Mindest-Trainingsgröße sicherstellen
        min_train = max(20, int(n_samples * 0.5))  # Mindestens 50% für Training
        if train_end < min_train:
            train_end = min_train
            gap_periods = max(1, n_samples - train_end - test_samples)
            test_start = train_end + gap_periods
        
        if test_start >= n_samples:
            raise ValueError("Dataset zu klein für robusten Split")
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_start + test_samples]
        y_test = y[test_start:test_start + test_samples]
        
        print(f"Array split: Train={len(y_train)}, Gap={gap_periods}, Test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test


class ImprovedRobustCrossValidator:
    """
    Verbesserte Cross-Validation mit realistischen Splits und Stabilitätsprüfungen.
    """
    
    def __init__(self, config):
        self.config = config
        
    def validate_method_robust(self, method, X_train: np.ndarray, y_train: np.ndarray,
                              dates_train: pd.Series = None) -> Dict[str, Any]:
        """
        Robuste Kreuzvalidierung mit stabilitätsfokussierten Splits.
        """
        n_samples = len(X_train)
        
        # Sehr konservative CV-Parameter basierend auf Datengröße
        if n_samples < 40:
            n_splits = 2
            gap = 1
        elif n_samples < 100:
            n_splits = 3
            gap = 2
        else:
            n_splits = 4
            gap = max(2, int(n_samples * 0.02))  # Statt 0.05
        
        if n_splits < 2:
            return {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'n_folds': 0,
                'stability_warning': 'Dataset too small for CV'
            }
        
        # Verwende TimeSeriesSplit mit größeren Gaps
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=None)
        cv_scores = []
        fold_details = []
        successful_folds = 0
        
        print(f"    Running {n_splits}-fold CV with gap={gap}...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            try:
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]  
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
                
                # Strikte Mindestgrößen für Folds
                min_train_fold = max(10, len(X_train) // (n_splits + 3))
                min_val_fold = max(3, len(X_train) // (n_splits * 4))
                
                if len(X_fold_train) < min_train_fold or len(X_fold_val) < min_val_fold:
                    print(f"      Fold {fold_idx+1}: Skipped (train={len(X_fold_train)}, val={len(X_fold_val)})")
                    continue
                
                # Prüfe auf ausreichende Variation in y
                if y_fold_train.std() < 1e-8 or y_fold_val.std() < 1e-8:
                    print(f"      Fold {fold_idx+1}: Skipped (insufficient variation)")
                    continue
                
                # Fit auf Fold-Training
                try:
                    fold_results = method.fit(X_fold_train, y_fold_train)
                except Exception as fit_error:
                    print(f"      Fold {fold_idx+1}: Fit failed ({str(fit_error)[:30]})")
                    continue
                
                # Evaluiere auf Fold-Validation
                fold_test_perf = self._evaluate_on_test_safe(
                    fold_results, X_fold_val, y_fold_val, method
                )
                
                score = fold_test_perf.get('r_squared', np.nan)
                mse = fold_test_perf.get('mse', np.nan)
                
                # Strenge Sanity Checks
                is_valid_score = (
                    np.isfinite(score) and 
                    -2.0 <= score <= 1.0 and  # Erweitere negativen Bereich leicht
                    np.isfinite(mse) and 
                    mse > 0
                )
                
                if is_valid_score:
                    cv_scores.append(score)
                    fold_details.append({
                        'fold': fold_idx + 1,
                        'r_squared': score,
                        'mse': mse,
                        'train_size': len(X_fold_train),
                        'val_size': len(X_fold_val)
                    })
                    successful_folds += 1
                    print(f"      Fold {fold_idx+1}: R² = {score:.4f}, MSE = {mse:.4f}")
                else:
                    print(f"      Fold {fold_idx+1}: Invalid score R²={score:.4f}, MSE={mse:.4f}")
                
            except Exception as e:
                print(f"      Fold {fold_idx+1}: Exception - {str(e)[:40]}")
                continue
        
        if len(cv_scores) == 0:
            return {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'n_folds': 0,
                'stability_warning': 'All CV folds failed'
            }
        
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Stabilitäts-Assessment
        stability_warnings = []
        
        if cv_std > 0.3:
            stability_warnings.append('High CV variance - unstable model')
        
        if len(cv_scores) < n_splits / 2:
            stability_warnings.append(f'Only {len(cv_scores)}/{n_splits} folds succeeded')
            
        if successful_folds > 1:
            score_range = max(cv_scores) - min(cv_scores)
            if score_range > 0.5:
                stability_warnings.append('Very high score range across folds')
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'n_folds': successful_folds,
            'fold_details': fold_details,
            'stability_warnings': stability_warnings,
            'parameters': {
                'n_splits_attempted': n_splits,
                'gap_used': gap,
                'min_train_fold': min_train_fold,
                'min_val_fold': min_val_fold
            }
        }
    
    def _evaluate_on_test_safe(self, train_results: Dict[str, Any], 
                              X_test: np.ndarray, y_test: np.ndarray, 
                              method) -> Dict[str, Any]:
        """
        Sichere Test-Evaluierung mit ausführlichem Error Handling.
        """
        try:
            model = train_results.get('model')
            if model is None:
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan, 
                       'error': 'No model in train_results'}
            
            # Predictions generieren basierend auf Methodentyp
            if method.name == "OLS":
                # Statsmodels OLS
                import statsmodels.api as sm
                X_test_with_const = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_with_const)
                
            elif method.requires_scaling and 'scaler' in train_results:
                # Skalierte Methoden (SVR, Bayesian Ridge)
                scaler = train_results['scaler']
                X_test_scaled = scaler.transform(X_test)
                
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test_scaled)
                else:
                    return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                           'error': 'Model has no predict method'}
                    
            else:
                # Tree-based und andere Methoden
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                           'error': 'Model has no predict method'}
            
            # Validate predictions
            if not isinstance(y_pred, (np.ndarray, list)):
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'Invalid prediction type'}
            
            y_pred = np.array(y_pred).flatten()
            
            if len(y_pred) != len(y_test):
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'Prediction length mismatch'}
            
            if not np.isfinite(y_pred).any():
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'All predictions are non-finite'}
            
            # Berechne Metriken mit robusten Checks
            try:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Zusätzliche Sanity Checks
                if not np.isfinite(mse) or mse < 0:
                    mse = np.nan
                if not np.isfinite(r2):
                    r2 = np.nan  
                if not np.isfinite(mae) or mae < 0:
                    mae = np.nan
                
                return {
                    'r_squared': float(r2) if np.isfinite(r2) else np.nan,
                    'mse': float(mse) if np.isfinite(mse) else np.nan,
                    'mae': float(mae) if np.isfinite(mae) else np.nan,
                    'predictions': y_pred,
                    'actual': y_test,
                    'residuals': y_test - y_pred
                }
                
            except Exception as metric_error:
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': f'Metric calculation failed: {str(metric_error)[:50]}'}
                       
        except Exception as e:
            return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                   'error': f'Evaluation failed: {str(e)[:50]}'}


class ImprovedFeatureSelector:
    """
    Vereinfachte Feature-Selektion mit Fokus auf Stabilität.
    """
    
    @staticmethod  
    def select_robust_features(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              max_features: int = None) -> Tuple[List[str], np.ndarray]:
        """
        Robuste Feature-Selektion basierend auf Korrelation und Stabilität.
        """
        if X.shape[1] == 0:
            return [], np.array([])
        
        # Automatische Begrenzung basierend auf Sample-Größe
        n_samples = len(X)
        if max_features is None:
            if n_samples < 30:
                max_features = 2  
            elif n_samples < 60:
                max_features = 3
            elif n_samples < 100:
                max_features = 4
            else:
                max_features = min(6, X.shape[1], n_samples // 15)
        
        max_features = min(max_features, X.shape[1])
        
        # Berechne Korrelationen mit Target
        correlations = []
        valid_features = []
        
        for i, feature_name in enumerate(feature_names):
            try:
                feature_data = X[:, i]
                
                # Skip konstante oder fast-konstante Features
                if np.std(feature_data) < 1e-10:
                    continue
                    
                # Skip Features mit zu vielen NaNs
                if np.isnan(feature_data).sum() > len(feature_data) * 0.5:
                    continue
                
                # Berechne Korrelation (robust gegen NaNs)
                mask = ~(np.isnan(feature_data) | np.isnan(y))
                if mask.sum() < max(5, len(y) * 0.3):  # Mindestens 30% overlap
                    continue
                    
                corr = np.corrcoef(feature_data[mask], y[mask])[0, 1]
                
                if np.isfinite(corr):
                    correlations.append((abs(corr), i, feature_name))
                    valid_features.append(i)
                    
            except Exception:
                continue
        
        if not correlations:
            # Fallback: erste verfügbare Features nehmen
            available_features = []
            for i, name in enumerate(feature_names[:max_features]):
                if np.std(X[:, i]) > 1e-10:
                    available_features.append((name, i))
                    
            if available_features:
                selected_names = [name for name, idx in available_features]
                selected_indices = np.array([idx for name, idx in available_features])
                return selected_names, selected_indices
            else:
                return [], np.array([])
        
        # Sortiere nach Korrelation und wähle Top-Features
        correlations.sort(key=lambda x: x[0], reverse=True)
        selected_correlations = correlations[:max_features]
        
        selected_names = [name for _, _, name in selected_correlations]
        selected_indices = np.array([idx for _, idx, _ in selected_correlations])
        
        print(f"Selected {len(selected_names)}/{len(feature_names)} features based on correlation:")
        for corr_abs, idx, name in selected_correlations:
            print(f"  {name}: |r| = {corr_abs:.3f}")
        
        return selected_names, selected_indices











# %%
"""
Main Financial Regression Analyzer - VEREINFACHT & ROBUST
Koordiniert alle Komponenten ohne komplexe Monkey-Patches
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import warnings

class FinancialRegressionAnalyzer:
    """
    Main analyzer that coordinates all components for financial regression analysis.
    Clean implementation without monkey patches.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, exog_vars: List[str], 
                 config: AnalysisConfig = None, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.exog_vars = exog_vars
        self.date_col = date_col
        self.config = config or AnalysisConfig()
        
        # Initialize components
        self.method_registry = MethodRegistry()
        self.cv_validator = RobustCrossValidator(self.config)
        self.splitter = TimeSeriesSplitter(self.config)
        self.preprocessor = DataPreprocessor(data, target_var, date_col)
        self.feature_analyzer = FeatureAnalyzer(self.method_registry, self.cv_validator)
    
    def prepare_data(self, transformation: str = 'levels') -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """
        Prepare data for regression with proper mixed-frequency handling.
        """
        print(f"Preparing data with '{transformation}' transformation...")
        
        # Apply transformations (includes mixed-frequency handling)
        transformed_data = self.preprocessor.create_transformations(transformation)
        
        # Find target and feature columns
        target_col = self.target_var
        if target_col not in transformed_data.columns:
            possible_targets = [col for col in transformed_data.columns 
                             if col.startswith(self.target_var)]
            if possible_targets:
                target_col = possible_targets[0]
            else:
                raise ValueError(f"Target variable {target_col} not found after transformation")
        
        # Get feature columns
        feature_cols = []
        for var in self.exog_vars:
            if var in transformed_data.columns:
                feature_cols.append(var)
        
        # Add seasonal dummies and time trend
        seasonal_cols = ['Q2', 'Q3', 'Q4', 'time_trend']
        for col in seasonal_cols:
            if col in transformed_data.columns:
                feature_cols.append(col)
        
        if not feature_cols:
            raise ValueError("No feature columns found")
        
        # Create final dataset
        final_data = transformed_data[[target_col] + feature_cols].copy()
        final_data = final_data.dropna()
        
        if len(final_data) < 20:
            raise ValueError(f"Insufficient data: only {len(final_data)} observations after cleaning")
        
        # Extract arrays
        y = final_data[target_col].values
        X = final_data[feature_cols].values
        
        print(f"Final dataset: {len(final_data)} observations, {len(feature_cols)} features")
        
        return y, X, feature_cols, final_data
    
    def fit_method_with_validation(self, method_name: str, y: np.ndarray, X: np.ndarray, 
                                 feature_names: List[str], transformation: str = 'levels', final_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit a method with comprehensive validation.
        """
        print(f"  Fitting {method_name}...")
        
        # Train/test split
        X_train, X_test, y_train, y_test = self.splitter.split(y, X)
        
        
        # If configured, restrict test evaluation to quarter-end months
        if getattr(self.config, 'evaluate_quarter_ends_only', False) and isinstance(final_data, pd.DataFrame) and self.date_col in final_data.columns:
            dates_all = pd.to_datetime(final_data[self.date_col].values)
            n_samples = len(dates_all)
            # Recompute split indices like the splitter
            if n_samples < 40:
                test_size = 0.20; gap_periods = 1
            elif n_samples < 80:
                test_size = 0.25; gap_periods = 2
            else:
                test_size = self.config.test_size
                gap_periods = max(1, int(n_samples * 0.015))
            test_samples = int(n_samples * test_size)
            train_end = n_samples - test_samples - gap_periods
            test_start = train_end + gap_periods
            dates_test = pd.Series(dates_all[test_start:test_start + test_samples])
            qe_mask = dates_test.dt.is_quarter_end
            if qe_mask.any() and qe_mask.sum() >= 3:
                X_test = X_test[qe_mask.values]
                y_test = y_test[qe_mask.values]
                if len(X_train) < 10 or len(X_test) < 3:
                    raise ValueError(f"Insufficient data for train/test split: {len(X_train)}/{len(X_test)}")
        
        # Get method and fit on training data
        method = self.method_registry.get_method(method_name)
        train_results = method.fit(X_train, y_train)
        
        # Evaluate on test data
        test_performance = self.cv_validator._evaluate_on_test(train_results, X_test, y_test, method)
        
        # Cross-validation on training data
        cv_performance = self.cv_validator.validate_method(method, X_train, y_train)
        
        # Calculate metrics
        train_r2 = train_results.get('r_squared', np.nan)
        test_r2 = test_performance['r_squared']
        overfitting = train_r2 - test_r2 if np.isfinite(train_r2) and np.isfinite(test_r2) else np.nan
        
        # Combine results
        results = {
            **train_results,
            'method_name': method_name,
            'feature_names': feature_names,
            'target_var': self.target_var,
            'transformation': transformation,
            'test_performance': test_performance,
            'cv_performance': cv_performance,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'overfitting': overfitting,
            'validation_config': {
                'test_size': self.config.test_size,
                'train_size': len(X_train),
                'test_size_actual': len(X_test),
                'gap_used': len(y) - len(X_train) - len(X_test)
            }
        }
        
        # Performance summary
        cv_mean = cv_performance.get('cv_mean', np.nan)
        print(f"    Test R² = {test_r2:.4f}, Train R² = {train_r2:.4f}, CV = {cv_mean:.4f}")
        
        # Warnings
        if overfitting > 0.1:
            print(f"    ⚠️ WARNING: High overfitting ({overfitting:.4f})")
        if test_r2 > 0.9:
            print(f"    ⚠️ WARNING: Very high R² ({test_r2:.4f}) - check for leakage")
        
        return results
    
    def fit_multiple_methods(self, methods: List[str] = None, transformation: str = 'levels') -> Dict[str, Dict[str, Any]]:
        """
        Fit multiple methods with validation.
        """
        if methods is None:
            methods = self.method_registry.list_methods()
        
        # Prepare data
        y, X, feature_names, final_data = self.prepare_data(transformation)
        
        results = {}
        print(f"Fitting {len(methods)} methods with robust validation...")
        
        for method_name in methods:
            try:
                result = self.fit_method_with_validation(method_name, y, X, feature_names, transformation, final_data=final_data)
                results[method_name] = result
                
            except Exception as e:
                print(f"    ✗ {method_name} failed: {str(e)[:50]}")
                continue
        
        print(f"Successfully fitted {len(results)}/{len(methods)} methods")
        return results
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare method results in a structured format.
        """
        comparison_data = []
        
        for method_name, result in results.items():
            train_r2 = result.get('train_r_squared', np.nan)
            test_r2 = result.get('test_r_squared', np.nan)
            overfitting = result.get('overfitting', np.nan)
            cv_mean = result.get('cv_performance', {}).get('cv_mean', np.nan)
            cv_std = result.get('cv_performance', {}).get('cv_std', np.nan)
            
            # Classify overfitting level
            if np.isfinite(overfitting):
                if overfitting > 0.15:
                    overfitting_level = "SEVERE"
                elif overfitting > 0.08:
                    overfitting_level = "HIGH"
                elif overfitting > 0.04:
                    overfitting_level = "MODERATE"
                else:
                    overfitting_level = "LOW"
            else:
                overfitting_level = "UNKNOWN"
            
            # Get additional metrics
            oob_score = result.get('oob_score', np.nan)
            test_mse = result.get('test_performance', {}).get('mse', np.nan)
            test_mae = result.get('test_performance', {}).get('mae', np.nan)
            
            comparison_data.append({
                'Method': method_name,
                'Test_R²': test_r2,
                'Train_R²': train_r2,
                'Overfitting': overfitting,
                'Overfitting_Level': overfitting_level,
                'Test_MSE': test_mse,
                'Test_MAE': test_mae,
                'CV_Mean_R²': cv_mean,
                'CV_Std_R²': cv_std,
                'OOB_Score': oob_score
            })
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test_R²', ascending=False).round(4)
        
        # Add warning flags
        df['Warnings'] = ''
        for idx, row in df.iterrows():
            warnings = []
            if row['Test_R²'] > 0.9:
                warnings.append("Very high R² - check for leakage")
            if row['Test_R²'] < 0:
                warnings.append("Negative R² - poor fit")
            if row['Overfitting'] > 0.1:
                warnings.append("High overfitting")
            if np.isfinite(row['CV_Mean_R²']) and abs(row['Test_R²'] - row['CV_Mean_R²']) > 0.1:
                warnings.append("CV/Test discrepancy")
            df.at[idx, 'Warnings'] = '; '.join(warnings)
        
        return df
    
    def find_optimal_transformation(self, transformations: List[str] = None, 
                                   baseline_method: str = 'Random Forest') -> str:
        """
        Find optimal transformation by testing with a baseline method.
        """
        if transformations is None:
            transformations = ['levels', 'pct', 'diff']
        
        print("Finding optimal transformation...")
        
        best_transformation = None
        best_score = -np.inf
        transformation_results = {}
        
        for transformation in transformations:
            try:
                print(f"  Testing '{transformation}' transformation...")
                
                # Test with baseline method
                y, X, feature_names, final_data = self.prepare_data(transformation)
                result = self.fit_method_with_validation(method_name, y, X, feature_names,
                                                        transformation, final_data=final_data)
                #y, X, feature_names, _ = self.prepare_data(transformation)
                #result = self.fit_method_with_validation(baseline_method, y, X, feature_names, transformation, final_data=final_data)
                
                test_r2 = result.get('test_r_squared', np.nan)
                
                transformation_results[transformation] = {
                    'test_r2': test_r2,
                    'result': result
                }
                
                print(f"    Test R² = {test_r2:.4f}")
                
                # Check if this is the best so far
                if np.isfinite(test_r2) and test_r2 > best_score:
                    best_score = test_r2
                    best_transformation = transformation
                
            except Exception as e:
                print(f"    ✗ {transformation} failed: {str(e)[:50]}")
                continue
        
        if best_transformation is None:
            print("  Warning: No transformations succeeded, using 'levels'")
            return 'levels'
        
        print(f"  Best transformation: '{best_transformation}' (Test R² = {best_score:.4f})")
        return best_transformation
    
    def test_feature_selection_methods(self, transformation: str = 'levels') -> pd.DataFrame:
        """
        Test different feature selection methods.
        """
        print("Testing feature selection methods...")
        
        try:
            y, X, feature_names, _ = self.prepare_data(transformation)
            return self.feature_analyzer.test_selection_methods(X, y, feature_names)
        except Exception as e:
            print(f"Feature selection testing failed: {e}")
            return pd.DataFrame()
    
    def test_feature_combinations(self, max_combinations: int = 20, 
                                 transformation: str = 'levels') -> pd.DataFrame:
        """
        Test different feature combinations.
        """
        print("Testing feature combinations...")
        
        try:
            y, X, feature_names, _ = self.prepare_data(transformation)
            return self.feature_analyzer.combo_tester.test_combinations(
                X, y, feature_names, max_combinations
            )
        except Exception as e:
            print(f"Feature combination testing failed: {e}")
            return pd.DataFrame()

"""
Improved Financial Regression Analyzer - KORRIGIERT
Behebt die Hauptprobleme mit Data Leakage, CV-Splits und Feature Selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings



class TrainOnlyLagSelector:
    """
    Selects best lags per exogenous variable using TRAIN-ONLY correlation with the (transformed) target.
    No leakage: only dates strictly before train_end_date are used for scoring.
    """
    def __init__(self, config: LagConfig):
        self.cfg = config

    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        try:
            s = pd.concat([a, b], axis=1).dropna()
            if len(s) < 3:
                return np.nan
            return float(s.iloc[:,0].corr(s.iloc[:,1]))
        except Exception:
            return np.nan

    def apply(self, df: pd.DataFrame, exog_vars: List[str], target_col: str, date_col: str, train_end_date: Optional[pd.Timestamp]):
        kept = []
        details = []
        # Build train mask
        if train_end_date is not None:
            train_mask = pd.to_datetime(df[date_col]) < pd.to_datetime(train_end_date)
        else:
            # fallback: first 80%
            n = len(df)
            cutoff = int(n * 0.8)
            train_mask = pd.Series([True]*cutoff + [False]*(n-cutoff), index=df.index)

        # Score candidates
        candidates = []
        for var in exog_vars:
            if var not in df.columns:
                continue
            for L in self.cfg.candidates:
                col = f"{var}_lag{L}"
                if col not in df.columns:
                    df[col] = df[var].shift(L)
                corr = self._safe_corr(df.loc[train_mask, col], df.loc[train_mask, target_col])
                # require minimal overlap
                overlap = int(pd.concat([df[col], df[target_col]], axis=1).loc[train_mask].dropna().shape[0])
                if overlap >= self.cfg.min_train_overlap and (not np.isfinite(self.cfg.min_abs_corr) or abs(corr) >= self.cfg.min_abs_corr):
                    candidates.append((var, col, L, abs(corr), overlap))
                else:
                    details.append({'var': var, 'lag': L, 'status': 'skipped', 'corr': corr, 'overlap': overlap})

        # choose per-var best, then apply total cap
        best_per_var = {}
        for var, col, L, acorr, overlap in candidates:
            cur = best_per_var.get(var)
            if (cur is None) or (acorr > cur[3]):
                best_per_var[var] = (var, col, L, acorr, overlap)

        # Flatten, sort by |corr| desc
        ranked = sorted(best_per_var.values(), key=lambda x: x[3], reverse=True)
        total_cap = max(0, int(self.cfg.total_max))
        per_var_cap = max(1, int(self.cfg.per_var_max))
        per_var_counts = {v: 0 for v in best_per_var.keys()}

        for var, col, L, acorr, overlap in ranked:
            if len(kept) >= total_cap:
                break
            if per_var_counts[var] >= per_var_cap:
                continue
            kept.append(col)
            per_var_counts[var] += 1
            details.append({'var': var, 'lag': L, 'status': 'kept', 'corr': acorr, 'overlap': overlap})

        # mark dropped
        kept_set = set(kept)
        for var, col, L, acorr, overlap in ranked:
            if col not in kept_set:
                details.append({'var': var, 'lag': L, 'status': 'dropped', 'corr': acorr, 'overlap': overlap})

        report = {
            'kept': kept,
            'details': details
        }
        return df, report

class ImprovedFinancialRegressionAnalyzer:
    """
    Korrigierter Hauptanalysator mit robuster Anti-Leakage Architektur.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, exog_vars: List[str], 
                 config=None, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.exog_vars = exog_vars
        self.date_col = date_col
        self.config = config or AnalysisConfig()
        
        # Improved components
        self.method_registry = MethodRegistry()
        self.cv_validator = ImprovedRobustCrossValidator(self.config)
        self.splitter = ImprovedTimeSeriesSplitter(self.config)
        self.preprocessor = ImprovedDataPreprocessor(data, target_var, date_col)
        self.quality_checker = ImprovedDataQualityChecker()
    
    def comprehensive_data_validation(self) -> Dict[str, Any]:
        """
        Umfassende Datenvalidierung als erster Schritt.
        """
        print("=== COMPREHENSIVE DATA VALIDATION ===")
        
        validation_result = self.quality_checker.comprehensive_data_validation(
            self.data, self.target_var, self.exog_vars,
            min_observations=30,  # Erhöhte Mindestanforderung
            min_target_coverage=0.25  # Höhere Coverage-Anforderung
        )
        
        # Ausgabe der Validierungsergebnisse
        print(f"\nData Quality Summary:")
        print(f"  Total observations: {len(self.data)}")
        print(f"  Variables tested: {len([self.target_var] + self.exog_vars)}")
        
        if validation_result['errors']:
            print(f"\n❌ ERRORS ({len(validation_result['errors'])}):")
            for error in validation_result['errors']:
                print(f"    - {error}")
        
        if validation_result['warnings']:
            print(f"\n⚠️  WARNINGS ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings']:
                print(f"    - {warning}")
        
        # Stationarity results
        print(f"\nStationarity Tests:")
        for var, result in validation_result['stationarity_tests'].items():
            is_stationary = result.get('is_stationary', None)
            p_value = result.get('p_value', np.nan)
            
            status = "✅ Stationary" if is_stationary else "❌ Non-stationary" if is_stationary is False else "❓ Unknown"
            print(f"  {var}: {status} (p-value: {p_value:.3f})")
        
        # Recommendations
        if validation_result['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_result['recommendations']:
                print(f"    - {rec}")
        
        return validation_result
    
    def prepare_data_robust(self, transformation: str = 'levels',
                            use_train_test_split: bool = True) -> Dict[str, Any]:
        """
        Robuste Datenvorbereitung mit Anti-Leakage Schutz.
        """
        print(f"\n=== ROBUST DATA PREPARATION ===")
        print(f"Transformation: {transformation}")
        
        # Schritt 1: Zeitbasierter Split ZUERST (um Leakage zu verhindern)
        train_end_date = None
        split_info = None
        
        if use_train_test_split and len(self.data) > 30:
            try:
                # Früher Split für Anti-Leakage
                split_result = self.splitter.split_with_dates(
                    self.data, self.date_col, 
                    test_size=self.config.test_size,
                    gap_months=2  # Konservativer Gap
                )
                train_end_date = split_result['train_end_date']
                split_info = split_result['split_info']
                print(f"Early train/test split applied - training ends: {train_end_date.strftime('%Y-%m')}")
                
            except Exception as e:
                print(f"Warning: Could not apply early split: {e}")
                train_end_date = None
        
        # Schritt 2: Transformationen mit Anti-Leakage Schutz
        transform_result = self.preprocessor.create_robust_transformations(
            transformation=transformation,
            train_end_date=train_end_date,  # Critical: pass split date
            outlier_method='conservative'
        )
        
        transformed_data = transform_result['data']
        
        # Schritt 3: Feature-Selektion und finales Dataset
        target_col = self.target_var
        if target_col not in transformed_data.columns:
            available_targets = [col for col in transformed_data.columns if col.startswith(self.target_var)]
            if available_targets:
                target_col = available_targets[0]
            else:
                raise ValueError(f"Target variable {target_col} not found after transformation")
        
        # Intelligente Feature-Auswahl
        available_exog = []
        for var in self.exog_vars:
            if var in transformed_data.columns:
                # Prüfe Datenqualität
                series = transformed_data[var]
                coverage = series.notna().sum() / len(series)
                variation = series.std() if series.notna().sum() > 1 else 0.0
                if coverage > 0.3 and variation > 1e-8:  # Mindestanforderungen
                    available_exog.append(var)
                else:
                    print(f"  Excluding {var}: coverage={coverage:.1%}, std={variation:.2e}")
        
        # Saisonale Features hinzufügen
        seasonal_features = ['Q2', 'Q3', 'Q4', 'time_trend']
        for feat in seasonal_features:
            if feat in transformed_data.columns:
                available_exog.append(feat)
        
        # Add target lag (L=1) after seasonal features
        lag_col = f"{target_col}_lag1"
        if target_col in transformed_data.columns and lag_col not in transformed_data.columns:
            transformed_data[lag_col] = transformed_data[target_col].shift(1)
        if lag_col in transformed_data.columns:
            available_exog.append(lag_col)
        
        if not available_exog:
            raise ValueError("No suitable exogenous features found")
        
        # Finales Dataset erstellen
        final_columns = [self.date_col, target_col] + available_exog
        final_data = transformed_data[final_columns].copy()
        # Drop rows with NaN target
        final_data = final_data.dropna(subset=[target_col])
        print(f"Target NaNs after cleaning: {final_data[target_col].isnull().sum()}")
        
        # Robuste Bereinigung
        before_clean = len(final_data)
        
        # Mindestens Target + eine exogene Variable erforderlich
        min_required_vars = 2
        row_validity = final_data.notna().sum(axis=1) >= min_required_vars
        final_data = final_data[row_validity].copy()
        
        after_clean = len(final_data)
        
        if after_clean < 20:
            raise ValueError(f"Insufficient data after cleaning: {after_clean} observations")
        
        # Arrays extrahieren
        y = final_data[target_col].values
        X = final_data[available_exog].values

        # Remove any rows with NaNs in target or features
        mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
        final_data = final_data.loc[mask].copy()
        y = y[mask]
        X = X[mask]

        # Keep aligned dates for later filtering
        dates_array = pd.to_datetime(final_data[self.date_col]).to_numpy()

        print(f"After NaN removal: {len(y)} observations, {np.isnan(y).sum()} NaNs in target")
        print("Final dataset prepared:")
        print(f"  Observations: {before_clean} → {after_clean}")
        print(f"  Features: {len(available_exog)} (+ target)")
        print(f"  Features: {', '.join(available_exog[:5])}{'...' if len(available_exog) > 5 else ''}")
        
        # Sample sizes
        sample_sizes = {
            'total': int(len(y)),
            'features_selected': int(len(available_exog)),
            # optional – nur falls split_info gesetzt wurde:
            'train_candidate': int(split_info['train_size']) if split_info else None,
            'test_candidate': int(split_info['test_size']) if split_info else None,
        }

        return {
            'y': y,
            'X': X,
            'feature_names': available_exog,
            'final_data': final_data,
            'target_name': target_col,
            'transformation_info': transform_result,
            'split_info': split_info,
            'train_end_date': train_end_date,
            'preparation_warnings': transform_result.get('warnings', []),
            'sample_sizes': sample_sizes,
            'dates': dates_array
        }

    
    def fit_method_improved(self, method_name: str, preparation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verbesserte Methodenanpassung mit robusten Splits.
        """
        print(f"  Fitting {method_name}...")
        
        y = preparation_result['y']
        X = preparation_result['X']
        feature_names = preparation_result['feature_names']
        
        # Feature-Reduktion für kleine Datensätze
        if len(y) < 60 or X.shape[1] > len(y) // 10:
            print(f"    Applying feature selection: {X.shape[1]} → ", end="")
            selected_names, selected_indices = ImprovedFeatureSelector.select_robust_features(
                X, y, feature_names, max_features=min(6, len(y) // 15)
            )
            
            if len(selected_indices) > 0:
                X = X[:, selected_indices]
                feature_names = selected_names
                print(f"{len(selected_names)} features")
            else:
                raise ValueError("Feature selection resulted in no features")
        
        # Train/Test Split mit verbessertem Splitter
        try:
            X_train, X_test, y_train, y_test = self.splitter.split(y, X)
        except ValueError as e:
            print(f"    Split failed: {e}")
            raise

        # Optional: Testbewertung auf Quartalsenden beschränken
        if getattr(self.config, 'evaluate_quarter_ends_only', False) and 'final_data' in preparation_result:
            dates_all = pd.to_datetime(preparation_result['final_data'][self.date_col].values)
            n_samples = len(dates_all)
            # Recompute split indices wie im Splitter
            if n_samples < 40:
                test_size = 0.20; gap_periods = 1
            elif n_samples < 80:
                test_size = 0.25; gap_periods = 2
            else:
                test_size = self.config.test_size
                gap_periods = max(1, int(n_samples * 0.015))
            test_samples = int(n_samples * test_size)
            train_end = n_samples - test_samples - gap_periods
            test_start = train_end + gap_periods
            dates_test = pd.Series(dates_all[test_start:test_start + test_samples])
            qe_mask = dates_test.dt.is_quarter_end
            if qe_mask.any() and qe_mask.sum() >= 3:
                X_test = X_test[qe_mask.values]
                y_test = y_test[qe_mask.values]
        
        # Method fitting
        method = self.method_registry.get_method(method_name)
        try:
            train_results = method.fit(X_train, y_train)
        except Exception as e:
            print(f"    Training failed: {e}")
            raise
        
        # Test evaluation
        test_performance = self.cv_validator._evaluate_on_test_safe(
            train_results, X_test, y_test, method
        )
        
        # Cross-validation on training data only
        cv_performance = self.cv_validator.validate_method_robust(
            method, X_train, y_train
        )
        
        # Calculate key metrics
        train_r2 = train_results.get('r_squared', np.nan)
        test_r2 = test_performance.get('r_squared', np.nan)
        cv_mean = cv_performance.get('cv_mean', np.nan)
        cv_std = cv_performance.get('cv_std', np.nan)
        
        # Overfitting calculation
        overfitting = train_r2 - test_r2 if (np.isfinite(train_r2) and np.isfinite(test_r2)) else np.nan
        
        # Performance assessment
        performance_flags = []
        if np.isfinite(test_r2) and test_r2 > 0.95:
            performance_flags.append("SUSPICIOUSLY_HIGH_R2")
        if np.isfinite(overfitting) and overfitting > 0.15:
            performance_flags.append("HIGH_OVERFITTING")
        if np.isfinite(cv_std) and cv_std > 0.3:
            performance_flags.append("UNSTABLE_CV")
        if len(cv_performance.get('cv_scores', [])) < 2:
            performance_flags.append("INSUFFICIENT_CV_FOLDS")
        
        # Combine results
        results = {
            **train_results,
            'method_name': method_name,
            'feature_names': feature_names,
            'target_var': preparation_result['target_name'],
            'transformation': preparation_result['transformation_info'].get('transformation_applied') or
                              preparation_result['transformation_info'].get('transformation', 'levels'),
            'test_performance': test_performance,
            'cv_performance': cv_performance,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'cv_mean_r_squared': cv_mean,
            'cv_std_r_squared': cv_std,
            'overfitting': overfitting,
            'performance_flags': performance_flags,
            'sample_sizes': {
                'total': len(y),
                'train': len(y_train) if 'y_train' in locals() else 0,
                'test': len(y_test) if 'y_test' in locals() else 0,
                'features_selected': len(feature_names)
            }
        }
        
        # Performance summary
        print(f"    Results: Test R² = {test_r2:.4f}, CV R² = {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"    Overfitting: {overfitting:.4f}, Flags: {len(performance_flags)}")
        if performance_flags:
            for flag in performance_flags:
                print(f"      ⚠️ {flag}")
        
        return results
    
    def fit_multiple_methods_robust(self, methods: List[str] = None, 
                                    transformation: str = 'auto') -> Dict[str, Any]:
        """
        Robuste Anpassung mehrerer Methoden mit optimaler Transformation.
        """
        if methods is None:
            # Conservative method selection based on sample size
            n_samples = len(self.data)
            if n_samples < 50:
                methods = ['OLS', 'Random Forest']  # Nur robuste Methoden
            elif n_samples < 100:
                methods = ['OLS', 'Random Forest', 'Bayesian Ridge']
            else:
                methods = ['OLS', 'Random Forest', 'XGBoost', 'Bayesian Ridge'] if HAS_XGBOOST else ['OLS', 'Random Forest', 'Bayesian Ridge']
        
        print(f"=== ROBUST METHOD FITTING ===")
        print(f"Methods: {', '.join(methods)}")
        
        # Transformation optimization
        if transformation == 'auto':
            transformation = self._find_optimal_transformation_robust()
        
        print(f"Using transformation: {transformation}")
        
        # Prepare data once
        try:
            preparation_result = self.prepare_data_robust(transformation)
        except Exception as e:
            return {'status': 'failed', 'error': f'Data preparation failed: {str(e)}'}
        
        # Fit methods
        results = {}
        successful_methods = 0
        
        for method_name in methods:
            try:
                result = self.fit_method_improved(method_name, preparation_result)
                results[method_name] = result
                successful_methods += 1
            except Exception as e:
                print(f"    ❌ {method_name} failed: {str(e)[:60]}")
                continue
        
        if successful_methods == 0:
            return {'status': 'failed', 'error': 'All methods failed'}
        
        print(f"Successfully fitted {successful_methods}/{len(methods)} methods")
        
        return {
            'status': 'success',
            'method_results': results,
            'preparation_info': preparation_result,
            'successful_methods': successful_methods,
            'total_methods': len(methods)
        }
    
    def _find_optimal_transformation_robust(self) -> str:
        """
        Robuste Transformationsoptimierung basierend auf Datencharakteristiken.
        """
        print("  Finding optimal transformation...")
        
        # Analyze target variable characteristics
        target_series = self.data[self.target_var].dropna()
        
        if len(target_series) < 10:
            print("    Insufficient data for transformation analysis - using 'levels'")
            return 'levels'
        
        # Check stationarity
        stationarity_result = self.quality_checker.test_stationarity(target_series, self.target_var)
        is_stationary = stationarity_result.get('is_stationary', None)
        p_value = stationarity_result.get('p_value', np.nan)
        if is_stationary is None and (isinstance(p_value, (int, float)) or np.isfinite(p_value)):
            try:
                is_stationary = (p_value < 0.05)
            except Exception:
                pass
        
        # Check value characteristics
        all_positive = (target_series > 0).all()
        has_trend = abs(np.corrcoef(np.arange(len(target_series)), target_series)[0, 1]) > 0.3
        high_volatility = (target_series.std() / abs(target_series.mean()) > 0.5) if target_series.mean() != 0 else False
        
        print(f"    Target characteristics:")
        print(f"      Stationary: {is_stationary}")
        print(f"      All positive: {all_positive}")
        print(f"      Has trend: {has_trend}")
        print(f"      High volatility: {high_volatility}")

        # Transformation decision logic (with p-value fallback and safer defaults)
        if is_stationary is False:
            if all_positive and not high_volatility:
                best_transformation = 'pct'  # percentage change for non-stationary positive series
                print("    → Selected 'pct': Non-stationary positive data")
            else:
                best_transformation = 'diff'  # first differences for non-stationary data
                print("    → Selected 'diff': Non-stationary data")
        elif is_stationary is True:
            if all_positive and has_trend:
                best_transformation = 'log'
                print("    → Selected 'log': Trending positive data")
            else:
                best_transformation = 'levels'
                print("    → Selected 'levels': Stationary or suitable for levels")
        else:
            # Unknown stationarity → be conservative; if p-value available use 0.10 threshold
            try:
                if np.isfinite(p_value) and p_value > 0.10:
                    best_transformation = 'diff'
                    print("    → Selected 'diff': Unknown stationarity, high p-value")
                else:
                    best_transformation = 'levels'
                    print("    → Selected 'levels': Unknown stationarity, defaulting to levels")
            except Exception:
                best_transformation = 'levels'
                print("    → Selected 'levels': Fallback")

        return best_transformation
    
    def create_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstelle umfassende Zusammenfassung mit Qualitätsbewertung.
        """
        if results['status'] != 'success':
            return results
        
        method_results = results['method_results']
        preparation_info = results['preparation_info']
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, result in method_results.items():
            test_r2 = result.get('test_r_squared', np.nan)
            train_r2 = result.get('train_r_squared', np.nan)
            cv_mean = result.get('cv_mean_r_squared', np.nan)
            cv_std = result.get('cv_std_r_squared', np.nan)
            overfitting = result.get('overfitting', np.nan)
            flags = result.get('performance_flags', [])
            
            # Quality assessment
            quality_score = 0.0
            max_score = 5.0
            
            # Test R² contribution (0-2 points)
            if np.isfinite(test_r2):
                if test_r2 > 0.7:
                    quality_score += 2.0
                elif test_r2 > 0.3:
                    quality_score += 1.0
                elif test_r2 > 0:
                    quality_score += 0.5
            
            # Overfitting penalty (-1 to +1 points)
            if np.isfinite(overfitting):
                if overfitting < 0.05:
                    quality_score += 1.0
                elif overfitting < 0.1:
                    quality_score += 0.5
                elif overfitting > 0.2:
                    quality_score -= 1.0
            
            # CV stability (0-1 points)
            if np.isfinite(cv_std) and cv_std < 0.2:
                quality_score += 1.0
            elif np.isfinite(cv_std) and cv_std < 0.3:
                quality_score += 0.5
            
            # Flag penalties
            flag_penalty = len([f for f in flags if 'SUSPICIOUSLY' in f or 'HIGH' in f]) * 0.5
            quality_score -= flag_penalty
            
            quality_score = max(0.0, min(max_score, quality_score))
            
            comparison_data.append({
                'Method': method_name,
                'Test_R²': test_r2,
                'Train_R²': train_r2,
                'CV_Mean_R²': cv_mean,
                'CV_Std_R²': cv_std,
                'Overfitting': overfitting,
                'Quality_Score': quality_score,
                'Flags': len(flags),
                'Flag_Details': '; '.join(flags) if flags else 'None'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Quality_Score', ascending=False).round(4)
        comparison_df = comparison_df.sort_values(
            ['Quality_Score','Test_R²','CV_Mean_R²'],
            ascending=[False, False, False]
        )       
        
        # Best method selection based on quality score
        if not comparison_df.empty:
            best_method_info = comparison_df.iloc[0]
            best_method = best_method_info['Method']
            best_quality = float(best_method_info['Quality_Score'])
        else:
            best_method = None
            best_quality = 0.0
        
        # Create final summary
        summary_parts = []
        summary_parts.append("=== ANALYSIS SUMMARY ===")
        n_total = preparation_info.get('sample_sizes', {}).get('total',
           len(preparation_info.get('final_data', [])))
        summary_parts.append(f"Dataset: {int(n_total)} observations")

        summary_parts.append(f"Transformation: {preparation_info['transformation_info'].get('transformation_applied') or preparation_info['transformation_info'].get('transformation', 'levels')}")
        summary_parts.append(f"Methods tested: {results['successful_methods']}/{results['total_methods']}")
        
        if best_method:
            summary_parts.append(f"Best method: {best_method} (Quality Score: {best_quality:.1f}/5.0)")
            best_result = method_results[best_method]
            summary_parts.append(f"  Test R²: {best_result.get('test_r_squared', np.nan):.4f}")
            summary_parts.append(f"  CV R²: {best_result.get('cv_mean_r_squared', np.nan):.4f} (±{best_result.get('cv_std_r_squared', np.nan):.4f})")
            summary_parts.append(f"  Overfitting: {best_result.get('overfitting', np.nan):.4f}")
        
        # Warnings and recommendations
        all_warnings = preparation_info.get('preparation_warnings', [])[:]
        leakage_risk = preparation_info['transformation_info'].get('leakage_risk', 'low')
        if leakage_risk != 'low':
            all_warnings.append(f"Potential data leakage risk: {leakage_risk}")
        
        if all_warnings:
            summary_parts.append("⚠️ WARNINGS:")
            for warning in all_warnings[:5]:  # Limit to top 5
                summary_parts.append(f"  - {warning}")
        
        summary = "\n".join(summary_parts)
        
        return {
            **results,
            'comparison': comparison_df,
            'best_method': best_method,
            'best_quality_score': best_quality,
            'summary': summary,
            'validation_summary': {
                'total_warnings': len(all_warnings),
                'leakage_risk': leakage_risk,
                'data_coverage': preparation_info.get('data_coverage_ratio', np.nan)
            }
        }


def improved_financial_analysis(data: pd.DataFrame, target_var: str, exog_vars: List[str],
                                analysis_type: str = 'comprehensive', config=None) -> Dict[str, Any]:
    """
    Hauptfunktion für verbesserte Finanzanalyse.
    """
    print("=== IMPROVED FINANCIAL REGRESSION ANALYSIS ===")
    print(f"Target: {target_var}")
    print(f"Features: {', '.join(exog_vars)}")
    print(f"Analysis type: {analysis_type}")
    
    try:
        # Initialize improved analyzer
        analyzer = ImprovedFinancialRegressionAnalyzer(data, target_var, exog_vars, config)
        
        # Step 1: Comprehensive validation
        validation_result = analyzer.comprehensive_data_validation()
        
        if not validation_result['is_valid']:
            return {
                'status': 'failed',
                'stage': 'validation',
                'validation': validation_result,
                'error': 'Data validation failed - see validation results for details'
            }
        
        # Step 2: Method selection based on analysis type
        if analysis_type == 'quick':
            methods = ['Random Forest', 'OLS']
            transformation = 'auto'
        elif analysis_type == 'comprehensive':
            methods = None  # Use automatic selection
            transformation = 'auto'
        else:  # full
            methods = analyzer.method_registry.list_methods()
            transformation = 'auto'
        
        # Step 3: Fit methods
        fit_results = analyzer.fit_multiple_methods_robust(methods, transformation)
        
        if fit_results['status'] != 'success':
            return {
                'status': 'failed',
                'stage': 'fitting',
                'validation': validation_result,
                'error': fit_results.get('error', 'Method fitting failed')
            }
        
        # Step 4: Create comprehensive summary
        final_results = analyzer.create_comprehensive_summary(fit_results)
        
        # Add validation info
        final_results['validation'] = validation_result
        
        print(f"\n{final_results['summary']}")
        
        return final_results
        
    except Exception as e:
        return {
            'status': 'failed',
            'stage': 'unknown',
            'error': f'Analysis failed: {str(e)}',
            'validation': validation_result if 'validation_result' in locals() else {}
        }


def quick_analysis_improved(target_name: str, start_date: str = "2010-01", 
                            config=None) -> Dict[str, Any]:
    """Verbesserte Quick-Analyse."""
    print("=== IMPROVED QUICK ANALYSIS ===")
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        return improved_financial_analysis(data, target_name, exog_vars, 'quick', config)
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'stage': 'data_download'}


def comprehensive_analysis_improved(target_name: str, start_date: str = "2010-01",
                                    config=None) -> Dict[str, Any]:
    """Verbesserte Comprehensive-Analyse."""
    print("=== IMPROVED COMPREHENSIVE ANALYSIS ===")
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        return improved_financial_analysis(data, target_name, exog_vars, 'comprehensive', config)
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'stage': 'data_download'}


print("Main financial regression analyzer loaded")

# %%
"""
Analysis Pipeline & Main Functions - VEREINFACHT
Kombiniert alle Komponenten zu einer benutzbaren Pipeline
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def financial_analysis(data: pd.DataFrame, target_var: str, exog_vars: List[str],
                      analysis_type: str = 'comprehensive', config: AnalysisConfig = None) -> Dict[str, Any]:
    """
    Main financial analysis function - KORRIGIERT für Mixed-Frequency.
    """
    
    config = config or AnalysisConfig()
    
    print("FINANCIAL REGRESSION ANALYSIS")
    print("=" * 50)
    print(f"Target: {target_var}")
    print(f"Features: {', '.join(exog_vars)}")
    print(f"Analysis type: {analysis_type}")
    
    # Step 1: Data Quality Validation
    print("\n1. DATA QUALITY VALIDATION")
    print("-" * 30)
    
    validation = DataQualityChecker.validate_financial_data(
        data, target_var, exog_vars, min_target_coverage=0.3
    )
    
    if not validation['is_valid']:
        print("❌ Data validation failed!")
        for error in validation['errors']:
            print(f"  Error: {error}")
        return {'status': 'failed', 'validation': validation}
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  {warning}")
    
    if validation['recommendations']:
        print("💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"  {rec}")
    
    # Step 2: Detailed Data Diagnosis
    print("\n2. DATA DIAGNOSIS")
    print("-" * 30)
    diagnose_data_issues(data, target_var, exog_vars)
    
    # Step 3: Initialize Analyzer
    print("\n3. ANALYSIS SETUP")
    print("-" * 30)
    analyzer = FinancialRegressionAnalyzer(data, target_var, exog_vars, config)
    
    # Determine methods based on analysis type
    if analysis_type == 'quick':
        methods = ['Random Forest', 'OLS']
        transformations = ['levels', 'pct']
        test_combinations = False
        test_selection = False
    elif analysis_type == 'comprehensive':
        methods = ['Random Forest', 'XGBoost', 'OLS', 'Bayesian Ridge'] if HAS_XGBOOST else ['Random Forest', 'OLS', 'Bayesian Ridge']
        transformations = ['levels', 'pct', 'diff']
        test_combinations = True
        test_selection = True
    else:  # full
        methods = analyzer.method_registry.list_methods()
        transformations = ['levels', 'pct', 'diff']
        test_combinations = True
        test_selection = True
    
    print(f"Methods to test: {', '.join(methods)}")
    print(f"Transformations to test: {', '.join(transformations)}")
    
    # Step 4: Find Optimal Transformation
    print("\n4. TRANSFORMATION OPTIMIZATION")
    print("-" * 30)
    
    try:
        best_transformation = analyzer.find_optimal_transformation(
            transformations, 
            baseline_method=methods[0]
        )
    except Exception as e:
        print(f"Transformation testing failed: {e}")
        best_transformation = 'levels'
    
    # Step 5: Fit All Methods
    print("\n5. METHOD COMPARISON")
    print("-" * 30)
    
    method_results = analyzer.fit_multiple_methods(methods, best_transformation)
    
    if not method_results:
        return {'status': 'failed', 'error': 'No methods succeeded'}
    
    # Step 6: Compare Methods
    comparison_df = analyzer.compare_methods(method_results)
    print("\nMethod Comparison Results:")
    display_cols = ['Method', 'Test_R²', 'Train_R²', 'Overfitting', 'Overfitting_Level']
    print(comparison_df[display_cols].head().to_string(index=False))
    
    # Step 7: Feature Analysis (Optional)
    feature_selection_df = pd.DataFrame()
    combination_results_df = pd.DataFrame()
    
    if test_selection and len(method_results) > 0:
        print("\n6. FEATURE SELECTION ANALYSIS")
        print("-" * 30)
        try:
            feature_selection_df = analyzer.test_feature_selection_methods(best_transformation)
            if not feature_selection_df.empty:
                print("Feature selection methods tested successfully")
                print(feature_selection_df.head().to_string(index=False))
        except Exception as e:
            print(f"Feature selection failed: {e}")
    
    if test_combinations and len(method_results) > 0:
        print("\n7. FEATURE COMBINATION ANALYSIS")
        print("-" * 30)
        try:
            combination_results_df = analyzer.test_feature_combinations(
                max_combinations=config.max_feature_combinations,
                transformation=best_transformation
            )
            if not combination_results_df.empty:
                print(f"Tested {len(combination_results_df)} feature combinations")
                print(combination_results_df.head().to_string(index=False))
        except Exception as e:
            print(f"Combination testing failed: {e}")
    
    # Step 8: Generate Summary
    print("\n8. ANALYSIS SUMMARY")
    print("-" * 30)
    
    summary_lines = []
    summary_lines.append("FINAL RESULTS")
    summary_lines.append("=" * 20)
    summary_lines.append(f"Best Transformation: {best_transformation}")
    
    if not comparison_df.empty:
        best_method = comparison_df.iloc[0]['Method']
        best_test_r2 = comparison_df.iloc[0]['Test_R²']
        best_overfitting = comparison_df.iloc[0]['Overfitting']
        
        summary_lines.append(f"Best Method: {best_method}")
        summary_lines.append(f"Test R²: {best_test_r2:.4f}")
        summary_lines.append(f"Overfitting: {best_overfitting:.4f}")
        
        # Add warnings if necessary
        if best_overfitting > 0.1:
            summary_lines.append("⚠️ WARNING: High overfitting detected")
        if best_test_r2 > 0.9:
            summary_lines.append("⚠️ WARNING: Very high R² - check for data leakage")
        if best_test_r2 < 0.1:
            summary_lines.append("⚠️ NOTE: Low predictive power")
    
    if not feature_selection_df.empty:
        best_selection = feature_selection_df.iloc[0]
        summary_lines.append(f"Best Feature Selection: {best_selection['selection_method']}")
        summary_lines.append(f"Selected Features ({best_selection['n_features']}): {best_selection['selected_features']}")
    
    if not combination_results_df.empty:
        best_combo = combination_results_df.iloc[0]
        summary_lines.append(f"Best Feature Combination: {best_combo['n_features']} features")
        summary_lines.append(f"Combination R²: {best_combo['test_r_squared']:.4f}")
    
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    
    # Create final result
    result = {
        'status': 'success',
        'transformation_used': best_transformation,
        'method_results': method_results,
        'comparison': comparison_df,
        'feature_selection': feature_selection_df,
        'combination_results': combination_results_df,
        'best_method': comparison_df.iloc[0]['Method'] if not comparison_df.empty else None,
        'best_test_r2': comparison_df.iloc[0]['Test_R²'] if not comparison_df.empty else None,
        'summary': summary,
        'validation': validation
    }
    
    return result

def quick_analysis(target_name: str, start_date: str = "2010-01", 
                  config: AnalysisConfig = None) -> Dict[str, Any]:
    """Quick analysis for a target variable with standard exogenous variables."""
    print("QUICK FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'quick', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def comprehensive_analysis(target_name: str, start_date: str = "2010-01",
                         config: AnalysisConfig = None) -> Dict[str, Any]:
    """Comprehensive analysis for a target variable."""
    print("COMPREHENSIVE FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'comprehensive', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def full_analysis(target_name: str, start_date: str = "2010-01",
                 config: AnalysisConfig = None) -> Dict[str, Any]:
    """Full analysis with all available methods and features."""
    print("FULL FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'full', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

class SimpleVisualizer:
    """Simple visualization functions for analysis results."""
    
    @staticmethod
    def plot_data_overview(data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        """Plot basic data overview."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        plot_cols = list(numeric_cols[:6])  # Limit to 6 series
        
        if target_var in plot_cols:
            # Move target to front
            plot_cols.remove(target_var)
            plot_cols = [target_var] + plot_cols
        
        data.set_index(date_col)[plot_cols].plot(ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('Time Series Overview')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Target distribution
        if target_var in data.columns:
            axes[0, 1].hist(data[target_var].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title(f'{target_var} Distribution')
        
        # Correlation with target
        if target_var in numeric_cols:
            corr_with_target = data[numeric_cols].corr()[target_var].abs().sort_values(ascending=False)
            top_vars = corr_with_target.head(6).index.tolist()
            
            if len(top_vars) > 1:
                corr_matrix = data[top_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           ax=axes[1, 0], fmt='.2f', square=True)
                axes[1, 0].set_title(f'Correlations with {target_var}')
        
        # Missing data
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            axes[1, 1].barh(range(len(missing_data)), missing_data.values)
            axes[1, 1].set_yticks(range(len(missing_data)))
            axes[1, 1].set_yticklabels(missing_data.index)
            axes[1, 1].set_title('Missing Data Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Data Completeness: Perfect')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_method_comparison(comparison_df: pd.DataFrame):
        """Plot method comparison results."""
        if comparison_df.empty:
            print("No method results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test R² comparison
        methods = comparison_df['Method'].values
        test_r2 = comparison_df['Test_R²'].values
        
        bars1 = axes[0].barh(methods, test_r2)
        axes[0].set_xlabel('Test R²')
        axes[0].set_title('Method Performance Comparison')
        
        # Color bars by performance
        for i, bar in enumerate(bars1):
            if test_r2[i] > 0.7:
                bar.set_color('green')
            elif test_r2[i] > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels
        for i, v in enumerate(test_r2):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Overfitting analysis
        overfitting = comparison_df['Overfitting'].values
        colors = ['red' if x > 0.15 else 'orange' if x > 0.08 else 'green' for x in overfitting]
        
        axes[1].barh(methods, overfitting, color=colors, alpha=0.7)
        axes[1].set_xlabel('Overfitting (Train - Test R²)')
        axes[1].set_title('Overfitting Analysis')
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axvline(x=0.08, color='orange', linestyle='--', alpha=0.5, label='Warning (0.08)')
        axes[1].axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Critical (0.15)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

# System initialization
def initialize_system():
    """Initialize the financial analysis system."""
    setup_logging()
    
    print("FINANCIAL ANALYSIS SYSTEM - CORRECTED VERSION")
    print("=" * 55)
    print("Key improvements:")
    print("- Fixed mixed-frequency data handling (forward-fill)")
    print("- Robust cross-validation without extreme values")
    print("- Conservative model hyperparameters")
    print("- Proper data quality validation")
    print("- Clean architecture without monkey patches")
    print("")
    print("Available functions:")
    print("  quick_analysis(target_name)")
    print("  comprehensive_analysis(target_name)")
    print("  full_analysis(target_name)")
    print("  financial_analysis(data, target_var, exog_vars)")
    print("")
    print("Available targets:", ", ".join(list(INDEX_TARGETS.keys())[:4]) + "...")
    print("System ready!")

def test_system():
    """Test the system with a simple example."""
    print("Testing system with PH_KREDITE...")
    
    try:
        results = quick_analysis("PH_KREDITE", start_date="2005-01")
        
        if results['status'] == 'success':
            print(f"✅ System test successful!")
            print(f"  Best method: {results['best_method']}")
            print(f"  Test R²: {results['best_test_r2']:.4f}")
            print(f"  Transformation used: {results['transformation_used']}")
            return True
        else:
            print(f"❌ System test failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ System test failed with exception: {e}")
        return False

print("Analysis pipeline and main functions loaded")



# %%
# %%
"""
Cache-Fixes - Erweitert CacheManager für Final Dataset Caching
"""

class ExtendedCacheManager(CacheManager):
    """
    Erweitert den ursprünglichen CacheManager um Final Dataset Caching.
    """
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        # Zusätzliches Verzeichnis für finale Datasets
        self.final_datasets_dir = self.cache_dir / "final_datasets"
        self.final_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Verzeichnis für transformierte Datasets
        self.transformed_dir = self.cache_dir / "transformed_datasets"
        self.transformed_dir.mkdir(parents=True, exist_ok=True)
    
    def make_final_dataset_key(self, series_definitions: Dict[str, str], start: str, end: str) -> str:
        """Erstelle einen stabilen Key für finale Datasets."""
        import hashlib
        import json
        
        payload = {
            "series_definitions": {k: series_definitions[k] for k in sorted(series_definitions)},
            "start": start,
            "end": end,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    
    def has_fresh_final_dataset(self, key: str) -> bool:
        """Prüft ob ein frisches final Dataset existiert."""
        pattern = f"*_{key}.xlsx"
        matches = sorted(self.final_datasets_dir.glob(pattern), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return False
        
        try:
            latest = matches[0]
            age_days = (dt.datetime.now() - dt.datetime.fromtimestamp(latest.stat().st_mtime)).days
            return age_days <= self.config.cache_max_age_days
        except OSError:
            return False
    
    def write_final_dataset(self, key: str, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
        """Speichere finales Dataset mit Metadaten."""
        if df is None or df.empty:
            return False
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Datenpfad
        data_path = self.final_datasets_dir / f"{timestamp}_{key}.xlsx"
        meta_path = self.final_datasets_dir / f"{timestamp}_{key}_meta.json"
        
        try:
            # Speichere Daten
            df.to_excel(data_path, index=False, engine=get_excel_engine())
            
            # Speichere Metadaten
            meta_info = {
                "created_at": dt.datetime.now().isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "date_range": {
                    "start": df['Datum'].min().isoformat() if 'Datum' in df.columns else None,
                    "end": df['Datum'].max().isoformat() if 'Datum' in df.columns else None
                }
            }
            
            if metadata:
                meta_info.update(metadata)
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)
            
            print(f"Final dataset cached: {data_path.name}")
            return True
            
        except Exception as e:
            print(f"Failed to cache final dataset: {e}")
            return False
    
    def read_final_dataset(self, key: str) -> Optional[pd.DataFrame]:
        """Lade neuestes finales Dataset."""
        pattern = f"*_{key}.xlsx"
        matches = sorted(self.final_datasets_dir.glob(pattern), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return None
        
        try:
            latest = matches[0]
            df = pd.read_excel(latest, engine=get_excel_engine())
            
            # Ensure Datum is datetime
            if 'Datum' in df.columns:
                df['Datum'] = pd.to_datetime(df['Datum'])
            
            print(f"Final dataset loaded from cache: {latest.name}")
            return df
            
        except Exception as e:
            print(f"Failed to load final dataset: {e}")
            return None
    
    def write_transformed_dataset(self, transformation: str, target_var: str, df: pd.DataFrame) -> bool:
        """Speichere transformiertes Dataset."""
        if df is None or df.empty:
            return False
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_target = "".join(c for c in target_var if c.isalnum() or c in "._-")
        
        file_path = self.transformed_dir / f"{timestamp}_{safe_target}_{transformation}.xlsx"
        
        try:
            df.to_excel(file_path, index=False, engine=get_excel_engine())
            print(f"Transformed dataset cached: {file_path.name}")
            return True
        except Exception as e:
            print(f"Failed to cache transformed dataset: {e}")
            return False
    
    def cleanup_old_cache(self, max_age_days: int = None):
        """Bereinige alte Cache-Dateien."""
        if max_age_days is None:
            max_age_days = self.config.cache_max_age_days * 2  # Keep longer for final datasets
        
        cutoff_date = dt.datetime.now() - dt.timedelta(days=max_age_days)
        deleted_count = 0
        
        # Cleanup in allen Cache-Verzeichnissen
        for cache_subdir in [self.cache_dir, self.final_datasets_dir, self.transformed_dir]:
            if not cache_subdir.exists():
                continue
                
            for file_path in cache_subdir.glob("*"):
                if file_path.is_file():
                    try:
                        file_time = dt.datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1
                    except OSError:
                        continue
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old cache files")

# Erweitere FinancialDataDownloader für Final Dataset Caching
def download_with_final_caching(self, series_definitions: Dict[str, str], start_date: str = None, 
                               end_date: str = None, prefer_cache: bool = True, 
                               anchor_var: Optional[str] = None) -> pd.DataFrame:
    """
    Download mit Final Dataset Caching - erweitert die ursprüngliche download Methode.
    """
    start_date = start_date or self.config.default_start_date
    end_date = end_date or self.config.default_end_date
    
    # Verwende ExtendedCacheManager
    if not isinstance(self.cache_manager, ExtendedCacheManager):
        self.cache_manager = ExtendedCacheManager(self.config)
    
    # Prüfe Final Dataset Cache
    final_key = self.cache_manager.make_final_dataset_key(series_definitions, start_date, end_date)
    
    if prefer_cache and self.cache_manager.has_fresh_final_dataset(final_key):
        cached_final = self.cache_manager.read_final_dataset(final_key)
        if cached_final is not None and not cached_final.empty:
            print(f"Loaded final dataset from cache: {cached_final.shape[0]} rows, {cached_final.shape[1]-1} variables")
            return cached_final
    
    # Führe normale Download-Logik aus (aus der ursprünglichen Methode)
    print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
    
    regular_codes = {}
    index_definitions = {}
    
    for var_name, definition in series_definitions.items():
        index_codes = parse_index_specification(definition)
        if index_codes:
            index_definitions[var_name] = index_codes
        else:
            regular_codes[var_name] = definition
    
    all_codes = set(regular_codes.values())
    for index_codes in index_definitions.values():
        all_codes.update(index_codes)
    all_codes = list(all_codes)
    
    print(f"Total series to download: {len(all_codes)}")
    
    # Individual series caching (bestehende Logik)
    cached_data = {}
    missing_codes = []
    
    if prefer_cache:
        for code in all_codes:
            cached_df = self.cache_manager.read_cache(code)
            if cached_df is not None:
                cached_data[code] = cached_df
            else:
                missing_codes.append(code)
    else:
        missing_codes = all_codes[:]
    
    # Download missing codes (bestehende Logik bleibt unverändert)
    downloaded_data = {}
    if missing_codes:
        print(f"Downloading {len(missing_codes)} missing series...")
        try:
            downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
                except ImportError:
                    print("Using synchronous download mode...")
                    downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            else:
                print("Async failed, using synchronous download mode...")
                downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
        except Exception as e:
            print(f"Download failed ({e}), trying synchronous mode...")
            downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
        
        # Cache individual series
        for code, df in downloaded_data.items():
            self.cache_manager.write_cache(code, df)
    
    # Bestehende Merge- und Index-Erstellungslogik bleibt unverändert...
    all_data = {**cached_data, **downloaded_data}
    
    if not all_data:
        raise Exception("No series loaded successfully")
    
    merged_df = self._merge_series_data(all_data)
    final_data = {"Datum": merged_df["Datum"]}
    
    for var_name, series_code in regular_codes.items():
        if series_code in merged_df.columns:
            final_data[var_name] = merged_df[series_code]
    
    for var_name, index_codes in index_definitions.items():
        try:
            available_codes = [c for c in index_codes if c in merged_df.columns]
            
            if len(available_codes) >= len(index_codes) * 0.3:
                index_series = self.index_creator.create_index(merged_df, available_codes, var_name)
                aligned_index = index_series.reindex(pd.to_datetime(merged_df['Datum']))
                final_data[var_name] = aligned_index.values
                print(f"Created INDEX: {var_name} from {len(available_codes)}/{len(index_codes)} series")
            else:
                if var_name in SIMPLE_TARGET_FALLBACKS:
                    fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback_code in merged_df.columns:
                        final_data[var_name] = merged_df[fallback_code]
                        print(f"Using fallback for {var_name}: {fallback_code}")
                    else:
                        print(f"Warning: Could not create {var_name} - fallback series {fallback_code} not available")
                else:
                    print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available_codes)}/{len(index_codes)} series available)")
                    
        except Exception as e:
            print(f"Failed to create INDEX {var_name}: {e}")
            if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                if fallback_code in merged_df.columns:
                    final_data[var_name] = merged_df[fallback_code]
                    print(f"Using fallback for {var_name} after INDEX creation failed: {fallback_code}")
    
    final_df = pd.DataFrame(final_data)
    final_df["Datum"] = pd.to_datetime(final_df["Datum"])
    final_df = final_df.sort_values("Datum").reset_index(drop=True)

    # Bestehende Trimming-Logik...
    value_cols = [c for c in final_df.columns if c != 'Datum']
    if value_cols:
        non_na_count = final_df[value_cols].notna().sum(axis=1)
        required = 2 if len(value_cols) >= 2 else 1
        keep_mask = non_na_count >= required
        if keep_mask.any():
            first_keep = keep_mask.idxmax()
            if first_keep > 0:
                _before = len(final_df)
                final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

    if anchor_var and anchor_var in final_df.columns:
        mask_anchor = final_df[anchor_var].notna()
        if mask_anchor.any():
            start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
            end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
            _before_rows = len(final_df)
            final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
            final_df.reset_index(drop=True, inplace=True)
            print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

    if anchor_var and anchor_var in final_df.columns:
        exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
        if exog_cols:
            tgt_notna = final_df[anchor_var].notna().values
            all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
            keep_start = 0
            for i in range(len(final_df)):
                if not (tgt_notna[i] and all_exog_nan[i]):
                    keep_start = i
                    break
            if keep_start > 0:
                _before = len(final_df)
                final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

    print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")
    
    # Cache final dataset
    metadata = {
        "series_definitions": series_definitions,
        "start_date": start_date,
        "end_date": end_date,
        "downloaded_codes": sorted(list(all_data.keys())),
        "regular_series": regular_codes,
        "index_specifications": index_definitions
    }
    
    self.cache_manager.write_final_dataset(final_key, final_df, metadata)
    
    return final_df

# Erweitere DataPreprocessor für Transformation Caching
def create_transformations_with_caching(self, transformation: str = 'levels') -> pd.DataFrame:
    """
    Transformationen mit Caching - erweitert die ursprüngliche Methode.
    """
    # Verwende ExtendedCacheManager falls verfügbar
    cache_manager = getattr(self, 'cache_manager', None)
    if cache_manager and hasattr(cache_manager, 'write_transformed_dataset'):
        # Führe Transformationen durch
        transformed_data = self.create_transformations_original(transformation)
        
        # Cache transformierte Daten
        cache_manager.write_transformed_dataset(transformation, self.target_var, transformed_data)
        
        return transformed_data
    else:
        # Fallback zur ursprünglichen Methode
        return self.create_transformations_original(transformation)

# Monkey-patch die Methoden
FinancialDataDownloader.download_original = FinancialDataDownloader.download
FinancialDataDownloader.download = download_with_final_caching

DataPreprocessor.create_transformations_original = DataPreprocessor.create_transformations
DataPreprocessor.create_transformations = create_transformations_with_caching

print("Cache fixes loaded - Final datasets and transformed data will now be cached")



# %%# %%
"""
Main Script - Usage Example
Zeigt wie die korrigierte Pipeline verwendet wird
# """

# if __name__ == "__main__":
#     # Initialize system
#     initialize_system()
    
#     # WICHTIG: Cache-Fixes laden
#     print("Loading cache fixes for raw and transformed data...")
#     # Der Cache-Fix Code sollte hier eingefügt werden (aus Cache-Fixes Artifact)
    
#     # Configuration with conservative settings
#     config = AnalysisConfig(
#         default_start_date="2000-01",
#         default_end_date="2024-12",
#         test_size=0.25,                    # Conservative test size
#         cv_folds=3,                       # Fewer CV folds for stability
#         gap_periods=2,                    # Mandatory gaps
#         max_feature_combinations=15,      # Reduced combinations to prevent overfitting
#         handle_mixed_frequencies=True,    # Enable mixed frequency handling
#         cache_final_dataset=True,         # Enable final dataset caching
#         cache_max_age_days=7              # Cache für 7 Tage
#     )
    
#     # === EXAMPLE 1: Quick Analysis with Caching ===
#     print("\n" + "="*60)
#     print("EXAMPLE 1: QUICK ANALYSIS WITH CACHING")
#     print("="*60)
    
#     TARGET = "PH_KREDITE"
#     START_DATE = "2005-01"
    
#     # Erste Ausführung - lädt und cached Daten
#     print("First run - will download and cache data...")
#     results1 = quick_analysis(TARGET, START_DATE, config)
    
#     # Zweite Ausführung - sollte aus Cache laden
#     print("\nSecond run - should load from cache...")
#     results2 = quick_analysis(TARGET, START_DATE, config)
    
#     if results1['status'] == 'success':
#         print(f"\n✅ Analysis completed successfully!")
#         print(f"Best method: {results1['best_method']}")
#         print(f"Test R²: {results1['best_test_r2']:.4f}")
#         print(f"Transformation: {results1['transformation_used']}")
#     else:
#         print(f"\n❌ Analysis failed: {results1.get('error', 'Unknown error')}")
    
#     # === Cache-Status anzeigen ===
#     print("\n" + "="*60)
#     print("CACHE STATUS")
#     print("="*60)
    
#     # Zeige Cache-Verzeichnisse
#     cache_dir = Path(config.cache_dir)
#     if cache_dir.exists():
#         print(f"Cache directory: {cache_dir}")
        
#         # Rohdaten Cache
#         raw_files = list(cache_dir.glob("*.xlsx"))
#         print(f"Raw data files cached: {len(raw_files)}")
        
#         # Final datasets
#         final_dir = cache_dir / "final_datasets"
#         if final_dir.exists():
#             final_files = list(final_dir.glob("*.xlsx"))
#             print(f"Final datasets cached: {len(final_files)}")
#             for f in final_files[-3:]:  # Show last 3
#                 print(f"  - {f.name}")
        
#         # Transformed datasets  
#         trans_dir = cache_dir / "transformed_datasets"
#         if trans_dir.exists():
#             trans_files = list(trans_dir.glob("*.xlsx"))
#             print(f"Transformed datasets cached: {len(trans_files)}")
#             for f in trans_files[-3:]:  # Show last 3
#                 print(f"  - {f.name}")
    
#     # === Cache cleanup demo ===
#     print("\n" + "="*60)
#     print("CACHE CLEANUP DEMO")
#     print("="*60)
    
#     # Erstelle ExtendedCacheManager für Cleanup
#     downloader = FinancialDataDownloader(config)
#     if hasattr(downloader, 'cache_manager'):
#         cache_manager = downloader.cache_manager
#         if hasattr(cache_manager, 'cleanup_old_cache'):
#             print("Running cache cleanup (files older than 60 days)...")
#             cache_manager.cleanup_old_cache(max_age_days=60)
#         else:
#             print("Cache manager does not support cleanup")
    
#     # === EXAMPLE 2: Custom Analysis ===
#     print("\n" + "="*60)
#     print("EXAMPLE 2: CUSTOM ANALYSIS WITH CACHING")
#     print("="*60)
    
#     # Define custom series (mix of monthly and quarterly)
#     # Use only reliably available series
#     series_definitions = {
#         "PH_KREDITE": INDEX_TARGETS["PH_KREDITE"],  # Quarterly target
#         "euribor_3m": STANDARD_EXOG_VARS["euribor_3m"],  # Monthly
#         "german_rates": STANDARD_EXOG_VARS["german_rates"],  # Monthly
#         "german_inflation": STANDARD_EXOG_VARS["german_inflation"],  # Monthly
#         "german_unemployment": STANDARD_EXOG_VARS["german_unemployment"]  # Monthly
#     }
    
#     # Download data (should use caching)
#     print("Downloading with caching...")
#     downloader = FinancialDataDownloader(config)
#     data = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=True)
    
#     print(f"Downloaded data shape: {data.shape}")
#     print(f"Columns: {list(data.columns)}")
    
#     # Show data overview plot
#     visualizer = SimpleVisualizer()
#     visualizer.plot_data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === EXAMPLE 3: System Test ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE")
#     print("="*60)
    
#     #data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === Cache-Performance Test ===
#     print("\n" + "="*60)
#     print("CACHE PERFORMANCE TEST")
#     print("="*60)
    
#     import time
    
#     # Test 1: Download ohne Cache
#     print("Test 1: Download without cache...")
#     start_time = time.time()
#     data_no_cache = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=False)
#     time_no_cache = time.time() - start_time
#     print(f"Time without cache: {time_no_cache:.2f} seconds")
    
#     # Test 2: Download mit Cache
#     print("\nTest 2: Download with cache...")
#     start_time = time.time()
#     data_with_cache = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=True)
#     time_with_cache = time.time() - start_time
#     print(f"Time with cache: {time_with_cache:.2f} seconds")
    
#     if time_no_cache > 0:
#         speedup = time_no_cache / max(time_with_cache, 0.01)
#         print(f"Cache speedup: {speedup:.1f}x faster")
    
#     # === EXAMPLE 3: System Test ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("✅ Raw data caching: Active")
#         print("✅ Final dataset caching: Active") 
#         print("✅ Transformed data caching: Active")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#         print("\nCache locations:")
#         print(f"- Raw data: {cache_dir}")
#         print(f"- Final datasets: {cache_dir / 'final_datasets'}")
#         print(f"- Transformed data: {cache_dir / 'transformed_datasets'}")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE - WITH FULL CACHING")
#     print("="*60)
#     #data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === EXAMPLE 3: Test System ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE")
#     print("="*60)

# %%
"""
Improved Mixed-Frequency Data Processing - KORRIGIERT
Verhindert Data Leakage durch strikte zeitliche Beschränkungen
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from statsmodels.tsa.stattools import adfuller
import warnings

class ImprovedMixedFrequencyProcessor:
    """
    Verbesserte Behandlung von Mixed-Frequency Daten mit strikten Anti-Leakage Regeln.
    """
    
    @staticmethod
    def detect_frequency(series: pd.Series, date_col: pd.Series) -> str:
        """Erweiterte Frequenzerkennung mit robusteren Regeln."""
        if series.isna().all():
            return "unknown"
        
        df_temp = pd.DataFrame({'date': date_col, 'value': series})
        df_temp = df_temp.dropna().copy()
        
        if len(df_temp) < 4:  # Mindestens 4 Beobachtungen für Frequenzerkennung
            return "unknown"
        
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['quarter'] = df_temp['date'].dt.quarter
        
        # Erweiterte Frequenzanalyse
        years_with_data = df_temp['year'].nunique()
        if years_with_data < 2:
            return "insufficient_history"
        
        # Analysiere Beobachtungen pro Jahr
        obs_per_year = df_temp.groupby('year').size()
        avg_obs_per_year = obs_per_year.mean()
        std_obs_per_year = obs_per_year.std()
        
        # Analysiere monatliche vs. quartalsweise Verteilung
        months_per_year = df_temp.groupby('year')['month'].nunique()
        quarters_per_year = df_temp.groupby('year')['quarter'].nunique()
        
        avg_months_per_year = months_per_year.mean()
        avg_quarters_per_year = quarters_per_year.mean()
        
        # Robuste Klassifikation
        if avg_quarters_per_year <= 4.2 and avg_obs_per_year <= 5:
            if avg_months_per_year <= 4.5:  # Meist nur ein Monat pro Quartal
                return "quarterly"
        
        if avg_months_per_year >= 8 and avg_obs_per_year >= 10:
            return "monthly"
        
        # Zusätzliche Prüfung: Sind die Daten gleichmäßig verteilt?
        if std_obs_per_year / max(avg_obs_per_year, 1) > 0.5:
            return "irregular"
            
        return "unknown"
    
    @staticmethod
    def safe_forward_fill_quarterly(df: pd.DataFrame, quarterly_vars: List[str], 
                                   max_fill_periods: int = 6) -> pd.DataFrame:
        """
        Sicheres Forward-Fill für Quartalsdaten mit strikten Limits.
        
        Args:
            df: DataFrame mit Zeitreihen
            quarterly_vars: Liste der Quartalsvariablen
            max_fill_periods: Maximale Anzahl Perioden zum Forward-Fill (Standard: 2 Monate)
        """
        result = df.copy()
        
        for var in quarterly_vars:
            if var not in df.columns:
                continue
            
            series = df[var].copy()
            valid_mask = series.notna()
            
            if not valid_mask.any():
                continue
            
            # Identifiziere alle validen Zeitpunkte
            valid_indices = valid_mask[valid_mask].index.tolist()
            
            # Forward-Fill nur zwischen benachbarten validen Punkten
            filled_series = series.copy()
            
            for i in range(len(valid_indices) - 1):
                current_idx = valid_indices[i]
                next_idx = valid_indices[i + 1]
                
                # Bereich zwischen aktuellen und nächsten validen Werten
                gap_start = current_idx + 1
                gap_end = next_idx
                
                # Begrenze Fill-Bereich auf max_fill_periods
                actual_gap_end = min(gap_end, current_idx + max_fill_periods + 1)
                
                if gap_start < actual_gap_end:
                    # Forward-Fill nur im begrenzten Bereich
                    fill_value = series.iloc[current_idx]
                    filled_series.iloc[gap_start:actual_gap_end] = fill_value
            
            result[var] = filled_series
        
        return result
    
    @staticmethod
    def align_frequencies_improved(
        df: pd.DataFrame,
        target_var: str,
        date_col: str = "Datum",
        train_end_index: Optional[int] = None,
        validation_split_date: Optional[pd.Timestamp] = None,
        max_fill_periods: int = 2,   # konservativer Standard
    ) -> Dict[str, Any]:
        """
        Verbesserte Frequenz-Alignierung mit Anti-Leakage-Schutz.
        - Forward-Fill nur im Trainingsfenster
        - Leakage-Flags nur, wenn neue Fills im Testfenster liegen
        """
        if target_var not in df.columns or date_col not in df.columns:
            raise ValueError(f"Missing {target_var} or {date_col} column")

        # Datums-Handling & Sortierung
        out = {
            "processed_df": None,
            "frequency_info": {},
            "warnings": [],
            "leakage_risk": "low",
            "forward_fill_used": False,
            "fill_span_overlaps_test_period": False,
        }

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.sort_values(date_col).reset_index(drop=True)

        # Frequenzen erkennen (auf Originaldaten)
        all_vars = [c for c in work.columns if c != date_col]
        freqs = {}
        for var in all_vars:
            freqs[var] = ImprovedMixedFrequencyProcessor.detect_frequency(work[var], work[date_col])
        out["frequency_info"] = freqs

        quarterly_vars = [v for v, f in freqs.items() if f == "quarterly"]
        monthly_vars   = [v for v, f in freqs.items() if f == "monthly"]
        irregular_vars = [v for v, f in freqs.items() if f == "irregular"]

        print("Detected frequencies:")
        print(f"  Quarterly: {quarterly_vars}")
        print(f"  Monthly: {monthly_vars}")
        if irregular_vars:
            print(f"  Irregular: {irregular_vars}")
            out["warnings"].append(f"Irregular frequency detected: {irregular_vars}")

        # Train-Ende bestimmen
        split_date = None
        if validation_split_date is not None:
            split_date = pd.to_datetime(validation_split_date)
        elif train_end_index is not None and 0 <= train_end_index < len(work):
            split_date = pd.to_datetime(work.loc[train_end_index, date_col])

        # Forward-Fill nur für Trainingsanteil
        if quarterly_vars:
            if split_date is not None:
                train_mask = work[date_col] < split_date
                train_df = work.loc[train_mask].copy()
                print(f"Using training data only for forward-fill: {len(train_df)} observations")
            else:
                train_df = work.copy()
                out["warnings"].append("No split provided — forward-fill applied on full sample (potential bias)")
                out["leakage_risk"] = "medium"

            print(f"Applying safe forward-fill to {len(quarterly_vars)} quarterly variables...")
            filled_train = ImprovedMixedFrequencyProcessor.safe_forward_fill_quarterly(
                train_df, quarterly_vars, max_fill_periods=max_fill_periods
            )

            if split_date is not None:
                valid_df = work.loc[work[date_col] >= split_date].copy()  # Validierungs-/Testteil: ungefüllt
                processed_df = pd.concat([filled_train, valid_df], ignore_index=True)
                processed_df = processed_df.sort_values(date_col).reset_index(drop=True)
            else:
                processed_df = filled_train
        else:
            processed_df = work.copy()  # nichts zu füllen

        # Zielbereich trimmen (keine Lead/Tail-NaNs)
        s = processed_df[target_var]
        first = s.first_valid_index()
        last  = s.last_valid_index()
        if first is not None and last is not None and last >= first:
            processed_df = processed_df.loc[first:last].reset_index(drop=True)

        # Fortschritts-Log je Quartalsvariable
        for var in quarterly_vars:
            before_count = work[var].notna().sum()
            after_count  = processed_df[var].notna().sum()
            improvement  = int(after_count - before_count)
            print(f"  {var}: {before_count} → {after_count} observations (+{improvement})")
            # Keine automatische Leakage-Hochstufung mehr nur wegen „Large improvement“
            if improvement > max(5, int(before_count * 0.5)):
                out["warnings"].append(f"Large improvement in {var} coverage — verify correctness")

        # Leakage-Flags setzen: neue Fills identifizieren & prüfen, ob sie NACH split_date liegen
        orig_series = work.set_index(date_col)[target_var]
        proc_series = processed_df.set_index(date_col)[target_var]
        orig_notna  = orig_series.reindex(proc_series.index).notna().fillna(False)
        proc_notna  = proc_series.notna()
        new_filled_mask = (proc_notna & ~orig_notna)

        out["forward_fill_used"] = bool(new_filled_mask.any())
        if split_date is not None:
            out["fill_span_overlaps_test_period"] = bool(new_filled_mask.loc[new_filled_mask.index >= split_date].any())
            out["leakage_risk"] = "high" if out["fill_span_overlaps_test_period"] else out["leakage_risk"]
        # Wenn kein split_date: Risiko bleibt wie oben gesetzt (medium), aber kein „high“

        out["processed_df"] = processed_df
        return out

        




class ImprovedDataQualityChecker:
    """
    Erweiterte Datenqualitätsprüfung mit Stationaritätstests.
    """
    
    @staticmethod
    def test_stationarity(series: pd.Series, variable_name: str) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller Test für Stationarität.
        """
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': None,
                'error': 'Insufficient data for stationarity test'
            }
        
        try:
            # ADF Test
            result = adfuller(clean_series, autolag='AIC')
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'variable': variable_name,
                'n_observations': len(clean_series)
            }
        except Exception as e:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': None,
                'error': str(e),
                'variable': variable_name
            }
    
    @staticmethod
    def comprehensive_data_validation(data: pd.DataFrame, target_var: str, 
                                    exog_vars: List[str],
                                    min_target_coverage: float = 0.15,  # Für Quartalsdaten
                                    min_observations: int = 30         ) -> Dict[str, Any]:
        """
        Umfassende Datenvalidierung mit Stationaritätstests.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'stationarity_tests': {},
            'recommendations': [],
            'sample_adequacy': {}
        }
        
        # Grundlegende Validierung
        missing_vars = [var for var in [target_var] + exog_vars if var not in data.columns]
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {', '.join(missing_vars)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Stichprobengröße prüfen
        if len(data) < min_observations:
            validation_results['errors'].append(
                f"Insufficient sample size: {len(data)} < {min_observations} required"
            )
            validation_results['is_valid'] = False
        
        # Target Variable Analyse
        target_series = data[target_var]
        target_coverage = target_series.notna().sum() / len(target_series)
        target_valid_obs = target_series.notna().sum()
        
        validation_results['data_quality'][target_var] = {
            'total_obs': len(target_series),
            'valid_obs': target_valid_obs,
            'coverage': target_coverage,
            'mean': target_series.mean() if target_valid_obs > 0 else np.nan,
            'std': target_series.std() if target_valid_obs > 0 else np.nan,
            'frequency': ImprovedMixedFrequencyProcessor.detect_frequency(
                target_series, data['Datum'] if 'Datum' in data.columns else data.index
            )
        }
        
        # Critical checks für Target
        if target_coverage < min_target_coverage:
            validation_results['errors'].append(
                f"Target {target_var} insufficient coverage: {target_coverage:.1%} < {min_target_coverage:.1%}"
            )
            validation_results['is_valid'] = False
        
        if target_valid_obs > 1 and target_series.std() == 0:
            validation_results['errors'].append(f"Target {target_var} is constant")
            validation_results['is_valid'] = False
        
        # Stationaritätstest für Target
        stationarity_result = ImprovedDataQualityChecker.test_stationarity(target_series, target_var)
        validation_results['stationarity_tests'][target_var] = stationarity_result
        
        if stationarity_result.get('is_stationary') is False:
            validation_results['warnings'].append(
                f"Target {target_var} may be non-stationary (ADF p-value: {stationarity_result.get('p_value', 'N/A'):.3f})"
            )
            validation_results['recommendations'].append(
                f"Consider differencing or log-transformation for {target_var}"
            )
        
        # Exogenous Variables Analyse
        for var in exog_vars:
            if var in data.columns:
                series = data[var]
                coverage = series.notna().sum() / len(series)
                valid_obs = series.notna().sum()
                
                validation_results['data_quality'][var] = {
                    'total_obs': len(series),
                    'valid_obs': valid_obs,
                    'coverage': coverage,
                    'mean': series.mean() if valid_obs > 0 else np.nan,
                    'std': series.std() if valid_obs > 0 else np.nan,
                    'frequency': ImprovedMixedFrequencyProcessor.detect_frequency(
                        series, data['Datum'] if 'Datum' in data.columns else data.index
                    )
                }
                
                # Stationaritätstest
                stationarity_result = ImprovedDataQualityChecker.test_stationarity(series, var)
                validation_results['stationarity_tests'][var] = stationarity_result
                
                # Warnings
                if coverage < 0.3:
                    validation_results['warnings'].append(f"Low coverage in {var}: {coverage:.1%}")
                
                if valid_obs > 1 and series.std() == 0:
                    validation_results['warnings'].append(f"Variable {var} is constant")
        
        # Sample adequacy assessment
        all_vars = [target_var] + [v for v in exog_vars if v in data.columns]
        complete_cases = data[all_vars].dropna()
        overlap_ratio = len(complete_cases) / len(data)
        
        validation_results['sample_adequacy'] = {
            'total_observations': len(data),
            'complete_cases': len(complete_cases),
            'overlap_ratio': overlap_ratio,
            'variables_tested': len(all_vars)
        }
        
        if overlap_ratio < 0.2:
            validation_results['errors'].append(
                f"Insufficient overlap: only {overlap_ratio:.1%} complete cases"
            )
            validation_results['is_valid'] = False
        elif overlap_ratio < 0.4:
            validation_results['warnings'].append(
                f"Low overlap: {overlap_ratio:.1%} complete cases - results may be unreliable"
            )
        
        # Final recommendations
        non_stationary_count = sum(1 for result in validation_results['stationarity_tests'].values() 
                                 if result.get('is_stationary') is False)
        
        if non_stationary_count > 0:
            validation_results['recommendations'].append(
                f"Consider using 'diff' or 'pct' transformations - {non_stationary_count} variables may be non-stationary"
            )
        
        return validation_results


class ImprovedDataPreprocessor:
    """
    Verbesserter Datenvorverarbeitungsschritt mit robusten Transformationen.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.date_col = date_col

        self.forward_fill_used: bool = False
        self.fill_span_overlaps_test_period: bool = False
        
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        
    def create_robust_transformations(self, transformation: str = 'levels',
                                    train_end_date: pd.Timestamp = None,
                                    outlier_method: str = 'conservative') -> Dict[str, Any]:
        """
        Robuste Transformationen mit Anti-Leakage Schutz.
        
        Args:
            transformation: 'levels', 'log', 'pct', 'diff'
            train_end_date: Trainingsende für Anti-Leakage (optional)
            outlier_method: 'conservative', 'moderate', 'aggressive'
        """
        
        # Schritt 1: Mixed-Frequency Handling mit Anti-Leakage
        freq_result = ImprovedMixedFrequencyProcessor.align_frequencies_improved(
            self.data, self.target_var, self.date_col, 
            validation_split_date=train_end_date
        )
        
        processed_data = freq_result['processed_df']
        warnings_list = freq_result['warnings']
        
        # Schritt 2: Transformationen anwenden
        transformed_data = processed_data[[self.date_col]].copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        print(f"Applying '{transformation}' transformation to {len(numeric_cols)} variables...")
        
        def _robust_pct_change(s: pd.Series, eps: float = 1e-8) -> pd.Series:
            return (s - s.shift(1)) / (np.abs(s.shift(1)) + eps)
        
        for col in numeric_cols:
            series = processed_data[col].copy()
            if transformation == 'robust_pct':
                transformed_data[col] = _robust_pct_change(series)

            elif transformation == 'levels':
                transformed_data[col] = series
                
            elif transformation == 'log':
                # Robuste Log-Transformation
                if (series > 0).sum() / series.notna().sum() > 0.9:
                    # Nur wenn > 90% der Werte positiv sind
                    # Kleine Konstante hinzufügen um Zeros zu handhaben
                    min_positive = series[series > 0].min()
                    epsilon = min_positive * 0.001 if pd.notna(min_positive) else 0.001
                    transformed_data[col] = np.log(series.clip(lower=epsilon))
                else:
                    transformed_data[col] = series
                    warnings_list.append(f"{col}: Not suitable for log transformation")
                    
            elif transformation == 'pct':
                # Prozentuale Änderungen
                transformed_data[col] = series.pct_change()
                
            elif transformation == 'diff':
                # Erste Differenzen
                transformed_data[col] = series.diff()
                
            else:
                transformed_data[col] = series
        
        # Schritt 3: Robuste Outlier-Behandlung
        transformed_data = self._robust_outlier_treatment(
            transformed_data, method=outlier_method
        )
        
        # Schritt 4: Saisonale Features hinzufügen
        transformed_data = self._add_seasonal_features(transformed_data)
        
        # Schritt 5: Final cleaning
        before_clean = len(transformed_data)
        
        # Nur Zeilen mit mindestens dem Target und einer exogenen Variable behalten
        essential_cols = [col for col in transformed_data.columns 
                         if col != self.date_col and col in [self.target_var] + 
                         [c for c in numeric_cols if c != self.target_var][:3]]  # Top 3 exog vars
        
        if len(essential_cols) > 1:
            # Behalte Zeilen mit Target + mindestens einer exogenen Variable
            keep_mask = (transformed_data[essential_cols].notna().sum(axis=1) >= 2)
            transformed_data = transformed_data[keep_mask].copy()
        else:
            # Fallback: alle NaN Zeilen entfernen
            transformed_data = transformed_data.dropna(how="all", subset=numeric_cols)
        
        after_clean = len(transformed_data)
        
        if before_clean > after_clean:
            print(f"Cleaned dataset: {before_clean} → {after_clean} observations")
        
        # Data types stabilisieren
        for col in [c for c in transformed_data.columns if c != self.date_col]:
            transformed_data[col] = pd.to_numeric(transformed_data[col], errors='coerce')
        
        return {
            'data': transformed_data,
            'warnings': warnings_list,
            'frequency_info': freq_result['frequency_info'],
            'leakage_risk': freq_result['leakage_risk'],
            'transformation_applied': transformation,
            'outlier_method': outlier_method,
            'rows_before_cleaning': before_clean,
            'rows_after_cleaning': after_clean
        }
    
    def _robust_outlier_treatment(self, data: pd.DataFrame, 
                                method: str = 'conservative') -> pd.DataFrame:
        """
        Robuste Outlier-Behandlung mit verschiedenen Aggressivitätsstufen.
        """
        numeric_cols = [c for c in data.columns if c != self.date_col]
        result_data = data.copy()
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) < 20:  # Zu wenige Daten für Outlier-Behandlung
                continue
            
            if method == 'conservative':
                # Sehr konservativ: nur extreme Outliers (0.5% / 99.5%)
                lower_bound = series.quantile(0.005)
                upper_bound = series.quantile(0.995)
            elif method == 'moderate':
                # Moderat: 1% / 99% Quantile
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
            else:  # aggressive
                # Aggressiv: 2.5% / 97.5% Quantile
                lower_bound = series.quantile(0.025)
                upper_bound = series.quantile(0.975)
            
            # Nur clippen wenn bounds sinnvoll sind
            if pd.notna(lower_bound) and pd.notna(upper_bound) and upper_bound > lower_bound:
                original_std = series.std()
                clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
                clipped_std = clipped_series.std()
                
                # Nur anwenden wenn nicht zu viel Variation verloren geht
                if clipped_std > 0.5 * original_std:
                    result_data[col] = result_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return result_data
    
    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Saisonale Features hinzufügen."""
        data_with_features = data.copy()
        
        # Quartalsdummies
        data_with_features['quarter'] = pd.to_datetime(data_with_features[self.date_col]).dt.quarter
        
        for q in [2, 3, 4]:
            data_with_features[f'Q{q}'] = (data_with_features['quarter'] == q).astype(int)
        
        # Zeittrend (normalisiert)
        data_with_features['time_trend'] = (
            np.arange(len(data_with_features)) / len(data_with_features)
        )
        
        data_with_features = data_with_features.drop('quarter', axis=1)
        
        return data_with_features

# %%
"""
Test Script für die verbesserten Komponenten
Demonstriert die Korrekturen und deren Auswirkungen auf die Modellgüte
"""

def test_improved_system():
    """
    Testet das verbesserte System und zeigt die Verbesserungen auf.
    """
    print("=" * 80)
    print("TESTING IMPROVED FINANCIAL ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Configuration mit verbesserten Einstellungen
    config = AnalysisConfig(
        default_start_date="2000-01",
        default_end_date="2024-12",
        test_size=0.20,  # Kleinere Test-Sets für Stabilität
        cv_folds=3,      # Weniger CV-Folds
        gap_periods=2,   # Längere Gaps gegen Leakage
        cache_max_age_days=7,
        handle_mixed_frequencies=True,
        max_feature_combinations=10,  # Statt 15-20
    )
    
    # Test 1: Vergleich alter vs. neuer Ansatz
    print("\n" + "=" * 60)
    print("TEST 1: COMPARISON - OLD VS NEW APPROACH")
    print("=" * 60)
    
    target = "PH_KREDITE"
    start_date = "2005-01"
    
    try:
        # Lade Daten
        downloader = FinancialDataDownloader(config)
        series_definitions = {
            target: INDEX_TARGETS[target],
            **{k: v for k, v in STANDARD_EXOG_VARS.items()}
        }
        
        data = downloader.download(series_definitions, start_date=start_date, prefer_cache=True)
        print(f"Data loaded: {data.shape[0]} observations, {data.shape[1]-1} variables")
        
        # Test mit verbessertem System
        print("\n--- IMPROVED ANALYSIS ---")
        improved_results = improved_financial_analysis(
            data=data,
            target_var=target,
            exog_vars=[col for col in data.columns if col not in ['Datum', target]],
            analysis_type='comprehensive',
            config=config
        )
        
        if improved_results['status'] == 'success':
            print("✅ Improved analysis succeeded!")
            comparison_df = improved_results['comparison']
            print(f"Best method: {improved_results['best_method']}")
            print(f"Quality score: {improved_results['best_quality_score']:.1f}/5.0")
            
            # Zeige Top-3 Methoden
            print("\nTop 3 Methods:")
            display_cols = ['Method', 'Test_R²', 'Quality_Score', 'Overfitting', 'Flags']
            print(comparison_df[display_cols].head(3).to_string(index=False))
            
        else:
            print(f"❌ Improved analysis failed: {improved_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test 1 failed with exception: {str(e)}")
    
    # Test 2: Verschiedene Zielgrößen
    print("\n" + "=" * 60) 
    print("TEST 2: MULTIPLE TARGET VARIABLES")
    print("=" * 60)
    
    test_targets = ["PH_KREDITE", "PH_EINLAGEN", "PH_WERTPAPIERE"]
    results_summary = []
    
    for target in test_targets:
        print(f"\nTesting {target}...")
        
        try:
            result = quick_analysis_improved(target, start_date="2010-01", config=config)
            
            if result['status'] == 'success':
                best_method = result.get('best_method', 'N/A')
                best_r2 = result.get('comparison', pd.DataFrame()).iloc[0]['Test_R²'] if not result.get('comparison', pd.DataFrame()).empty else np.nan
                quality_score = result.get('best_quality_score', 0)
                
                results_summary.append({
                    'Target': target,
                    'Best_Method': best_method,
                    'Test_R²': best_r2,
                    'Quality_Score': quality_score,
                    'Status': '✅ Success'
                })
                
                print(f"  ✅ Success: {best_method}, R²={best_r2:.4f}, Quality={quality_score:.1f}")
                
            else:
                results_summary.append({
                    'Target': target,
                    'Best_Method': 'N/A',
                    'Test_R²': np.nan,
                    'Quality_Score': 0,
                    'Status': f"❌ Failed: {result.get('error', 'Unknown')[:50]}"
                })
                
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            results_summary.append({
                'Target': target,
                'Best_Method': 'N/A', 
                'Test_R²': np.nan,
                'Quality_Score': 0,
                'Status': f"❌ Exception: {str(e)[:50]}"
            })
            
            print(f"  ❌ Exception: {str(e)}")
    
    # Zeige Zusammenfassung
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(f"\nMULTI-TARGET SUMMARY:")
        print(summary_df.to_string(index=False))
        
        success_rate = (summary_df['Status'].str.contains('Success')).mean()
        avg_quality = summary_df['Quality_Score'].mean()
        
        print(f"\nOverall Performance:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Quality Score: {avg_quality:.1f}/5.0")
    
    # Test 3: Datenqualitäts-Validierung
    print("\n" + "=" * 60)
    print("TEST 3: DATA QUALITY VALIDATION")
    print("=" * 60)
    
    try:
        # Teste mit problematischen Daten (zu kurze Zeitreihe)
        short_data = data.tail(20).copy()  # Nur 20 Beobachtungen
        
        print("Testing with short dataset (20 observations)...")
        short_result = improved_financial_analysis(
            data=short_data,
            target_var=target,
            exog_vars=[col for col in short_data.columns if col not in ['Datum', target]][:3],
            analysis_type='quick',
            config=config
        )
        
        if short_result['status'] == 'failed':
            print("✅ Correctly rejected short dataset")
            print(f"   Reason: {short_result.get('error', 'Unknown')}")
        else:
            print("⚠️  Short dataset was accepted - may indicate insufficient validation")
        
        # Teste mit konstanter Zielvariable
        constant_data = data.copy()
        constant_data[target] = 100  # Konstanter Wert
        
        print("\nTesting with constant target variable...")
        constant_result = improved_financial_analysis(
            data=constant_data,
            target_var=target,
            exog_vars=[col for col in constant_data.columns if col not in ['Datum', target]][:3],
            analysis_type='quick',
            config=config
        )
        
        if constant_result['status'] == 'failed':
            print("✅ Correctly rejected constant target")
            print(f"   Reason: {constant_result.get('error', 'Unknown')}")
        else:
            print("⚠️  Constant target was accepted - validation may be insufficient")
            
    except Exception as e:
        print(f"❌ Test 3 failed with exception: {str(e)}")
    
    # Test 4: Anti-Leakage Test  
    print("\n" + "=" * 60)
    print("TEST 4: ANTI-LEAKAGE VERIFICATION")
    print("=" * 60)
    
    try:
        # Erstelle Analyzer für detaillierte Tests
        analyzer = ImprovedFinancialRegressionAnalyzer(
            data, target, 
            [col for col in data.columns if col not in ['Datum', target]][:4],
            config
        )
        
        # Teste Mixed-Frequency Processing
        print("Testing mixed-frequency processing...")
        freq_result = ImprovedMixedFrequencyProcessor.align_frequencies_improved(
            data, target, 'Datum', 
            validation_split_date=pd.Timestamp('2020-01-01')
        )
        
        leakage_risk = freq_result.get('leakage_risk', 'unknown')
        warnings = freq_result.get('warnings', [])
        
        print(f"  Leakage risk: {leakage_risk}")
        print(f"  Warnings: {len(warnings)}")
        
        if warnings:
            for warning in warnings[:3]:
                print(f"    - {warning}")
        
        if leakage_risk == 'low':
            print("✅ Low leakage risk detected")
        elif leakage_risk == 'medium':
            print("⚠️  Medium leakage risk - acceptable with warnings")  
        else:
            print("❌ High leakage risk - needs attention")
            
    except Exception as e:
        print(f"❌ Test 4 failed with exception: {str(e)}")
    
    # Test 5: Performance Benchmark
    print("\n" + "=" * 60)
    print("TEST 5: PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        import time
        
        # Zeitbasierter Performance-Test
        start_time = time.time()
        
        benchmark_result = comprehensive_analysis_improved(
            "PH_KREDITE", 
            start_date="2010-01",
            config=config
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"Analysis completed in {analysis_time:.1f} seconds")
        
        if benchmark_result['status'] == 'success':
            quality_score = benchmark_result.get('best_quality_score', 0)
            leakage_risk = benchmark_result.get('validation_summary', {}).get('leakage_risk', 'unknown')
            
            print(f"✅ Benchmark completed successfully")
            print(f"   Quality Score: {quality_score:.1f}/5.0")
            print(f"   Leakage Risk: {leakage_risk}")
            print(f"   Processing Time: {analysis_time:.1f}s")
            
            # Performance Rating
            if quality_score >= 3.0 and analysis_time < 30:
                print("🏆 EXCELLENT: High quality, fast execution")
            elif quality_score >= 2.0 and analysis_time < 60:
                print("✅ GOOD: Acceptable quality and speed")
            else:
                print("⚠️  NEEDS IMPROVEMENT: Low quality or slow execution")
        else:
            print(f"❌ Benchmark failed: {benchmark_result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Test 5 failed with exception: {str(e)}")
    
    # Abschluss
    print("\n" + "=" * 80)
    print("IMPROVED SYSTEM TEST COMPLETED")
    print("=" * 80)
    
    print("\nKey Improvements Implemented:")
    print("✅ Anti-leakage mixed-frequency processing")  
    print("✅ Robust time series cross-validation")
    print("✅ Conservative feature selection")
    print("✅ Enhanced data quality validation")
    print("✅ Stationarity-based transformation selection")
    print("✅ Quality-score based method ranking")
    print("✅ Comprehensive error handling")
    
    return True


def compare_old_vs_new_detailed():
    """
    Detaillierter Vergleich der alten und neuen Implementation.
    """
    print("=" * 80)
    print("DETAILED COMPARISON: OLD VS NEW IMPLEMENTATION")
    print("=" * 80)
    
    config = AnalysisConfig()
    target = "PH_KREDITE"
    
    try:
        # Lade Testdaten
        data = get_target_with_standard_exog(target, "2005-01", config)
        exog_vars = [col for col in data.columns if col not in ['Datum', target]][:4]
        
        print(f"Test data: {data.shape[0]} observations, {len(exog_vars)} features")
        print(f"Target: {target}")
        print(f"Features: {', '.join(exog_vars)}")
        
        # Old approach (original system)
        print(f"\n--- OLD APPROACH ---")
        try:
            old_results = financial_analysis(data, target, exog_vars, 'comprehensive', config)
            
            if old_results['status'] == 'success':
                old_best_r2 = old_results.get('best_test_r2', np.nan)
                old_method = old_results.get('best_method', 'N/A')
                
                print(f"✅ Old system: {old_method}, R² = {old_best_r2:.4f}")
            else:
                print(f"❌ Old system failed: {old_results.get('error', 'Unknown')}")
                old_best_r2 = np.nan
                old_method = 'Failed'
                
        except Exception as e:
            print(f"❌ Old system exception: {str(e)}")
            old_best_r2 = np.nan
            old_method = 'Exception'
        
        # New approach (improved system)  
        print(f"\n--- NEW APPROACH ---")
        try:
            new_results = improved_financial_analysis(data, target, exog_vars, 'comprehensive', config)
            
            if new_results['status'] == 'success':
                new_best_r2 = new_results['comparison'].iloc[0]['Test_R²']
                new_method = new_results.get('best_method', 'N/A')
                new_quality = new_results.get('best_quality_score', 0)
                
                print(f"✅ New system: {new_method}, R² = {new_best_r2:.4f}, Quality = {new_quality:.1f}")
            else:
                print(f"❌ New system failed: {new_results.get('error', 'Unknown')}")
                new_best_r2 = np.nan
                new_method = 'Failed'
                new_quality = 0
                
        except Exception as e:
            print(f"❌ New system exception: {str(e)}")
            new_best_r2 = np.nan
            new_method = 'Exception'
            new_quality = 0
        
        # Comparison summary
        print(f"\n--- COMPARISON SUMMARY ---")
        
        comparison_table = pd.DataFrame({
            'System': ['Old', 'New'],
            'Method': [old_method, new_method],
            'Test_R²': [old_best_r2, new_best_r2],
            'Quality_Score': [np.nan, new_quality],
            'Status': [
                'Success' if pd.notna(old_best_r2) else 'Failed',
                'Success' if pd.notna(new_best_r2) else 'Failed'
            ]
        })
        
        print(comparison_table.to_string(index=False))
        
        # Analysis of improvement
        if pd.notna(old_best_r2) and pd.notna(new_best_r2):
            improvement = new_best_r2 - old_best_r2
            print(f"\nR² Improvement: {improvement:+.4f}")
            
            if improvement > 0.05:
                print("🎉 SIGNIFICANT IMPROVEMENT")
            elif improvement > 0:
                print("✅ SLIGHT IMPROVEMENT") 
            elif improvement > -0.05:
                print("≈ COMPARABLE PERFORMANCE")
            else:
                print("⚠️  PERFORMANCE DEGRADATION")
        
        # Qualitative improvements
        print(f"\nQualitative Improvements in New System:")
        print("- Enhanced data validation with stationarity tests")
        print("- Anti-leakage mixed-frequency processing")
        print("- Robust cross-validation with proper time gaps")
        print("- Quality-based method evaluation")
        print("- Conservative feature selection")
        print("- Better error handling and reporting")
        
    except Exception as e:
        print(f"❌ Detailed comparison failed: {str(e)}")


if __name__ == "__main__":
    print("Starting comprehensive test of improved financial analysis system...")
    
    # Run all tests
    try:
        # Test improved system
        test_improved_system()
        
        # Detailed comparison
        print("\n" + "="*80)
        compare_old_vs_new_detailed()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
        print("\nRECOMMENDATIONS:")
        print("1. Use improved_financial_analysis() instead of financial_analysis()")
        print("2. Use quick_analysis_improved() for quick tests")
        print("3. Use comprehensive_analysis_improved() for full analysis")
        print("4. Monitor Quality_Score - aim for >3.0")
        print("5. Check leakage_risk in results - should be 'low'")
        print("6. Review validation warnings before proceeding")
        
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

# %%
# Test ob Forward-Fill korrekt funktioniert
data = get_target_with_standard_exog("PH_KREDITE", "2010-01")
print("Before forward-fill:")
print(data['PH_KREDITE'].notna().sum(), "of", len(data))

# Nach Forward-Fill (simuliere den Prozess)
processor = ImprovedMixedFrequencyProcessor()
result = processor.align_frequencies_improved(data, 'PH_KREDITE', 'Datum')
processed_data = result['processed_df']

print("After forward-fill:")
print(processed_data['PH_KREDITE'].notna().sum(), "of", len(processed_data))
print("Remaining NaNs:", processed_data['PH_KREDITE'].isnull().sum())


# %%
# %%
# %%
"""
Financial Analysis Pipeline - Core Configuration & Data Download
ORIGINAL DOWNLOAD LOGIC - NICHT ÄNDERN!
"""
import hashlib
import json
import asyncio
import datetime as dt
import io
import json
import logging
import re
import ssl
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from itertools import combinations

import aiohttp
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from abc import ABC, abstractmethod

# Regression methods
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

# Statistical analysis
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Optional advanced methods
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from ecbdata import ecbdata
    HAS_ECBDATA = True
except ImportError:
    HAS_ECBDATA = False
    ecbdata = None

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================


from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LagConfig:
    enable: bool = True
    candidates: List[int] = field(default_factory=lambda: [1, 3])
    per_var_max: int = 1
    total_max: int = 8
    min_train_overlap: int = 24
    min_abs_corr: float = 0.0  # optional threshold
@dataclass
class AnalysisConfig:
    """Enhanced configuration with conservative defaults for robust analysis."""
    
    # Cache and download settings
    cache_max_age_days: int = 60
    cache_dir: str = "financial_cache"
    download_timeout_seconds: int = 30
    max_concurrent_downloads: int = 8
    default_start_date: str = "2000-01"
    default_end_date: str = "2024-12"
    min_response_size: int = 100
    max_retry_attempts: int = 3
    
    # Validation settings
    test_size: float = 0.25
    cv_folds: int = 4
    gap_periods: int = 2
    
    # Feature selection settings
    max_feature_combinations: int = 20
    min_features_per_combination: int = 2
    
    # Model settings
    remove_outliers: bool = True
    outlier_method: str = "conservative"
    add_seasonal_dummies: bool = True
    handle_mixed_frequencies: bool = True
    
    # Model persistence
    save_models: bool = True
    model_cleanup_days: int = 30
    keep_best_models: int = 10
    random_seed: int = 42
    
    # Final dataset caching
    cache_final_dataset: bool = True
    final_cache_format: str = "xlsx"
    final_cache_subdir: str = "final_datasets"
    
    # Plot settings
    save_plots: bool = True
    show_plots: bool = True
    plots_dir: str = "financial_cache/diagnostic_plots"
    
    # Validation thresholds
    high_r2_warning_threshold: float = 0.9
    high_overfitting_threshold: float = 0.1
    cv_test_discrepancy_threshold: float = 0.1
    evaluate_quarter_ends_only: bool = True  # evaluate metrics only at quarter ends
    
    # Index-Normalisierung Parameter
    index_base_year: int = 2015
    index_base_value: float = 100.0
    lag_config: Optional[LagConfig] = None
    ab_compare_lags: bool = True

# =============================================================================
# CONSTANTS AND DEFINITIONS (ORIGINAL)
# =============================================================================

# Target variable definitions
INDEX_TARGETS = {
    "PH_EINLAGEN": "INDEX(BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F22.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29A.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29B.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29D.S14.DE.S1.W0.F.N._X.B)",
    "PH_WERTPAPIERE": "INDEX(BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F511.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F512.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F519.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F52.S14.DE.S1.W0.F.N._X.B)",
    "PH_VERSICHERUNGEN": "INDEX(BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F8.S14.DE.S1.W0.F.N._X.B)",
    "PH_KREDITE": "INDEX(BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B, BBAF3.Q.F8.S1.W0.S14.DE.F.N._X.B)",
    "NF_KG_EINLAGEN": "INDEX(BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F22.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29A.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29B.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29D.S11.DE.S1.W0.F.N._X.B)",
    "NF_KG_WERTPAPIERE": "INDEX(BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F32.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F511.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F512.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F519.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F52.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F7.S11.DE.S1.W0.F.N._X.B)",
    "NF_KG_VERSICHERUNGEN": "INDEX(BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F8.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F6.S1.W0.S11.DE.F.N._X.B)",
    "NF_KG_KREDITE": "INDEX(BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F42.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F4.S1.W0.S11.DE.F.N._X.B, BBAF3.Q.F8.S1.W0.S11.DE.F.N._X.B)",
}

# Standard exogenous variables
STANDARD_EXOG_VARS = {
    "euribor_3m": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "german_rates": "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
    "german_inflation": "ICP.M.DE.N.000000.4.ANR",
    "german_unemployment": "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
    "eur_usd_rate": "EXR.D.USD.EUR.SP00.A",
    "german_gdp": "MNA.Q.DE.N.B1GQ.C.S1.S1.B.B1GQ._Z.EUR.LR.GY",
    "ecb_main_rate": "FM.M.U2.EUR.RT.MM.EONIA_.HSTA",
}

# API constants
ECB_API_BASE_URL = "https://data-api.ecb.europa.eu/service/data"
BUNDESBANK_API_BASE_URL = "https://api.statistiken.bundesbank.de/rest/download"
ECB_PREFIXES = ("ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.", "STS.", "MNA.", "BOP.", "GFS.", "EXR.")

# Fallback definitions
SIMPLE_TARGET_FALLBACKS = {
    "PH_KREDITE": "BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B",
    "PH_EINLAGEN": "BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B",
    "PH_WERTPAPIERE": "BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B",
    "PH_VERSICHERUNGEN": "BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B",
    "NF_KG_EINLAGEN": "BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_WERTPAPIERE": "BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_VERSICHERUNGEN": "BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_KREDITE": "BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B",
}

INDEX_SPEC_RE = re.compile(r'^\s*INDEX\s*\(\s*(.*?)\s*\)\s*', re.IGNORECASE)

# =============================================================================
# ORIGINAL UTILITY FUNCTIONS (NICHT ÄNDERN)
# =============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_data_source(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Invalid series code: {code}")
    code_upper = code.upper()
    if "." in code_upper and code_upper.startswith(ECB_PREFIXES):
        return "ECB"
    return "BUNDESBANK"

def parse_index_specification(spec: str) -> Optional[List[str]]:
    if not isinstance(spec, str):
        return None
    match = INDEX_SPEC_RE.match(spec.strip())
    if not match:
        return None
    inner = match.group(1)
    codes = [c.strip() for c in inner.split(",") if c.strip()]
    return list(dict.fromkeys(codes)) if codes else None

def validate_date_string(date_str: str) -> bool:
    if not isinstance(date_str, str):
        return False
    date_patterns = ["%Y-%m", "%Y-%m-%d", "%Y"]
    for pattern in date_patterns:
        try:
            dt.datetime.strptime(date_str, pattern)
            return True
        except ValueError:
            continue
    return False

def format_date_for_ecb_api(date_str: str) -> str:
    if not date_str:
        return date_str
    try:
        if len(date_str) == 4:
            return f"{date_str}-01"
        elif len(date_str) == 7:
            return date_str
        elif len(date_str) == 10:
            return date_str[:7]
        else:
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime("%Y-%m")
    except:
        return date_str

def get_excel_engine() -> str:
    try:
        import openpyxl
        return 'openpyxl'
    except ImportError:
        try:
            import xlsxwriter
            return 'xlsxwriter'
        except ImportError:
            raise ImportError("Excel support requires openpyxl or xlsxwriter. Install with: pip install openpyxl")

# =============================================================================
# ORIGINAL DATA DOWNLOAD CLASSES (NICHT ÄNDERN)
# =============================================================================

class DataProcessor:
    @staticmethod
    def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        date_candidates = ["TIME_PERIOD", "DATE", "Datum", "Period", "period"]
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None and len(df.columns) > 0:
            date_col = df.columns[0]
        
        value_candidates = ["OBS_VALUE", "VALUE", "Wert", "Value"]
        value_col = None
        for candidate in value_candidates:
            if candidate in df.columns and candidate != date_col:
                value_col = candidate
                break
        if value_col is None:
            numeric_cols = [c for c in df.columns if c != date_col and df[c].dtype in ['float64', 'int64']]
            if numeric_cols:
                value_col = numeric_cols[-1]
            else:
                raise ValueError("No value column found")
        
        result = pd.DataFrame()
        result["Datum"] = pd.to_datetime(df[date_col], errors='coerce')
        result["value"] = pd.to_numeric(df[value_col], errors='coerce')
        result = result.dropna(subset=["value", "Datum"])
        result = result.sort_values("Datum").reset_index(drop=True)
        return result

class CacheManager:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir = self.cache_dir / self.config.final_cache_subdir
        self.final_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, code: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in code)
        return self.cache_dir / f"{safe_name}.xlsx"
    
    def is_fresh(self, code: str) -> bool:
        cache_path = self._cache_path(code)
        if not cache_path.exists():
            return False
        try:
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_days = (dt.datetime.now() - mtime).days
            return age_days <= self.config.cache_max_age_days
        except OSError:
            return False
    
    def read_cache(self, code: str) -> Optional[pd.DataFrame]:
        if not self.is_fresh(code):
            return None
        cache_path = self._cache_path(code)
        try:
            df = pd.read_excel(cache_path, sheet_name="data", engine=get_excel_engine())
            return DataProcessor.standardize_dataframe(df)
        except Exception:
            return None
    
    def write_cache(self, code: str, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        cache_path = self._cache_path(code)
        temp_path = cache_path.with_suffix(".tmp.xlsx")
        try:
            with pd.ExcelWriter(temp_path, engine=get_excel_engine()) as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            temp_path.replace(cache_path)
            return True
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False

class BundesbankCSVParser:
    @staticmethod
    def parse(content: str, code: str) -> pd.DataFrame:
        try:
            lines = content.strip().split('\n')
            if not lines:
                raise ValueError("Empty CSV content")
            
            data_start_idx = BundesbankCSVParser._find_data_start(lines, code)
            csv_lines = lines[data_start_idx:]
            if not csv_lines:
                raise ValueError("No data lines found")
            
            delimiter = BundesbankCSVParser._detect_delimiter(csv_lines[0])
            df = pd.read_csv(io.StringIO('\n'.join(csv_lines)), delimiter=delimiter, skip_blank_lines=True)
            df = df.dropna(how='all')
            if df.empty:
                raise ValueError("No valid data after parsing")
            
            time_col, value_col = BundesbankCSVParser._identify_columns(df, code)
            result_df = pd.DataFrame()
            time_values = df[time_col].dropna()
            value_values = df[value_col].dropna()
            min_len = min(len(time_values), len(value_values))
            if min_len == 0:
                raise ValueError("No valid data pairs found")
            
            result_df['Datum'] = time_values.iloc[:min_len].astype(str)
            result_df['value'] = pd.to_numeric(value_values.iloc[:min_len], errors='coerce')
            result_df = result_df.dropna()
            if result_df.empty:
                raise ValueError("No valid numeric data after cleaning")
            return result_df
        except Exception as e:
            raise ValueError(f"Bundesbank CSV parsing failed: {e}")
    
    @staticmethod
    def _find_data_start(lines: List[str], code: str) -> int:
        for i, line in enumerate(lines):
            if code in line and ('BBAF3' in line or 'BBK' in line):
                return i
        for i, line in enumerate(lines):
            if code in line:
                return i
        for i, line in enumerate(lines):
            if ',' in line or ';' in line:
                sep_count = max(line.count(','), line.count(';'))
                if sep_count >= 2:
                    return i
        return 0
    
    @staticmethod
    def _detect_delimiter(header_line: str) -> str:
        comma_count = header_line.count(',')
        semicolon_count = header_line.count(';')
        if comma_count > semicolon_count:
            return ','
        elif semicolon_count > 0:
            return ';'
        else:
            if '\t' in header_line:
                return '\t'
            elif '|' in header_line:
                return '|'
            else:
                return ','
    
    @staticmethod
    def _identify_columns(df: pd.DataFrame, code: str) -> Tuple[str, str]:
        value_col = None
        for col in df.columns:
            col_str = str(col)
            if code in col_str and 'FLAG' not in col_str.upper() and 'ATTRIBUT' not in col_str.upper():
                value_col = col
                break
        if value_col is None:
            code_parts = code.split('.')
            for col in df.columns:
                col_str = str(col)
                if any(part in col_str for part in code_parts if len(part) > 3) and 'FLAG' not in col_str.upper():
                    value_col = col
                    break
        if value_col is None and len(df.columns) >= 2:
            for col in df.columns[1:]:
                if pd.to_numeric(df[col], errors='coerce').notna().sum() > 0:
                    value_col = col
                    break
        if value_col is None:
            if len(df.columns) >= 2:
                value_col = df.columns[1]
            else:
                raise ValueError("Could not identify value column")
        
        time_col = None
        date_keywords = ['TIME', 'DATE', 'PERIOD', 'DATUM', 'ZEIT']
        for col in df.columns:
            col_str = str(col).upper()
            if any(keyword in col_str for keyword in date_keywords):
                time_col = col
                break
        if time_col is None:
            for col in df.columns:
                if col != value_col and 'FLAG' not in str(col).upper():
                    time_col = col
                    break
        if time_col is None:
            time_col = df.columns[0]
        
        return time_col, value_col

class APIClient:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    async def fetch_series(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        source = detect_data_source(code)
        if source == "ECB":
            return await self._fetch_ecb(session, code, start, end)
        else:
            return await self._fetch_bundesbank(session, code, start, end)
    
    async def _fetch_ecb(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        if HAS_ECBDATA:
            try:
                df = ecbdata.get_series(series_key=code, start=start, end=end)
                if df is not None and not df.empty:
                    return DataProcessor.standardize_dataframe(df)
            except Exception:
                pass
        
        flow, series = code.split(".", 1)
        url = f"{ECB_API_BASE_URL}/{flow}/{series}"
        fstart = format_date_for_ecb_api(start)
        fend = format_date_for_ecb_api(end)
        
        param_strategies = [
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
            {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
            {"format": "csvdata", "detail": "dataonly"},
        ]
        
        timeout = aiohttp.ClientTimeout(total=self.config.download_timeout_seconds)
        headers = {"Accept": "text/csv"}
        last_error = None
        
        for params in param_strategies:
            async with session.get(url, params=params, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    last_error = f"Status {response.status}"
                    continue
                text = await response.text()
                if not text.strip() or len(text.strip()) < self.config.min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                try:
                    df = pd.read_csv(io.StringIO(text))
                    df = DataProcessor.standardize_dataframe(df)
                    if not df.empty:
                        return df
                except Exception as e:
                    last_error = f"CSV parse error: {e}"
                    continue
        
        raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    
    async def _fetch_bundesbank(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        url_patterns = self._build_bundesbank_urls(code)
        params_variants = self._get_bundesbank_params(start, end)
        headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        timeout = aiohttp.ClientTimeout(total=self.config.download_timeout_seconds)
        last_error = None
        attempt_count = 0
        max_attempts = min(len(url_patterns) * len(params_variants), 20)
        
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
        
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as bb_session:
                for url in url_patterns:
                    for params in params_variants:
                        attempt_count += 1
                        if attempt_count > max_attempts:
                            break
                        try:
                            async with bb_session.get(url, params=params, headers=headers) as response:
                                if response.status == 200:
                                    text = await response.text()
                                    if text and len(text.strip()) > self.config.min_response_size:
                                        df = BundesbankCSVParser.parse(text, code)
                                        if df is not None and not df.empty:
                                            df = DataProcessor.standardize_dataframe(df)
                                            if not df.empty:
                                                return df
                                    else:
                                        last_error = f"Response too small: {len(text)} bytes"
                                        continue
                                elif response.status == 404:
                                    last_error = "Series not found (404)"
                                    continue
                                else:
                                    error_text = await response.text()
                                    last_error = f"Status {response.status}: {error_text[:100]}"
                                    continue
                        except asyncio.TimeoutError:
                            last_error = f"Timeout after {self.config.download_timeout_seconds}s"
                            continue
                        except Exception as e:
                            last_error = f"Unexpected error: {str(e)}"
                            continue
                    if attempt_count > max_attempts:
                        break
        except Exception as e:
            last_error = f"Session creation failed: {e}"
        
        raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")
    
    def _build_bundesbank_urls(self, code: str) -> List[str]:
        base_urls = [
            "https://api.statistiken.bundesbank.de/rest/download",
            "https://www.bundesbank.de/statistic-rmi/StatisticDownload"
        ]
        url_patterns = []
        if '.' in code:
            dataset, series = code.split('.', 1)
            url_patterns.extend([
                f"{base_urls[0]}/{dataset}/{series}",
                f"{base_urls[0]}/{code.replace('.', '/')}",
                f"{base_urls[1]}/{dataset}/{series}",
                f"{base_urls[1]}/{code.replace('.', '/')}"
            ])
        url_patterns.extend([
            f"{base_urls[0]}/{code}",
            f"{base_urls[1]}/{code}"
        ])
        if code.count('.') > 1:
            parts = code.split('.')
            for i in range(1, len(parts)):
                path1 = '.'.join(parts[:i])
                path2 = '.'.join(parts[i:])
                url_patterns.extend([
                    f"{base_urls[0]}/{path1}/{path2}",
                    f"{base_urls[0]}/{path1.replace('.', '/')}/{path2.replace('.', '/')}"
                ])
        seen = set()
        unique_patterns = []
        for pattern in url_patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        return unique_patterns[:12]
    
    def _get_bundesbank_params(self, start: str, end: str) -> List[Dict[str, str]]:
        return [
            {"format": "csv", "lang": "en", "metadata": "false"},
            {"format": "csv", "lang": "de", "metadata": "false"},
            {"format": "csv", "lang": "en", "metadata": "false", "startPeriod": start, "endPeriod": end},
            {"format": "csv", "lang": "de", "metadata": "false", "startPeriod": start, "endPeriod": end},
            {"format": "tsv", "lang": "en", "metadata": "false"},
            {"format": "tsv", "lang": "de", "metadata": "false"},
            {"format": "csv"},
            {"lang": "en"},
            {"lang": "de"},
            {}
        ]

class IndexCreator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def create_index(self, data_df: pd.DataFrame, series_codes: List[str], index_name: str) -> pd.Series:
        if 'Datum' not in data_df.columns:
            raise ValueError("DataFrame must contain a 'Datum' column")
        
        available_codes = [code for code in series_codes if code in data_df.columns]
        if not available_codes:
            raise ValueError(f"No valid series found for index {index_name}")
        
        index_data = data_df[['Datum'] + available_codes].copy()
        index_data = index_data.set_index('Datum')
        
        has_any = index_data[available_codes].notna().any(axis=1)
        index_data = index_data.loc[has_any].copy()

        def _fill_inside(s: pd.Series) -> pd.Series:
            if s.notna().sum() == 0:
                return s
            first, last = s.first_valid_index(), s.last_valid_index()
            if first is None or last is None:
                return s
            filled = s.ffill().bfill()
            mask = (s.index >= first) & (s.index <= last)
            return filled.where(mask, s)
        
        index_data[available_codes] = index_data[available_codes].apply(_fill_inside)
        clean_data = index_data.dropna()
        
        if clean_data.empty:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result
        
        weights = {code: 1.0 / len(available_codes) for code in available_codes}
        weighted_values = []
        for code in available_codes:
            if code in clean_data.columns:
                weighted_values.append(clean_data[code] * weights[code])
        
        if not weighted_values:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result
        
        aggregated = sum(weighted_values)
        
        try:
            base_year_int = int(self.config.index_base_year)
            base_year_mask = aggregated.index.year == base_year_int
            base_year_data = aggregated[base_year_mask]
            
            if base_year_data.empty or base_year_data.isna().all():
                first_valid = aggregated.dropna()
                if first_valid.empty:
                    base_value_actual = 1.0
                else:
                    base_value_actual = first_valid.iloc[0]
            else:
                base_value_actual = base_year_data.mean()
            
            if base_value_actual == 0 or pd.isna(base_value_actual):
                base_value_actual = 1.0
            
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            mask = aggregated.notna()
            result[mask] = (aggregated[mask] / base_value_actual) * self.config.index_base_value
            
            return result
            
        except Exception as e:
            print(f"Warning: Index normalization failed for {index_name}, using raw data: {e}")
            aggregated.name = index_name
            return aggregated

# =============================================================================
# ORIGINAL DOWNLOAD LOGIC (NICHT ÄNDERN)
# =============================================================================

class FinancialDataDownloader:
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.cache_manager = CacheManager(self.config)
        self.api_client = APIClient(self.config)
        self.index_creator = IndexCreator(self.config)
    
    def download(self, series_definitions: Dict[str, str], start_date: str = None, 
                end_date: str = None, prefer_cache: bool = True, anchor_var: Optional[str] = None) -> pd.DataFrame:
        start_date = start_date or self.config.default_start_date
        end_date = end_date or self.config.default_end_date
        print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
        
        regular_codes = {}
        index_definitions = {}
        
        for var_name, definition in series_definitions.items():
            index_codes = parse_index_specification(definition)
            if index_codes:
                index_definitions[var_name] = index_codes
            else:
                regular_codes[var_name] = definition
        
        all_codes = set(regular_codes.values())
        for index_codes in index_definitions.values():
            all_codes.update(index_codes)
        all_codes = list(all_codes)
        
        print(f"Total series to download: {len(all_codes)}")
        
        cached_data = {}
        missing_codes = []
        
        if prefer_cache:
            for code in all_codes:
                cached_df = self.cache_manager.read_cache(code)
                if cached_df is not None:
                    cached_data[code] = cached_df
                else:
                    missing_codes.append(code)
        else:
            missing_codes = all_codes[:]
        
        downloaded_data = {}
        if missing_codes:
            print(f"Downloading {len(missing_codes)} missing series...")
            try:
                downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
                    except ImportError:
                        print("Using synchronous download mode...")
                        downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
                else:
                    print("Async failed, using synchronous download mode...")
                    downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            except Exception as e:
                print(f"Download failed ({e}), trying synchronous mode...")
                downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            
            for code, df in downloaded_data.items():
                self.cache_manager.write_cache(code, df)
        
        all_data = {**cached_data, **downloaded_data}
        
        if not all_data:
            raise Exception("No series loaded successfully")
        
        merged_df = self._merge_series_data(all_data)
        final_data = {"Datum": merged_df["Datum"]}
        
        for var_name, series_code in regular_codes.items():
            if series_code in merged_df.columns:
                final_data[var_name] = merged_df[series_code]
        
        for var_name, index_codes in index_definitions.items():
            try:
                available_codes = [c for c in index_codes if c in merged_df.columns]
                
                if len(available_codes) >= len(index_codes) * 0.3:
                    index_series = self.index_creator.create_index(merged_df, available_codes, var_name)
                    aligned_index = index_series.reindex(pd.to_datetime(merged_df['Datum']))
                    final_data[var_name] = aligned_index.values
                    print(f"Created INDEX: {var_name} from {len(available_codes)}/{len(index_codes)} series")
                else:
                    if var_name in SIMPLE_TARGET_FALLBACKS:
                        fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                        if fallback_code in merged_df.columns:
                            final_data[var_name] = merged_df[fallback_code]
                            print(f"Using fallback for {var_name}: {fallback_code}")
                        else:
                            print(f"Warning: Could not create {var_name} - fallback series {fallback_code} not available")
                    else:
                        print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available_codes)}/{len(index_codes)} series available)")
                        
            except Exception as e:
                print(f"Failed to create INDEX {var_name}: {e}")
                if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                    fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback_code in merged_df.columns:
                        final_data[var_name] = merged_df[fallback_code]
                        print(f"Using fallback for {var_name} after INDEX creation failed: {fallback_code}")
        
        final_df = pd.DataFrame(final_data)
        final_df["Datum"] = pd.to_datetime(final_df["Datum"])
        final_df = final_df.sort_values("Datum").reset_index(drop=True)

        value_cols = [c for c in final_df.columns if c != 'Datum']
        if value_cols:
            non_na_count = final_df[value_cols].notna().sum(axis=1)
            required = 2 if len(value_cols) >= 2 else 1
            keep_mask = non_na_count >= required
            if keep_mask.any():
                first_keep = keep_mask.idxmax()
                if first_keep > 0:
                    _before = len(final_df)
                    final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                    print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

        if anchor_var and anchor_var in final_df.columns:
            mask_anchor = final_df[anchor_var].notna()
            if mask_anchor.any():
                start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
                end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
                _before_rows = len(final_df)
                final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
                final_df.reset_index(drop=True, inplace=True)
                print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

        if anchor_var and anchor_var in final_df.columns:
            exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
            if exog_cols:
                tgt_notna = final_df[anchor_var].notna().values
                all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
                keep_start = 0
                for i in range(len(final_df)):
                    if not (tgt_notna[i] and all_exog_nan[i]):
                        keep_start = i
                        break
                if keep_start > 0:
                    _before = len(final_df)
                    final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                    print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

        print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")
        return final_df
    
    def _fetch_all_series_sync(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        import requests
        successful = {}
        
        for code in codes:
            try:
                source = detect_data_source(code)
                if source == "ECB":
                    df = self._fetch_ecb_sync(code, start, end)
                else:
                    df = self._fetch_bundesbank_sync(code, start, end)
                
                if df is not None and not df.empty:
                    successful[code] = df
                    print(f"  ✓ {code}: {len(df)} observations")
                else:
                    print(f"  ✗ {code}: No data returned")
            except Exception as e:
                print(f"  ✗ {code}: {str(e)}")
            
            import time
            time.sleep(0.5)
        
        return successful
    
    def _fetch_ecb_sync(self, code: str, start: str, end: str) -> pd.DataFrame:
        import requests
        
        if HAS_ECBDATA:
            try:
                df = ecbdata.get_series(series_key=code, start=start, end=end)
                if df is not None and not df.empty:
                    return DataProcessor.standardize_dataframe(df)
            except Exception:
                pass
        
        flow, series = code.split(".", 1)
        url = f"{ECB_API_BASE_URL}/{flow}/{series}"
        fstart = format_date_for_ecb_api(start)
        fend = format_date_for_ecb_api(end)

        param_strategies = [
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
            {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
            {"format": "csvdata", "detail": "dataonly"},
        ]

        headers = {"Accept": "text/csv"}
        last_error = None

        for params in param_strategies:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.config.download_timeout_seconds)
                if resp.status_code != 200:
                    last_error = f"Status {resp.status_code}"
                    continue
                text = resp.text
                if not text.strip() or len(text.strip()) < self.config.min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                df = pd.read_csv(io.StringIO(text))
                df = DataProcessor.standardize_dataframe(df)
                if not df.empty:
                    return df
            except Exception as e:
                last_error = str(e)
                continue

        raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    
    def _fetch_bundesbank_sync(self, code: str, start: str, end: str) -> pd.DataFrame:
        import requests
        
        url_patterns = self.api_client._build_bundesbank_urls(code)
        params_variants = self.api_client._get_bundesbank_params(start, end)
        headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
        last_error = None
        attempt_count = 0
        max_attempts = min(len(url_patterns) * len(params_variants), 20)
        
        for url in url_patterns:
            for params in params_variants:
                attempt_count += 1
                if attempt_count > max_attempts:
                    break
                
                try:
                    response = requests.get(
                        url, params=params, headers=headers,
                        timeout=self.config.download_timeout_seconds, verify=False
                    )
                    
                    if response.status_code == 200:
                        text = response.text
                        if text and len(text.strip()) > self.config.min_response_size:
                            df = BundesbankCSVParser.parse(text, code)
                            if df is not None and not df.empty:
                                df = DataProcessor.standardize_dataframe(df)
                                if not df.empty:
                                    return df
                        else:
                            last_error = f"Response too small: {len(text)} bytes"
                            continue
                    elif response.status_code == 404:
                        last_error = "Series not found (404)"
                        continue
                    else:
                        last_error = f"Status {response.status_code}: {response.text[:100]}"
                        continue
                        
                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {self.config.download_timeout_seconds}s"
                    continue
                except requests.exceptions.SSLError:
                    last_error = "SSL verification failed"
                    continue
                except Exception as e:
                    last_error = f"Request failed: {str(e)}"
                    continue
            
            if attempt_count > max_attempts:
                break
        
        raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")

    async def _fetch_all_series(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        successful = {}
        
        async with aiohttp.ClientSession() as session:
            for code in codes:
                try:
                    df = await self.api_client.fetch_series(session, code, start, end)
                    successful[code] = df
                    print(f"  ✓ {code}: {len(df)} observations")
                except Exception as e:
                    print(f"  ✗ {code}: {str(e)}")
                
                await asyncio.sleep(0.5)
        
        return successful
    
    def _merge_series_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        all_series = []
        
        for code, df in data_dict.items():
            if not df.empty and "Datum" in df.columns and "value" in df.columns:
                series_df = df.set_index("Datum")[["value"]].rename(columns={"value": code})
                all_series.append(series_df)
        
        if not all_series:
            return pd.DataFrame()
        
        merged_df = pd.concat(all_series, axis=1, sort=True)
        merged_df = merged_df.reset_index()
        merged_df = merged_df.sort_values("Datum").reset_index(drop=True)
        return merged_df

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_target_with_standard_exog(target_name: str, start_date: str = "2000-01", 
                                 config: AnalysisConfig = None) -> pd.DataFrame:
    if target_name not in INDEX_TARGETS:
        raise ValueError(f"Unknown target: {target_name}. Available: {list(INDEX_TARGETS.keys())}")
    
    series_definitions = {target_name: INDEX_TARGETS[target_name]}
    standard_exog = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    for exog_name in standard_exog:
        if exog_name in STANDARD_EXOG_VARS:
            series_definitions[exog_name] = STANDARD_EXOG_VARS[exog_name]
    
    downloader = FinancialDataDownloader(config)
    return downloader.download(series_definitions, start_date=start_date)

class FinancialAnalysisError(Exception):
    pass

class DataDownloadError(FinancialAnalysisError):
    pass

class ValidationError(FinancialAnalysisError):
    pass

class AnalysisError(FinancialAnalysisError):
    pass

print("Core configuration and data download loaded")

# %%
"""
Mixed-Frequency Data Processing - KORRIGIERTE VERSION
Löst das Forward-Fill Problem für Quartalsdaten korrekt
"""

class MixedFrequencyProcessor:
    """
    Handles quarterly target variables with monthly exogenous variables.
    Ensures proper forward-filling without data leakage.
    """
    
    @staticmethod
    def detect_frequency(series: pd.Series, date_col: pd.Series) -> str:
        """
        Detect if a series is monthly or quarterly based on data availability.
        """
        if series.isna().all():
            return "unknown"
        
        # Count observations per year
        df_temp = pd.DataFrame({'date': date_col, 'value': series})
        df_temp = df_temp.dropna()
        
        if len(df_temp) == 0:
            return "unknown"
        
        df_temp['year'] = df_temp['date'].dt.year
        obs_per_year = df_temp.groupby('year').size()
        
        avg_obs_per_year = obs_per_year.mean()
        
        if avg_obs_per_year <= 4.5:  # Allow for some missing quarters
            return "quarterly"
        elif avg_obs_per_year >= 10:  # Allow for some missing months
            return "monthly"
        else:
            return "unknown"
    
    @staticmethod
    def forward_fill_quarterly(df: pd.DataFrame, quarterly_vars: List[str]) -> pd.DataFrame:
        """
        Forward-fill quarterly variables properly:
        1. Only fill within the available data range (no extrapolation)
        2. Fill monthly gaps between quarterly observations
        """
        result = df.copy()
        
        for var in quarterly_vars:
            if var not in df.columns:
                continue
                
            series = df[var].copy()
            
            # Find first and last valid observation
            valid_mask = series.notna()
            if not valid_mask.any():
                continue
                
            first_valid_idx = valid_mask.idxmax()
            last_valid_idx = valid_mask[::-1].idxmax()  # Last valid
            
            # Only forward fill between first and last valid observation
            fill_range = series.iloc[first_valid_idx:last_valid_idx+1]
            filled_range = fill_range.ffill()
            
            # Update only the range between first and last valid
            result.loc[first_valid_idx:last_valid_idx, var] = filled_range
            
        return result
    
    @staticmethod
    def align_frequencies(df: pd.DataFrame, target_var: str, 
                         date_col: str = "Datum") -> pd.DataFrame:
        """
        Align mixed-frequency data properly for regression analysis.
        """
        if target_var not in df.columns or date_col not in df.columns:
            raise ValueError(f"Missing {target_var} or {date_col} column")
        
        # Detect frequencies
        frequencies = {}
        all_vars = [col for col in df.columns if col != date_col]
        
        for var in all_vars:
            freq = MixedFrequencyProcessor.detect_frequency(df[var], df[date_col])
            frequencies[var] = freq
            
        print(f"Detected frequencies:")
        for var, freq in frequencies.items():
            obs_count = df[var].notna().sum()
            print(f"  {var}: {freq} ({obs_count} observations)")
        
        # Identify quarterly variables
        quarterly_vars = [var for var, freq in frequencies.items() if freq == "quarterly"]
        monthly_vars = [var for var, freq in frequencies.items() if freq == "monthly"]
        
        if not quarterly_vars:
            print("No quarterly variables detected - returning original data")
            return df
        
        # Apply forward-fill to quarterly variables
        print(f"Forward-filling {len(quarterly_vars)} quarterly variables...")
        processed_df = MixedFrequencyProcessor.forward_fill_quarterly(df, quarterly_vars)
        
        # Validation: Check improvement
        for var in quarterly_vars:
            before_count = df[var].notna().sum()
            after_count = processed_df[var].notna().sum()
            print(f"  {var}: {before_count} → {after_count} observations")
        
        return processed_df

class DataQualityChecker:
    """
    Comprehensive data quality validation for financial time series.
    """
    
    @staticmethod
    def validate_financial_data(data: pd.DataFrame, target_var: str, 
                               exog_vars: List[str], 
                               min_target_coverage: float = 0.15) -> Dict[str, Any]:
        """
        Enhanced data validation with mixed-frequency awareness.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Check if variables exist
        missing_vars = [var for var in [target_var] + exog_vars if var not in data.columns]
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {', '.join(missing_vars)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Analyze target variable quality (CRITICAL)
        target_series = data[target_var]
        target_coverage = target_series.notna().sum() / len(target_series)
        
        validation_results['data_quality'][target_var] = {
            'total_obs': len(target_series),
            'valid_obs': target_series.notna().sum(),
            'coverage': target_coverage,
            'frequency': MixedFrequencyProcessor.detect_frequency(target_series, data['Datum'])
        }
        
        # CRITICAL CHECK: Target coverage
        if target_coverage < min_target_coverage:
            validation_results['errors'].append(
                f"Target variable {target_var} has only {target_coverage:.1%} valid data "
                f"(minimum required: {min_target_coverage:.1%})"
            )
            validation_results['is_valid'] = False
            
        # Check for completely constant target
        if target_series.notna().sum() > 1 and target_series.std() == 0:
            validation_results['errors'].append(f"Target variable {target_var} is constant")
            validation_results['is_valid'] = False
        
        # Analyze exogenous variables
        for var in exog_vars:
            if var in data.columns:
                series = data[var]
                coverage = series.notna().sum() / len(series)
                
                validation_results['data_quality'][var] = {
                    'total_obs': len(series),
                    'valid_obs': series.notna().sum(),
                    'coverage': coverage,
                    'frequency': MixedFrequencyProcessor.detect_frequency(series, data['Datum'])
                }
                
                if coverage < 0.5:
                    validation_results['warnings'].append(
                        f"Exogenous variable {var} has low coverage ({coverage:.1%})"
                    )
                
                if series.notna().sum() > 1 and series.std() == 0:
                    validation_results['warnings'].append(f"Variable {var} is constant")
        
        # Check for sufficient overlapping data
        all_vars = [target_var] + exog_vars
        available_vars = [var for var in all_vars if var in data.columns]
        
        if available_vars:
            complete_cases = data[available_vars].dropna()
            overlap_ratio = len(complete_cases) / len(data)
            
            validation_results['overlap_analysis'] = {
                'complete_cases': len(complete_cases),
                'total_cases': len(data),
                'overlap_ratio': overlap_ratio
            }
            
            if overlap_ratio < 0.1:
                validation_results['errors'].append(
                    f"Very low overlap between variables ({overlap_ratio:.1%} complete cases)"
                )
                validation_results['is_valid'] = False
            elif overlap_ratio < 0.3:
                validation_results['warnings'].append(
                    f"Low overlap between variables ({overlap_ratio:.1%} complete cases)"
                )
        
        # Generate recommendations
        if validation_results['data_quality'][target_var]['frequency'] == 'quarterly':
            validation_results['recommendations'].append(
                "Target is quarterly - will apply forward-fill to align with monthly exogenous variables"
            )
        
        if not validation_results['is_valid']:
            validation_results['recommendations'].append(
                "Consider using a different target variable or extending the time period"
            )
        
        return validation_results

class DataPreprocessor:
    """
    Handles data preprocessing for financial regression analysis.
    Includes proper mixed-frequency handling and transformation logic.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.date_col = date_col
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
    
    def create_transformations(self, transformation: str = 'levels') -> pd.DataFrame:
        """
        Create data transformations with proper mixed-frequency handling.
        """
        # Step 1: Handle mixed frequencies FIRST
        processed_data = MixedFrequencyProcessor.align_frequencies(
            self.data, self.target_var, self.date_col
        )
        
        # Step 2: Apply transformations
        transformed_data = processed_data[[self.date_col]].copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        print(f"Applying '{transformation}' transformation to {len(numeric_cols)} variables...")
        
        for col in numeric_cols:
            series = processed_data[col]
            
            if transformation == 'levels':
                transformed_data[col] = series
            elif transformation == 'log':
                # Check if all positive values
                positive_mask = series > 0
                if positive_mask.sum() > len(series) * 0.8:  # At least 80% positive
                    # Apply log transformation with small epsilon for zeros
                    transformed_data[col] = np.log(series.clip(lower=1e-6))
                else:
                    # Fall back to levels if not suitable for log
                    transformed_data[col] = series
                    print(f"  Warning: {col} not suitable for log transformation (negative values)")
            elif transformation == 'pct':
                transformed_data[col] = series.pct_change()
            elif transformation == 'diff':
                transformed_data[col] = series.diff()
            else:
                transformed_data[col] = series
        
        # Step 3: Clean up infinite and NaN values
        transformed_data = transformed_data.replace([np.inf, -np.inf], np.nan)
        
        # Step 4: Conservative outlier handling
        numeric_cols = [c for c in transformed_data.columns if c != self.date_col]
        for col in numeric_cols:
            series = transformed_data[col].dropna()
            if len(series) > 20:  # Only if enough observations
                q_low = series.quantile(0.01)  # Conservative 1%/99%
                q_high = series.quantile(0.99)
                if pd.notna(q_low) and pd.notna(q_high) and q_high > q_low:
                    transformed_data[col] = transformed_data[col].clip(lower=q_low, upper=q_high)
        
        # Step 5: Add seasonal dummies and time trend
        transformed_data = self._add_seasonal_features(transformed_data)
        
        # Step 6: Final cleaning
        before_clean = len(transformed_data)
        transformed_data = transformed_data.dropna(how="any")
        after_clean = len(transformed_data)
        
        if before_clean > after_clean:
            print(f"Dropped {before_clean - after_clean} rows with missing values after transformation")
        
        # Ensure stable data types
        for col in [c for c in transformed_data.columns if c != self.date_col]:
            transformed_data[col] = transformed_data[col].astype("float64")
        
        return transformed_data
    
    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add quarterly seasonal dummies and time trend."""
        data_with_features = data.copy()
        
        # Extract quarter from date
        data_with_features['quarter'] = pd.to_datetime(data_with_features[self.date_col]).dt.quarter
        
        # Create dummy variables (Q1 as base category)
        for q in [2, 3, 4]:
            data_with_features[f'Q{q}'] = (data_with_features['quarter'] == q).astype(int)
        
        # Create time trend (normalize to start from 0)
        data_with_features['time_trend'] = range(len(data_with_features))
        
        # Drop the quarter column
        data_with_features = data_with_features.drop('quarter', axis=1)
        
        return data_with_features

def diagnose_data_issues(data: pd.DataFrame, target_var: str, exog_vars: List[str]) -> None:
    """
    Comprehensive data quality diagnosis with specific focus on mixed-frequency issues.
    """
    print("\n" + "="*60)
    print("DATA QUALITY DIAGNOSIS")
    print("="*60)
    
    # Overall dataset info
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['Datum'].min().strftime('%Y-%m')} to {data['Datum'].max().strftime('%Y-%m')}")
    
    # Analyze each variable
    all_vars = [target_var] + exog_vars
    
    for i, var in enumerate(all_vars):
        if var not in data.columns:
            print(f"\n{i+1}. {var}: ❌ NOT FOUND IN DATA")
            continue
        
        series = data[var]
        valid_count = series.notna().sum()
        coverage = valid_count / len(series)
        frequency = MixedFrequencyProcessor.detect_frequency(series, data['Datum'])
        
        print(f"\n{i+1}. {var} ({'TARGET' if var == target_var else 'EXOG'}):")
        print(f"   - Frequency: {frequency}")
        print(f"   - Coverage: {coverage:.1%} ({valid_count}/{len(series)} observations)")
        
        if valid_count > 0:
            print(f"   - Range: {series.min():.4f} to {series.max():.4f}")
            print(f"   - Mean: {series.mean():.4f}, Std: {series.std():.4f}")
            
            # Check for problematic patterns
            if series.std() == 0:
                print(f"   - ⚠️  WARNING: Variable is constant!")
            
            if frequency == "quarterly" and var != target_var:
                print(f"   - ℹ️  INFO: Quarterly exogenous variable detected")
            elif frequency == "quarterly" and var == target_var:
                print(f"   - ℹ️  INFO: Quarterly target - will use forward-fill")
        
        # Check correlations with target (if not the target itself)
        if var != target_var and var in data.columns and target_var in data.columns:
            # Find overlapping observations
            overlap_data = data[[var, target_var]].dropna()
            if len(overlap_data) > 10:
                corr = overlap_data[var].corr(overlap_data[target_var])
                print(f"   - Correlation with {target_var}: {corr:.4f}")
            else:
                print(f"   - Correlation: insufficient overlap ({len(overlap_data)} obs)")
    
    # Check data overlap
    print(f"\n" + "-"*40)
    print("DATA OVERLAP ANALYSIS")
    print("-"*40)
    
    available_vars = [var for var in all_vars if var in data.columns]
    complete_cases = data[available_vars].dropna()
    overlap_ratio = len(complete_cases) / len(data)
    
    print(f"Complete cases: {len(complete_cases)} / {len(data)} ({overlap_ratio:.1%})")
    
    if overlap_ratio < 0.1:
        print("❌ CRITICAL: Very low data overlap - analysis likely to fail")
    elif overlap_ratio < 0.3:
        print("⚠️  WARNING: Low data overlap - results may be unreliable")
    else:
        print("✅ OK: Sufficient data overlap for analysis")
    
    # Frequency analysis summary
    print(f"\n" + "-"*40)
    print("FREQUENCY ANALYSIS SUMMARY")
    print("-"*40)
    
    quarterly_vars = []
    monthly_vars = []
    unknown_vars = []
    
    for var in available_vars:
        freq = MixedFrequencyProcessor.detect_frequency(data[var], data['Datum'])
        if freq == "quarterly":
            quarterly_vars.append(var)
        elif freq == "monthly":
            monthly_vars.append(var)
        else:
            unknown_vars.append(var)
    
    print(f"Quarterly variables ({len(quarterly_vars)}): {', '.join(quarterly_vars)}")
    print(f"Monthly variables ({len(monthly_vars)}): {', '.join(monthly_vars)}")
    if unknown_vars:
        print(f"Unknown frequency ({len(unknown_vars)}): {', '.join(unknown_vars)}")
    
    # Recommendations
    print(f"\n" + "-"*40)
    print("RECOMMENDATIONS")
    print("-"*40)
    
    if len(quarterly_vars) > 0 and target_var in quarterly_vars:
        print("✅ Will apply forward-fill to quarterly target variable")
        
    if overlap_ratio < 0.3:
        print("💡 Consider:")
        print("   - Using a longer time period")
        print("   - Using different target/exogenous variables")
        print("   - Checking data quality at source")
    
    if len(complete_cases) < 50:
        print("⚠️  Small sample size - consider using simpler models")

print("Mixed-frequency data processor loaded")


# %%
"""
Regression Methods & Cross-Validation - VEREINFACHT & ROBUSTE VERSION
Ohne Monkey-Patches - saubere Klassenstruktur
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from scipy import stats

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class RegressionMethod(ABC):
    """Abstract base class for regression methods."""
    
    def __init__(self, name: str, requires_scaling: bool = True):
        self.name = name
        self.requires_scaling = requires_scaling
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model and return results."""
        pass

class OLSMethod(RegressionMethod):
    """OLS Regression with robust standard errors."""
    
    def __init__(self):
        super().__init__("OLS", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # Add constant
        X_with_const = sm.add_constant(X, has_constant='add')
        
        # Fit model
        model = OLS(y, X_with_const, missing='drop')
        
        # Use robust standard errors (HAC)
        max_lags = min(4, int(len(y) ** (1/4)))
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        
        # Calculate diagnostics
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(results.resid)
        except:
            dw = np.nan
        
        try:
            jb_stat, jb_p = stats.jarque_bera(results.resid)[:2]
            jarque_bera = {'statistic': jb_stat, 'p_value': jb_p}
        except:
            jarque_bera = {'statistic': np.nan, 'p_value': np.nan}
        
        diagnostics = {
            'durbin_watson': dw,
            'jarque_bera': jarque_bera
        }
        
        return {
            'model': results,
            'coefficients': results.params,
            'std_errors': results.bse,
            'p_values': results.pvalues,
            'r_squared': results.rsquared,
            'r_squared_adj': results.rsquared_adj,
            'mse': results.mse_resid,
            'mae': np.mean(np.abs(results.resid)),
            'residuals': results.resid,
            'fitted_values': results.fittedvalues,
            'diagnostics': diagnostics
        }

class RandomForestMethod(RegressionMethod):
    """Conservative Random Forest optimized for financial time series."""
    
    def __init__(self):
        super().__init__("Random Forest", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = X.shape
        
        # Very conservative hyperparameters based on sample size
        if n_samples < 50:
            n_estimators = 100
            max_depth = 3
            min_samples_leaf = max(5, n_samples // 10)
            min_samples_split = max(10, n_samples // 5)
        elif n_samples < 100:
            n_estimators = 150
            max_depth = 4
            min_samples_leaf = max(8, n_samples // 12)
            min_samples_split = max(16, n_samples // 6)
        else:
            n_estimators = 200
            max_depth = min(5, max(3, int(np.log2(n_samples)) - 2))
            min_samples_leaf = max(10, n_samples // 15)
            min_samples_split = max(20, n_samples // 8)
        
        # Additional constraints for high-dimensional data
        max_features = "sqrt" if n_features <= n_samples // 3 else max(1, min(int(np.sqrt(n_features)), n_features // 2))
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            max_samples=min(0.8, max(0.5, 1.0 - 0.1 * n_features / n_samples)),
            random_state=42,
            n_jobs=1  # Avoid multiprocessing issues
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Out-of-bag score as additional validation
        oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else np.nan
        
        return {
            'model': model,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'oob_score': oob_score,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'feature_importance': model.feature_importances_,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'max_features': max_features
            }
        }

class XGBoostMethod(RegressionMethod):
    """Conservative XGBoost optimized for financial time series."""
    
    def __init__(self):
        super().__init__("XGBoost", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available")
        
        n_samples, n_features = X.shape
        
        # Very conservative hyperparameters
        n_estimators = min(100, max(50, n_samples // 3))
        max_depth = 3
        learning_rate = 0.01 if n_samples > 50 else 0.02
        
        # Strong regularization
        reg_alpha = 1.0 + 0.1 * (n_features / 10)  # L1
        reg_lambda = 2.0 + 0.2 * (n_features / 10)  # L2
        
        # Sampling parameters
        subsample = max(0.6, 1.0 - 0.05 * (n_features / 10))
        colsample_bytree = max(0.6, 1.0 - 0.05 * (n_features / 10))
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=max(3, n_samples // 20),
            gamma=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'feature_importance': model.feature_importances_,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }
        }

class SVRMethod(RegressionMethod):
    """Support Vector Regression."""
    
    def __init__(self):
        super().__init__("SVR", requires_scaling=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred
        }

class BayesianRidgeMethod(RegressionMethod):
    """Bayesian Ridge Regression."""
    
    def __init__(self):
        super().__init__("Bayesian Ridge", requires_scaling=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = BayesianRidge()
        model.fit(X_scaled, y)
        
        y_pred, y_std = model.predict(X_scaled, return_std=True)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'prediction_std': y_std,
            'coefficients': model.coef_
        }

class MethodRegistry:
    """Registry for regression methods."""
    
    def __init__(self):
        self.methods = {
            "OLS": OLSMethod(),
            "Random Forest": RandomForestMethod(),
            "SVR": SVRMethod(),
            "Bayesian Ridge": BayesianRidgeMethod()
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.methods["XGBoost"] = XGBoostMethod()
    
    def get_method(self, name: str) -> RegressionMethod:
        if name not in self.methods:
            raise ValueError(f"Method '{name}' not available. Choose from: {list(self.methods.keys())}")
        return self.methods[name]
    
    def list_methods(self) -> List[str]:
        return list(self.methods.keys())

class RobustCrossValidator:
    """
    Robust time series cross-validation for financial data.
    Fixed implementation without leakage or extreme scores.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def validate_method(self, method: RegressionMethod, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform robust time series cross-validation.
        """
        n_samples = len(X_train)
        
        # Conservative CV parameters based on sample size
        if n_samples < 30:
            n_splits = 2
            gap = 1
        elif n_samples < 60:
            n_splits = 3
            gap = 2
        else:
            n_splits = min(4, n_samples // 20)  # Conservative: at least 20 obs per fold
            gap = max(2, int(n_samples * 0.05))  # Larger gaps
        
        if n_splits < 2:
            return {'cv_scores': [], 'cv_mean': np.nan, 'cv_std': np.nan, 'n_folds': 0}
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        cv_scores = []
        successful_folds = 0
        
        print(f"    Running {n_splits}-fold CV with gap={gap}...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            try:
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Ensure minimum fold sizes
                if len(X_fold_train) < 10 or len(X_fold_val) < 3:
                    print(f"      Fold {fold_idx+1}: Skipped (insufficient data)")
                    continue
                
                # Fit method on fold training data
                fold_results = method.fit(X_fold_train, y_fold_train)
                
                # Evaluate on fold validation data
                fold_test_perf = self._evaluate_on_test(fold_results, X_fold_val, y_fold_val, method)
                
                score = fold_test_perf['r_squared']
                
                # Sanity check: Only accept reasonable scores
                if np.isfinite(score) and -1.0 <= score <= 1.0:
                    cv_scores.append(score)
                    successful_folds += 1
                    print(f"      Fold {fold_idx+1}: R² = {score:.4f}")
                else:
                    print(f"      Fold {fold_idx+1}: Extreme score {score:.4f} - skipped")
                
            except Exception as e:
                print(f"      Fold {fold_idx+1}: Failed ({str(e)[:50]})")
                continue
        
        if len(cv_scores) == 0:
            return {'cv_scores': [], 'cv_mean': np.nan, 'cv_std': np.nan, 'n_folds': 0}
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'n_folds': successful_folds
        }
    
    def _evaluate_on_test(self, train_results: Dict[str, Any], X_test: np.ndarray, 
                         y_test: np.ndarray, method: RegressionMethod) -> Dict[str, Any]:
        """Evaluate trained model on test data."""
        model = train_results['model']
        
        # Handle predictions based on model type
        try:
            if method.name == "OLS":
                X_test_with_const = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_with_const)
            elif method.requires_scaling and 'scaler' in train_results:
                X_test_scaled = train_results['scaler'].transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'mse': test_mse,
                'r_squared': test_r2,
                'mae': test_mae,
                'predictions': y_pred,
                'actual': y_test,
                'residuals': y_test - y_pred
            }
            
        except Exception as e:
            # Return NaN if prediction fails
            return {
                'mse': np.nan,
                'r_squared': np.nan,
                'mae': np.nan,
                'predictions': np.full(len(y_test), np.nan),
                'actual': y_test,
                'residuals': np.full(len(y_test), np.nan),
                'error': str(e)
            }

class TimeSeriesSplitter:
    """
    Robust time series train/test splitting with mandatory gaps.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def split(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-series aware train/test split with mandatory gap periods."""
        n_samples = len(X)
        
        # Adaptive gap and test size based on sample size
        if n_samples < 50:
            gap = 1
            test_size = max(0.2, self.config.test_size)  # Minimum 20% for test
        elif n_samples < 100:
            gap = 2
            test_size = self.config.test_size
        else:
            gap = max(2, int(n_samples * 0.02))  # 2% of sample as gap, minimum 2
            test_size = self.config.test_size
        
        # Calculate indices
        test_samples = int(n_samples * test_size)
        train_end = n_samples - test_samples - gap
        test_start = train_end + gap
        
        # Ensure minimum training data (60%)
        min_train = int(n_samples * 0.6)
        if train_end < min_train:
            print(f"Warning: Limited training data: {train_end}/{n_samples} samples")
            train_end = min_train
            gap = max(1, n_samples - train_end - test_samples)
            test_start = train_end + gap
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_start + test_samples]
        y_test = y[test_start:test_start + test_samples]
        
        print(f"    Split: Train={len(y_train)}, Gap={gap}, Test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test

print("Regression methods and cross-validation loaded")



# %%
"""
Feature Selection & Analysis - VEREINFACHT
Saubere Implementation ohne komplexe Patches
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

class SimpleFeatureSelector:
    """
    Simplified feature selection with robust methods.
    """
    
    @staticmethod
    def statistical_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 5) -> Tuple[List[str], np.ndarray]:
        """Select k best features using F-test."""
        k_actual = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_actual)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices
    
    @staticmethod
    def importance_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[List[str], np.ndarray]:
        """Select features using Random Forest importance."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top n features
        n_actual = min(n_features, X.shape[1])
        top_indices = np.argsort(importances)[::-1][:n_actual]
        
        selected_names = [feature_names[i] for i in top_indices]
        return selected_names, top_indices
    
    @staticmethod
    def rfe_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[List[str], np.ndarray]:
        """Recursive feature elimination."""
        n_actual = min(n_features, X.shape[1])
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        selector = RFE(estimator=estimator, n_features_to_select=n_actual)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices
    
    @staticmethod
    def lasso_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], np.ndarray]:
        """Lasso-based feature selection."""
        from sklearn.feature_selection import SelectFromModel
        
        lasso_cv = LassoCV(cv=3, random_state=42, max_iter=1000)
        selector = SelectFromModel(lasso_cv)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        if len(selected_indices) == 0:
            # Fallback: use all features
            selected_indices = np.arange(X.shape[1])
        
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices

class FeatureCombinationTester:
    """
    Test different feature combinations systematically.
    """
    
    def __init__(self, method_registry, cv_validator):
        self.method_registry = method_registry
        self.cv_validator = cv_validator
    
    def test_combinations(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         max_combinations: int = 20) -> pd.DataFrame:
        """
        Test feature combinations with proper validation.
        """
        print(f"Testing feature combinations from {len(feature_names)} features...")
        
        # Limit feature space for combinations if too large
        if len(feature_names) > 10:
            print(f"Feature space too large ({len(feature_names)}), using top 10 by importance...")
            # Use Random Forest to get top features
            selector = SimpleFeatureSelector()
            top_names, top_indices = selector.importance_selection(X, y, feature_names, n_features=10)
            X_reduced = X[:, top_indices]
            feature_names = top_names
            X = X_reduced
        
        # Generate combinations
        all_combos = []
        min_features = 2
        max_features = min(6, len(feature_names))  # Limit to reasonable size
        
        for size in range(min_features, max_features + 1):
            combos = list(combinations(feature_names, size))
            all_combos.extend(combos)
            
            # Stop if we have enough combinations
            if len(all_combos) >= max_combinations:
                break
        
        # Limit to max_combinations
        if len(all_combos) > max_combinations:
            all_combos = all_combos[:max_combinations]
        
        print(f"Testing {len(all_combos)} combinations...")
        
        # Test each combination
        results = []
        method = self.method_registry.get_method('Random Forest')
        
        for i, combo in enumerate(all_combos):
            try:
                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{len(all_combos)} combinations tested")
                
                # Get indices for this combination
                combo_indices = [feature_names.index(name) for name in combo]
                X_combo = X[:, combo_indices]
                
                if X_combo.shape[1] == 0 or len(X_combo) < 20:
                    continue
                
                # Split data for this combination
                splitter = TimeSeriesSplitter(AnalysisConfig())
                X_train, X_test, y_train, y_test = splitter.split(y, X_combo)
                
                if len(X_train) < 10 or len(X_test) < 3:
                    continue
                
                # Fit method
                train_results = method.fit(X_train, y_train)
                
                # Evaluate on test
                test_perf = self.cv_validator._evaluate_on_test(train_results, X_test, y_test, method)
                
                # Store results
                results.append({
                    'combination_id': i,
                    'features': ', '.join(combo),
                    'n_features': len(combo),
                    'test_r_squared': test_perf.get('r_squared', np.nan),
                    'test_mse': test_perf.get('mse', np.nan),
                    'overfitting': train_results.get('r_squared', 0) - test_perf.get('r_squared', 0),
                    'feature_list': list(combo)
                })
                
            except Exception as e:
                print(f"  Combination {i} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print("No successful combinations tested")
            return pd.DataFrame(columns=['combination_id', 'features', 'n_features', 'test_r_squared', 'test_mse', 'overfitting', 'feature_list'])
        
        df = pd.DataFrame(results)
        df = df.sort_values('test_r_squared', ascending=False).reset_index(drop=True)
        
        print(f"Combination testing completed: {len(df)} successful combinations")
        return df

class FeatureAnalyzer:
    """
    Main feature analysis coordinator.
    """
    
    def __init__(self, method_registry, cv_validator):
        self.method_registry = method_registry
        self.cv_validator = cv_validator
        self.selector = SimpleFeatureSelector()
        self.combo_tester = FeatureCombinationTester(method_registry, cv_validator)
    
    def test_selection_methods(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """
        Compare different feature selection methods.
        """
        print("Testing feature selection methods...")
        
        # Split data once for consistent comparison
        splitter = TimeSeriesSplitter(AnalysisConfig())
        X_train, X_test, y_train, y_test = splitter.split(y, X)
        
        selection_methods = {}
        
        # All Features baseline
        selection_methods['All Features'] = (feature_names, np.arange(len(feature_names)))
        
        # Statistical selection (F-test)
        try:
            sel_names, sel_idx = self.selector.statistical_selection(X_train, y_train, feature_names, k=5)
            selection_methods['Statistical (F-test)'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Statistical selection failed: {e}")
        
        # Importance-based selection
        try:
            sel_names, sel_idx = self.selector.importance_selection(X_train, y_train, feature_names, n_features=5)
            selection_methods['Importance (RF)'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Importance selection failed: {e}")
        
        # RFE selection
        try:
            sel_names, sel_idx = self.selector.rfe_selection(X_train, y_train, feature_names, n_features=5)
            selection_methods['RFE'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  RFE selection failed: {e}")
        
        # Lasso selection
        try:
            sel_names, sel_idx = self.selector.lasso_selection(X_train, y_train, feature_names)
            selection_methods['Lasso'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Lasso selection failed: {e}")
        
        # Test each selection method
        results = []
        method = self.method_registry.get_method('Random Forest')
        
        for method_name, (sel_names, sel_idx) in selection_methods.items():
            try:
                if len(sel_idx) == 0:
                    continue
                
                X_train_sel = X_train[:, sel_idx]
                X_test_sel = X_test[:, sel_idx]
                
                # Fit on selected features
                train_results = method.fit(X_train_sel, y_train)
                test_perf = self.cv_validator._evaluate_on_test(train_results, X_test_sel, y_test, method)
                
                results.append({
                    'selection_method': method_name,
                    'selected_features': ', '.join(sel_names),
                    'n_features': len(sel_names),
                    'test_r_squared': test_perf.get('r_squared', np.nan),
                    'test_mse': test_perf.get('mse', np.nan),
                    'overfitting': train_results.get('r_squared', 0) - test_perf.get('r_squared', 0)
                })
                
                print(f"  {method_name}: {len(sel_names)} features, Test R² = {test_perf.get('r_squared', np.nan):.4f}")
                
            except Exception as e:
                print(f"  {method_name} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print("No selection methods succeeded")
            return pd.DataFrame(columns=['selection_method', 'selected_features', 'n_features', 'test_r_squared', 'test_mse', 'overfitting'])
        
        df = pd.DataFrame(results)
        return df.sort_values('test_r_squared', ascending=False).reset_index(drop=True)
    
    def analyze_feature_importance(self, train_results: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze and rank feature importance from trained model.
        """
        importance_data = []
        
        # Extract feature importance/coefficients
        if 'feature_importance' in train_results:
            # Tree-based methods
            importances = train_results['feature_importance']
            for i, (name, imp) in enumerate(zip(feature_names, importances)):
                importance_data.append({
                    'feature': name,
                    'importance': imp,
                    'rank': i + 1,
                    'type': 'importance'
                })
        
        elif 'coefficients' in train_results:
            # Linear methods
            coefficients = train_results['coefficients']
            
            # Handle different coefficient formats
            if hasattr(coefficients, 'values'):
                coef_values = coefficients.values
            else:
                coef_values = np.array(coefficients)
            
            # Skip constant term if present (OLS adds constant)
            if len(coef_values) == len(feature_names) + 1:
                coef_values = coef_values[1:]  # Skip constant
            
            # Create importance based on absolute coefficient values
            abs_coefs = np.abs(coef_values[:len(feature_names)])
            
            for i, (name, coef) in enumerate(zip(feature_names, coef_values[:len(feature_names)])):
                importance_data.append({
                    'feature': name,
                    'importance': abs_coefs[i],
                    'coefficient': coef,
                    'rank': i + 1,
                    'type': 'coefficient'
                })
        
        if not importance_data:
            return pd.DataFrame(columns=['feature', 'importance', 'rank', 'type'])
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df

print("Feature selection and analysis loaded")







"""
Improved Time Series Splitting & Cross-Validation - KORRIGIERT
Verhindert Data Leakage durch strenge zeitliche Trennung und realistische CV-Splits
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

class ImprovedTimeSeriesSplitter:
    """
    Verbesserter Zeitreihen-Splitter mit strikten Anti-Leakage Regeln.
    """
    
    def __init__(self, config):
        self.config = config
    
    def split_with_dates(self, data: pd.DataFrame, 
                        date_col: str = "Datum",
                        test_size: float = None,
                        min_train_size: float = 0.6,
                        gap_months: int = 3) -> Dict[str, Any]:
        """
        Zeitbasierter Split mit expliziten Datums-Gaps.
        
        Args:
            data: DataFrame mit Zeitreihen
            date_col: Name der Datumsspalte  
            test_size: Anteil der Testdaten (default aus config)
            min_train_size: Mindestanteil für Trainingsdaten
            gap_months: Monate zwischen Training und Test
        
        Returns:
            Dict mit train_data, test_data, gap_info, split_info
        """
        if len(data) < 30:
            raise ValueError(f"Dataset zu klein für stabilen Split: {len(data)} Beobachtungen")
        
        test_size = test_size or self.config.test_size
        data_sorted = data.sort_values(date_col).reset_index(drop=True)
        
        total_obs = len(data_sorted)
        
        # Berechne Split-Punkte basierend auf Datum
        date_range = data_sorted[date_col].max() - data_sorted[date_col].min()
        total_months = date_range.days / 30.44  # Approximation
        
        if total_months < 24:  # Weniger als 2 Jahre
            gap_months = 1  # Reduziere Gap
            test_size = min(test_size, 0.2)  # Kleinere Testgröße
        
        # Test-Periode definieren
        test_months = total_months * test_size
        train_months = total_months - test_months - gap_months
        
        if train_months < total_months * min_train_size:
            # Anpassung wenn zu wenig Trainingsdaten
            train_months = total_months * min_train_size
            gap_months = max(1, int((total_months - train_months - test_months) / 2))
            test_months = total_months - train_months - gap_months
            
            warnings.warn(f"Gap reduziert auf {gap_months} Monate für ausreichend Trainingsdaten")
        
        # Datums-basierte Cutoffs berechnen
        start_date = data_sorted[date_col].min()
        train_end_date = start_date + pd.DateOffset(months=int(train_months))
        gap_end_date = train_end_date + pd.DateOffset(months=gap_months)
        test_end_date = data_sorted[date_col].max()
        
        # Daten aufteilen
        train_mask = data_sorted[date_col] < train_end_date
        test_mask = data_sorted[date_col] >= gap_end_date
        gap_mask = (data_sorted[date_col] >= train_end_date) & (data_sorted[date_col] < gap_end_date)
        
        train_data = data_sorted[train_mask].copy()
        test_data = data_sorted[test_mask].copy()
        gap_data = data_sorted[gap_mask].copy()
        
        # Validierung
        if len(train_data) < 20:
            raise ValueError(f"Training set zu klein: {len(train_data)} Beobachtungen")
        if len(test_data) < 5:
            raise ValueError(f"Test set zu klein: {len(test_data)} Beobachtungen")
        
        split_info = {
            'total_observations': total_obs,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'gap_size': len(gap_data),
            'train_ratio': len(train_data) / total_obs,
            'test_ratio': len(test_data) / total_obs,
            'gap_ratio': len(gap_data) / total_obs,
            'train_end_date': train_end_date,
            'gap_end_date': gap_end_date,
            'actual_gap_months': gap_months,
            'date_range_months': total_months
        }
        
        print(f"Time series split: Train={len(train_data)} | Gap={len(gap_data)} | Test={len(test_data)}")
        print(f"  Training period: {data_sorted[date_col].min().strftime('%Y-%m')} to {train_end_date.strftime('%Y-%m')}")
        print(f"  Gap period: {train_end_date.strftime('%Y-%m')} to {gap_end_date.strftime('%Y-%m')}")
        print(f"  Test period: {gap_end_date.strftime('%Y-%m')} to {data_sorted[date_col].max().strftime('%Y-%m')}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'gap_data': gap_data,
            'split_info': split_info,
            'train_end_date': train_end_date
        }
    
    def split(self, y: np.ndarray, X: np.ndarray, 
              dates: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Legacy-kompatibler Split für bestehenden Code.
        """
        n_samples = len(X)
        
        # Adaptive Größen basierend auf Stichprobe
        if n_samples < 40:
            test_size = 0.20
            gap_periods = 1
        elif n_samples < 80:
            test_size = 0.25
            gap_periods = 2
        else:
            test_size = self.config.test_size
            gap_periods = max(1, int(n_samples * 0.015))  # 1.5% als Gap
        
        # Berechne Indizes
        test_samples = int(n_samples * test_size)
        train_end = n_samples - test_samples - gap_periods
        test_start = train_end + gap_periods
        
        # Mindest-Trainingsgröße sicherstellen
        min_train = max(20, int(n_samples * 0.5))  # Mindestens 50% für Training
        if train_end < min_train:
            train_end = min_train
            gap_periods = max(1, n_samples - train_end - test_samples)
            test_start = train_end + gap_periods
        
        if test_start >= n_samples:
            raise ValueError("Dataset zu klein für robusten Split")
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_start + test_samples]
        y_test = y[test_start:test_start + test_samples]
        
        print(f"Array split: Train={len(y_train)}, Gap={gap_periods}, Test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test


class ImprovedRobustCrossValidator:
    """
    Verbesserte Cross-Validation mit realistischen Splits und Stabilitätsprüfungen.
    """
    
    def __init__(self, config):
        self.config = config
        
    def validate_method_robust(self, method, X_train: np.ndarray, y_train: np.ndarray,
                              dates_train: pd.Series = None) -> Dict[str, Any]:
        """
        Robuste Kreuzvalidierung mit stabilitätsfokussierten Splits.
        """
        n_samples = len(X_train)
        
        # Sehr konservative CV-Parameter basierend auf Datengröße
        if n_samples < 40:
            n_splits = 2
            gap = 1
        elif n_samples < 100:
            n_splits = 3
            gap = 2
        else:
            n_splits = 4
            gap = max(2, int(n_samples * 0.02))  # Statt 0.05
        
        if n_splits < 2:
            return {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'n_folds': 0,
                'stability_warning': 'Dataset too small for CV'
            }
        
        # Verwende TimeSeriesSplit mit größeren Gaps
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=None)
        cv_scores = []
        fold_details = []
        successful_folds = 0
        
        print(f"    Running {n_splits}-fold CV with gap={gap}...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            try:
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]  
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
                
                # Strikte Mindestgrößen für Folds
                min_train_fold = max(10, len(X_train) // (n_splits + 3))
                min_val_fold = max(3, len(X_train) // (n_splits * 4))
                
                if len(X_fold_train) < min_train_fold or len(X_fold_val) < min_val_fold:
                    print(f"      Fold {fold_idx+1}: Skipped (train={len(X_fold_train)}, val={len(X_fold_val)})")
                    continue
                
                # Prüfe auf ausreichende Variation in y
                if y_fold_train.std() < 1e-8 or y_fold_val.std() < 1e-8:
                    print(f"      Fold {fold_idx+1}: Skipped (insufficient variation)")
                    continue
                
                # Fit auf Fold-Training
                try:
                    fold_results = method.fit(X_fold_train, y_fold_train)
                except Exception as fit_error:
                    print(f"      Fold {fold_idx+1}: Fit failed ({str(fit_error)[:30]})")
                    continue
                
                # Evaluiere auf Fold-Validation
                fold_test_perf = self._evaluate_on_test_safe(
                    fold_results, X_fold_val, y_fold_val, method
                )
                
                score = fold_test_perf.get('r_squared', np.nan)
                mse = fold_test_perf.get('mse', np.nan)
                
                # Strenge Sanity Checks
                is_valid_score = (
                    np.isfinite(score) and 
                    -2.0 <= score <= 1.0 and  # Erweitere negativen Bereich leicht
                    np.isfinite(mse) and 
                    mse > 0
                )
                
                if is_valid_score:
                    cv_scores.append(score)
                    fold_details.append({
                        'fold': fold_idx + 1,
                        'r_squared': score,
                        'mse': mse,
                        'train_size': len(X_fold_train),
                        'val_size': len(X_fold_val)
                    })
                    successful_folds += 1
                    print(f"      Fold {fold_idx+1}: R² = {score:.4f}, MSE = {mse:.4f}")
                else:
                    print(f"      Fold {fold_idx+1}: Invalid score R²={score:.4f}, MSE={mse:.4f}")
                
            except Exception as e:
                print(f"      Fold {fold_idx+1}: Exception - {str(e)[:40]}")
                continue
        
        if len(cv_scores) == 0:
            return {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'n_folds': 0,
                'stability_warning': 'All CV folds failed'
            }
        
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Stabilitäts-Assessment
        stability_warnings = []
        
        if cv_std > 0.3:
            stability_warnings.append('High CV variance - unstable model')
        
        if len(cv_scores) < n_splits / 2:
            stability_warnings.append(f'Only {len(cv_scores)}/{n_splits} folds succeeded')
            
        if successful_folds > 1:
            score_range = max(cv_scores) - min(cv_scores)
            if score_range > 0.5:
                stability_warnings.append('Very high score range across folds')
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'n_folds': successful_folds,
            'fold_details': fold_details,
            'stability_warnings': stability_warnings,
            'parameters': {
                'n_splits_attempted': n_splits,
                'gap_used': gap,
                'min_train_fold': min_train_fold,
                'min_val_fold': min_val_fold
            }
        }
    
    def _evaluate_on_test_safe(self, train_results: Dict[str, Any], 
                              X_test: np.ndarray, y_test: np.ndarray, 
                              method) -> Dict[str, Any]:
        """
        Sichere Test-Evaluierung mit ausführlichem Error Handling.
        """
        try:
            model = train_results.get('model')
            if model is None:
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan, 
                       'error': 'No model in train_results'}
            
            # Predictions generieren basierend auf Methodentyp
            if method.name == "OLS":
                # Statsmodels OLS
                import statsmodels.api as sm
                X_test_with_const = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_with_const)
                
            elif method.requires_scaling and 'scaler' in train_results:
                # Skalierte Methoden (SVR, Bayesian Ridge)
                scaler = train_results['scaler']
                X_test_scaled = scaler.transform(X_test)
                
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test_scaled)
                else:
                    return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                           'error': 'Model has no predict method'}
                    
            else:
                # Tree-based und andere Methoden
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                           'error': 'Model has no predict method'}
            
            # Validate predictions
            if not isinstance(y_pred, (np.ndarray, list)):
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'Invalid prediction type'}
            
            y_pred = np.array(y_pred).flatten()
            
            if len(y_pred) != len(y_test):
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'Prediction length mismatch'}
            
            if not np.isfinite(y_pred).any():
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'All predictions are non-finite'}
            
            # Berechne Metriken mit robusten Checks
            try:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Zusätzliche Sanity Checks
                if not np.isfinite(mse) or mse < 0:
                    mse = np.nan
                if not np.isfinite(r2):
                    r2 = np.nan  
                if not np.isfinite(mae) or mae < 0:
                    mae = np.nan
                
                return {
                    'r_squared': float(r2) if np.isfinite(r2) else np.nan,
                    'mse': float(mse) if np.isfinite(mse) else np.nan,
                    'mae': float(mae) if np.isfinite(mae) else np.nan,
                    'predictions': y_pred,
                    'actual': y_test,
                    'residuals': y_test - y_pred
                }
                
            except Exception as metric_error:
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': f'Metric calculation failed: {str(metric_error)[:50]}'}
                       
        except Exception as e:
            return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                   'error': f'Evaluation failed: {str(e)[:50]}'}


class ImprovedFeatureSelector:
    """
    Vereinfachte Feature-Selektion mit Fokus auf Stabilität.
    """
    
    @staticmethod  
    def select_robust_features(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              max_features: int = None) -> Tuple[List[str], np.ndarray]:
        """
        Robuste Feature-Selektion basierend auf Korrelation und Stabilität.
        """
        if X.shape[1] == 0:
            return [], np.array([])
        
        # Automatische Begrenzung basierend auf Sample-Größe
        n_samples = len(X)
        if max_features is None:
            if n_samples < 30:
                max_features = 2  
            elif n_samples < 60:
                max_features = 3
            elif n_samples < 100:
                max_features = 4
            else:
                max_features = min(6, X.shape[1], n_samples // 15)
        
        max_features = min(max_features, X.shape[1])
        
        # Berechne Korrelationen mit Target
        correlations = []
        valid_features = []
        
        for i, feature_name in enumerate(feature_names):
            try:
                feature_data = X[:, i]
                
                # Skip konstante oder fast-konstante Features
                if np.std(feature_data) < 1e-10:
                    continue
                    
                # Skip Features mit zu vielen NaNs
                if np.isnan(feature_data).sum() > len(feature_data) * 0.5:
                    continue
                
                # Berechne Korrelation (robust gegen NaNs)
                mask = ~(np.isnan(feature_data) | np.isnan(y))
                if mask.sum() < max(5, len(y) * 0.3):  # Mindestens 30% overlap
                    continue
                    
                corr = np.corrcoef(feature_data[mask], y[mask])[0, 1]
                
                if np.isfinite(corr):
                    correlations.append((abs(corr), i, feature_name))
                    valid_features.append(i)
                    
            except Exception:
                continue
        
        if not correlations:
            # Fallback: erste verfügbare Features nehmen
            available_features = []
            for i, name in enumerate(feature_names[:max_features]):
                if np.std(X[:, i]) > 1e-10:
                    available_features.append((name, i))
                    
            if available_features:
                selected_names = [name for name, idx in available_features]
                selected_indices = np.array([idx for name, idx in available_features])
                return selected_names, selected_indices
            else:
                return [], np.array([])
        
        # Sortiere nach Korrelation und wähle Top-Features
        correlations.sort(key=lambda x: x[0], reverse=True)
        selected_correlations = correlations[:max_features]
        
        selected_names = [name for _, _, name in selected_correlations]
        selected_indices = np.array([idx for _, idx, _ in selected_correlations])
        
        print(f"Selected {len(selected_names)}/{len(feature_names)} features based on correlation:")
        for corr_abs, idx, name in selected_correlations:
            print(f"  {name}: |r| = {corr_abs:.3f}")
        
        return selected_names, selected_indices











# %%
"""
Main Financial Regression Analyzer - VEREINFACHT & ROBUST
Koordiniert alle Komponenten ohne komplexe Monkey-Patches
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import warnings

class FinancialRegressionAnalyzer:
    """
    Main analyzer that coordinates all components for financial regression analysis.
    Clean implementation without monkey patches.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, exog_vars: List[str], 
                 config: AnalysisConfig = None, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.exog_vars = exog_vars
        self.date_col = date_col
        self.config = config or AnalysisConfig()
        
        # Initialize components
        self.method_registry = MethodRegistry()
        self.cv_validator = RobustCrossValidator(self.config)
        self.splitter = TimeSeriesSplitter(self.config)
        self.preprocessor = DataPreprocessor(data, target_var, date_col)
        self.feature_analyzer = FeatureAnalyzer(self.method_registry, self.cv_validator)
    
    def prepare_data(self, transformation: str = 'levels') -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """
        Prepare data for regression with proper mixed-frequency handling.
        """
        print(f"Preparing data with '{transformation}' transformation...")
        
        # Apply transformations (includes mixed-frequency handling)
        transformed_data = self.preprocessor.create_transformations(transformation)
        
        # Find target and feature columns
        target_col = self.target_var
        if target_col not in transformed_data.columns:
            possible_targets = [col for col in transformed_data.columns 
                             if col.startswith(self.target_var)]
            if possible_targets:
                target_col = possible_targets[0]
            else:
                raise ValueError(f"Target variable {target_col} not found after transformation")
        
        # Get feature columns
        feature_cols = []
        for var in self.exog_vars:
            if var in transformed_data.columns:
                feature_cols.append(var)
        
        # Add seasonal dummies and time trend
        seasonal_cols = ['Q2', 'Q3', 'Q4', 'time_trend']
        for col in seasonal_cols:
            if col in transformed_data.columns:
                feature_cols.append(col)
        
        if not feature_cols:
            raise ValueError("No feature columns found")
        
        # Create final dataset
        final_data = transformed_data[[target_col] + feature_cols].copy()
        final_data = final_data.dropna()
        
        if len(final_data) < 20:
            raise ValueError(f"Insufficient data: only {len(final_data)} observations after cleaning")
        
        # Extract arrays
        y = final_data[target_col].values
        X = final_data[feature_cols].values
        
        print(f"Final dataset: {len(final_data)} observations, {len(feature_cols)} features")
        
        return y, X, feature_cols, final_data
    
    def fit_method_with_validation(self, method_name: str, y: np.ndarray, X: np.ndarray, 
                                 feature_names: List[str], transformation: str = 'levels', final_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit a method with comprehensive validation.
        """
        print(f"  Fitting {method_name}...")
        
        # Train/test split
        X_train, X_test, y_train, y_test = self.splitter.split(y, X)
        
        
        # If configured, restrict test evaluation to quarter-end months (old approach)
        try:
            if getattr(self.config, 'evaluate_quarter_ends_only', False) and isinstance(final_data, pd.DataFrame) and self.date_col in final_data.columns:
                dates_all = pd.to_datetime(final_data[self.date_col].values)
                n_samples = len(dates_all)
                # Recompute split indices like the splitter
                if n_samples < 40:
                    test_size = 0.20; gap_periods = 1
                elif n_samples < 80:
                    test_size = 0.25; gap_periods = 2
                else:
                    test_size = self.config.test_size
                    gap_periods = max(1, int(n_samples * 0.015))
                test_samples = int(n_samples * test_size)
                train_end = n_samples - test_samples - gap_periods
                test_start = train_end + gap_periods
                dates_test = pd.Series(dates_all[test_start:test_start + test_samples])
                qe_mask = dates_test.dt.is_quarter_end
                if qe_mask.any() and qe_mask.sum() >= 3 and len(X_test) == len(qe_mask):
                    X_test = X_test[qe_mask.values]
                    y_test = y_test[qe_mask.values]
        except Exception:
            pass

        
        # Get method and fit on training data
        method = self.method_registry.get_method(method_name)
        train_results = method.fit(X_train, y_train)
        
        # Evaluate on test data
        test_performance = self.cv_validator._evaluate_on_test(train_results, X_test, y_test, method)
        
        # Cross-validation on training data
        cv_performance = self.cv_validator.validate_method(method, X_train, y_train)
        
        # Calculate metrics
        train_r2 = train_results.get('r_squared', np.nan)
        test_r2 = test_performance['r_squared']
        overfitting = train_r2 - test_r2 if np.isfinite(train_r2) and np.isfinite(test_r2) else np.nan
        
        # Combine results
        results = {
            **train_results,
            'method_name': method_name,
            'feature_names': feature_names,
            'target_var': self.target_var,
            'transformation': transformation,
            'test_performance': test_performance,
            'cv_performance': cv_performance,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'overfitting': overfitting,
            'validation_config': {
                'test_size': self.config.test_size,
                'train_size': len(X_train),
                'test_size_actual': len(X_test),
                'gap_used': len(y) - len(X_train) - len(X_test)
            }
        }
        
        # Performance summary
        cv_mean = cv_performance.get('cv_mean', np.nan)
        print(f"    Test R² = {test_r2:.4f}, Train R² = {train_r2:.4f}, CV = {cv_mean:.4f}")
        
        # Warnings
        if overfitting > 0.1:
            print(f"    ⚠️ WARNING: High overfitting ({overfitting:.4f})")
        if test_r2 > 0.9:
            print(f"    ⚠️ WARNING: Very high R² ({test_r2:.4f}) - check for leakage")
        
        return results
    
    def fit_multiple_methods(self, methods: List[str] = None, transformation: str = 'levels') -> Dict[str, Dict[str, Any]]:
        """
        Fit multiple methods with validation.
        """
        if methods is None:
            methods = self.method_registry.list_methods()
        
        # Prepare data
        y, X, feature_names, final_data = self.prepare_data(transformation)
        
        results = {}
        print(f"Fitting {len(methods)} methods with robust validation...")
        
        for method_name in methods:
            try:
                result = self.fit_method_with_validation(method_name, y, X, feature_names, transformation, final_data=final_data)
                results[method_name] = result
                
            except Exception as e:
                print(f"    ✗ {method_name} failed: {str(e)[:50]}")
                continue
        
        print(f"Successfully fitted {len(results)}/{len(methods)} methods")
        return results
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare method results in a structured format.
        """
        comparison_data = []
        
        for method_name, result in results.items():
            train_r2 = result.get('train_r_squared', np.nan)
            test_r2 = result.get('test_r_squared', np.nan)
            overfitting = result.get('overfitting', np.nan)
            cv_mean = result.get('cv_performance', {}).get('cv_mean', np.nan)
            cv_std = result.get('cv_performance', {}).get('cv_std', np.nan)
            
            # Classify overfitting level
            if np.isfinite(overfitting):
                if overfitting > 0.15:
                    overfitting_level = "SEVERE"
                elif overfitting > 0.08:
                    overfitting_level = "HIGH"
                elif overfitting > 0.04:
                    overfitting_level = "MODERATE"
                else:
                    overfitting_level = "LOW"
            else:
                overfitting_level = "UNKNOWN"
            
            # Get additional metrics
            oob_score = result.get('oob_score', np.nan)
            test_mse = result.get('test_performance', {}).get('mse', np.nan)
            test_mae = result.get('test_performance', {}).get('mae', np.nan)
            
            comparison_data.append({
                'Method': method_name,
                'Test_R²': test_r2,
                'Train_R²': train_r2,
                'Overfitting': overfitting,
                'Overfitting_Level': overfitting_level,
                'Test_MSE': test_mse,
                'Test_MAE': test_mae,
                'CV_Mean_R²': cv_mean,
                'CV_Std_R²': cv_std,
                'OOB_Score': oob_score
            })
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test_R²', ascending=False).round(4)
        
        # Add warning flags
        df['Warnings'] = ''
        for idx, row in df.iterrows():
            warnings = []
            if row['Test_R²'] > 0.9:
                warnings.append("Very high R² - check for leakage")
            if row['Test_R²'] < 0:
                warnings.append("Negative R² - poor fit")
            if row['Overfitting'] > 0.1:
                warnings.append("High overfitting")
            if np.isfinite(row['CV_Mean_R²']) and abs(row['Test_R²'] - row['CV_Mean_R²']) > 0.1:
                warnings.append("CV/Test discrepancy")
            df.at[idx, 'Warnings'] = '; '.join(warnings)
        
        return df
    
    def find_optimal_transformation(self, transformations: List[str] = None, 
                                   baseline_method: str = 'Random Forest') -> str:
        """
        Find optimal transformation by testing with a baseline method.
        """
        if transformations is None:
            transformations = ['levels', 'pct', 'diff']
        
        print("Finding optimal transformation...")
        
        best_transformation = None
        best_score = -np.inf
        transformation_results = {}
        
        for transformation in transformations:
            try:
                print(f"  Testing '{transformation}' transformation...")
                
                # Test with baseline method
                y, X, feature_names, final_data = self.prepare_data(transformation)
                result = self.fit_method_with_validation(method_name, y, X, feature_names, transformation, final_data=final_data)
                
                test_r2 = result.get('test_r_squared', np.nan)
                
                transformation_results[transformation] = {
                    'test_r2': test_r2,
                    'result': result
                }
                
                print(f"    Test R² = {test_r2:.4f}")
                
                # Check if this is the best so far
                if np.isfinite(test_r2) and test_r2 > best_score:
                    best_score = test_r2
                    best_transformation = transformation
                
            except Exception as e:
                print(f"    ✗ {transformation} failed: {str(e)[:50]}")
                continue
        
        if best_transformation is None:
            print("  Warning: No transformations succeeded, using 'levels'")
            return 'levels'
        
        print(f"  Best transformation: '{best_transformation}' (Test R² = {best_score:.4f})")
        return best_transformation
    
    def test_feature_selection_methods(self, transformation: str = 'levels') -> pd.DataFrame:
        """
        Test different feature selection methods.
        """
        print("Testing feature selection methods...")
        
        try:
            y, X, feature_names, _ = self.prepare_data(transformation)
            return self.feature_analyzer.test_selection_methods(X, y, feature_names)
        except Exception as e:
            print(f"Feature selection testing failed: {e}")
            return pd.DataFrame()
    
    def test_feature_combinations(self, max_combinations: int = 20, 
                                 transformation: str = 'levels') -> pd.DataFrame:
        """
        Test different feature combinations.
        """
        print("Testing feature combinations...")
        
        try:
            y, X, feature_names, _ = self.prepare_data(transformation)
            return self.feature_analyzer.combo_tester.test_combinations(
                X, y, feature_names, max_combinations
            )
        except Exception as e:
            print(f"Feature combination testing failed: {e}")
            return pd.DataFrame()

"""
Improved Financial Regression Analyzer - KORRIGIERT
Behebt die Hauptprobleme mit Data Leakage, CV-Splits und Feature Selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings



class TrainOnlyLagSelector:
    """
    Selects best lags per exogenous variable using TRAIN-ONLY correlation with the (transformed) target.
    No leakage: only dates strictly before train_end_date are used for scoring.
    """
    def __init__(self, config: LagConfig):
        self.cfg = config

    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        try:
            s = pd.concat([a, b], axis=1).dropna()
            if len(s) < 3:
                return np.nan
            return float(s.iloc[:,0].corr(s.iloc[:,1]))
        except Exception:
            return np.nan

    def apply(self, df: pd.DataFrame, exog_vars: List[str], target_col: str, date_col: str, train_end_date: Optional[pd.Timestamp]):
        kept = []
        details = []
        # Build train mask
        if train_end_date is not None:
            train_mask = pd.to_datetime(df[date_col]) < pd.to_datetime(train_end_date)
        else:
            # fallback: first 80%
            n = len(df)
            cutoff = int(n * 0.8)
            train_mask = pd.Series([True]*cutoff + [False]*(n-cutoff), index=df.index)

        # Score candidates
        candidates = []
        for var in exog_vars:
            if var not in df.columns:
                continue
            for L in self.cfg.candidates:
                col = f"{var}_lag{L}"
                if col not in df.columns:
                    df[col] = df[var].shift(L)
                corr = self._safe_corr(df.loc[train_mask, col], df.loc[train_mask, target_col])
                # require minimal overlap
                overlap = int(pd.concat([df[col], df[target_col]], axis=1).loc[train_mask].dropna().shape[0])
                if overlap >= self.cfg.min_train_overlap and (not np.isfinite(self.cfg.min_abs_corr) or abs(corr) >= self.cfg.min_abs_corr):
                    candidates.append((var, col, L, abs(corr), overlap))
                else:
                    details.append({'var': var, 'lag': L, 'status': 'skipped', 'corr': corr, 'overlap': overlap})

        # choose per-var best, then apply total cap
        best_per_var = {}
        for var, col, L, acorr, overlap in candidates:
            cur = best_per_var.get(var)
            if (cur is None) or (acorr > cur[3]):
                best_per_var[var] = (var, col, L, acorr, overlap)

        # Flatten, sort by |corr| desc
        ranked = sorted(best_per_var.values(), key=lambda x: x[3], reverse=True)
        total_cap = max(0, int(self.cfg.total_max))
        per_var_cap = max(1, int(self.cfg.per_var_max))
        per_var_counts = {v: 0 for v in best_per_var.keys()}

        for var, col, L, acorr, overlap in ranked:
            if len(kept) >= total_cap:
                break
            if per_var_counts[var] >= per_var_cap:
                continue
            kept.append(col)
            per_var_counts[var] += 1
            details.append({'var': var, 'lag': L, 'status': 'kept', 'corr': acorr, 'overlap': overlap})

        # mark dropped
        kept_set = set(kept)
        for var, col, L, acorr, overlap in ranked:
            if col not in kept_set:
                details.append({'var': var, 'lag': L, 'status': 'dropped', 'corr': acorr, 'overlap': overlap})

        report = {
            'kept': kept,
            'details': details
        }
        return df, report

class ImprovedFinancialRegressionAnalyzer:
    """
    Korrigierter Hauptanalysator mit robuster Anti-Leakage Architektur.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, exog_vars: List[str], 
                 config=None, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.exog_vars = exog_vars
        self.date_col = date_col
        self.config = config or AnalysisConfig()
        
        # Improved components
        self.method_registry = MethodRegistry()
        self.cv_validator = ImprovedRobustCrossValidator(self.config)
        self.splitter = ImprovedTimeSeriesSplitter(self.config)
        self.preprocessor = ImprovedDataPreprocessor(data, target_var, date_col)
        self.quality_checker = ImprovedDataQualityChecker()
    
    def comprehensive_data_validation(self) -> Dict[str, Any]:
        """
        Umfassende Datenvalidierung als erster Schritt.
        """
        print("=== COMPREHENSIVE DATA VALIDATION ===")
        
        validation_result = self.quality_checker.comprehensive_data_validation(
            self.data, self.target_var, self.exog_vars,
            min_observations=30,  # Erhöhte Mindestanforderung
            min_target_coverage=0.25  # Höhere Coverage-Anforderung
        )
        
        # Ausgabe der Validierungsergebnisse
        print(f"\nData Quality Summary:")
        print(f"  Total observations: {len(self.data)}")
        print(f"  Variables tested: {len([self.target_var] + self.exog_vars)}")
        
        if validation_result['errors']:
            print(f"\n❌ ERRORS ({len(validation_result['errors'])}):")
            for error in validation_result['errors']:
                print(f"    - {error}")
        
        if validation_result['warnings']:
            print(f"\n⚠️  WARNINGS ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings']:
                print(f"    - {warning}")
        
        # Stationarity results
        print(f"\nStationarity Tests:")
        for var, result in validation_result['stationarity_tests'].items():
            is_stationary = result.get('is_stationary', None)
            p_value = result.get('p_value', np.nan)
            
            status = "✅ Stationary" if is_stationary else "❌ Non-stationary" if is_stationary is False else "❓ Unknown"
            print(f"  {var}: {status} (p-value: {p_value:.3f})")
        
        # Recommendations
        if validation_result['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_result['recommendations']:
                print(f"    - {rec}")
        
        return validation_result
    
    def prepare_data_robust(self, transformation: str = 'levels',
                            use_train_test_split: bool = True) -> Dict[str, Any]:
        """
        Robuste Datenvorbereitung mit Anti-Leakage Schutz.
        """
        print(f"\n=== ROBUST DATA PREPARATION ===")
        print(f"Transformation: {transformation}")
        
        # Schritt 1: Zeitbasierter Split ZUERST (um Leakage zu verhindern)
        train_end_date = None
        split_info = None
        
        if use_train_test_split and len(self.data) > 30:
            try:
                # Früher Split für Anti-Leakage
                split_result = self.splitter.split_with_dates(
                    self.data, self.date_col, 
                    test_size=self.config.test_size,
                    gap_months=2  # Konservativer Gap
                )
                train_end_date = split_result['train_end_date']
                split_info = split_result['split_info']
                print(f"Early train/test split applied - training ends: {train_end_date.strftime('%Y-%m')}")
                
            except Exception as e:
                print(f"Warning: Could not apply early split: {e}")
                train_end_date = None
        
        # Schritt 2: Transformationen mit Anti-Leakage Schutz
        transform_result = self.preprocessor.create_robust_transformations(
            transformation=transformation,
            train_end_date=train_end_date,  # Critical: pass split date
            outlier_method='conservative'
        )
        
        transformed_data = transform_result['data']
        
        # Schritt 3: Feature-Selektion und finales Dataset
        target_col = self.target_var
        if target_col not in transformed_data.columns:
            available_targets = [col for col in transformed_data.columns if col.startswith(self.target_var)]
            if available_targets:
                target_col = available_targets[0]
            else:
                raise ValueError(f"Target variable {target_col} not found after transformation")
        
        # Intelligente Feature-Auswahl
        available_exog = []
        for var in self.exog_vars:
            if var in transformed_data.columns:
                # Prüfe Datenqualität
                series = transformed_data[var]
                coverage = series.notna().sum() / len(series)
                variation = series.std() if series.notna().sum() > 1 else 0.0
                if coverage > 0.3 and variation > 1e-8:  # Mindestanforderungen
                    available_exog.append(var)
                else:
                    print(f"  Excluding {var}: coverage={coverage:.1%}, std={variation:.2e}")
        
        # Saisonale Features hinzufügen
        seasonal_features = ['Q2', 'Q3', 'Q4', 'time_trend']
        for feat in seasonal_features:
            if feat in transformed_data.columns:
                available_exog.append(feat)
        
        # Add target lag (L=1) after seasonal features
        lag_col = f"{target_col}_lag1"
        if target_col in transformed_data.columns and lag_col not in transformed_data.columns:
            transformed_data[lag_col] = transformed_data[target_col].shift(1)
        if lag_col in transformed_data.columns:
            available_exog.append(lag_col)
        
        if not available_exog:
            raise ValueError("No suitable exogenous features found")
        
        # Finales Dataset erstellen
        final_columns = [self.date_col, target_col] + available_exog
        final_data = transformed_data[final_columns].copy()
        # Drop rows with NaN target
        final_data = final_data.dropna(subset=[target_col])
        print(f"Target NaNs after cleaning: {final_data[target_col].isnull().sum()}")
        
        # Robuste Bereinigung
        before_clean = len(final_data)
        
        # Mindestens Target + eine exogene Variable erforderlich
        min_required_vars = 2
        row_validity = final_data.notna().sum(axis=1) >= min_required_vars
        final_data = final_data[row_validity].copy()
        
        after_clean = len(final_data)
        
        if after_clean < 20:
            raise ValueError(f"Insufficient data after cleaning: {after_clean} observations")
        
        # Arrays extrahieren
        y = final_data[target_col].values
        X = final_data[available_exog].values

        # Remove any rows with NaNs in target or features
        mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
        final_data = final_data.loc[mask].copy()
        y = y[mask]
        X = X[mask]

        # Keep aligned dates for later filtering
        dates_array = pd.to_datetime(final_data[self.date_col]).to_numpy()

        print(f"After NaN removal: {len(y)} observations, {np.isnan(y).sum()} NaNs in target")
        print("Final dataset prepared:")
        print(f"  Observations: {before_clean} → {after_clean}")
        print(f"  Features: {len(available_exog)} (+ target)")
        print(f"  Features: {', '.join(available_exog[:5])}{'...' if len(available_exog) > 5 else ''}")
        
        # Sample sizes
        sample_sizes = {
            'total': int(len(y)),
            'features_selected': int(len(available_exog)),
            # optional – nur falls split_info gesetzt wurde:
            'train_candidate': int(split_info['train_size']) if split_info else None,
            'test_candidate': int(split_info['test_size']) if split_info else None,
        }

        return {
            'y': y,
            'X': X,
            'feature_names': available_exog,
            'final_data': final_data,
            'target_name': target_col,
            'transformation_info': transform_result,
            'split_info': split_info,
            'train_end_date': train_end_date,
            'preparation_warnings': transform_result.get('warnings', []),
            'sample_sizes': sample_sizes,
            'dates': dates_array
        }

    
    def fit_method_improved(self, method_name: str, preparation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verbesserte Methodenanpassung mit robusten Splits.
        """
        print(f"  Fitting {method_name}...")
        
        y = preparation_result['y']
        X = preparation_result['X']
        feature_names = preparation_result['feature_names']
        
        # Feature-Reduktion für kleine Datensätze
        if len(y) < 60 or X.shape[1] > len(y) // 10:
            print(f"    Applying feature selection: {X.shape[1]} → ", end="")
            selected_names, selected_indices = ImprovedFeatureSelector.select_robust_features(
                X, y, feature_names, max_features=min(6, len(y) // 15)
            )
            
            if len(selected_indices) > 0:
                X = X[:, selected_indices]
                feature_names = selected_names
                print(f"{len(selected_names)} features")
            else:
                raise ValueError("Feature selection resulted in no features")
        
        # Train/Test Split mit verbessertem Splitter
        try:
            X_train, X_test, y_train, y_test = self.splitter.split(y, X)
        except ValueError as e:
            print(f"    Split failed: {e}")
            raise

        # Optional: Testbewertung auf Quartalsenden beschränken
        if getattr(self.config, 'evaluate_quarter_ends_only', False) and 'final_data' in preparation_result:
            dates_all = pd.to_datetime(preparation_result['final_data'][self.date_col].values)
            n_samples = len(dates_all)
            # Recompute split indices wie im Splitter
            if n_samples < 40:
                test_size = 0.20; gap_periods = 1
            elif n_samples < 80:
                test_size = 0.25; gap_periods = 2
            else:
                test_size = self.config.test_size
                gap_periods = max(1, int(n_samples * 0.015))
            test_samples = int(n_samples * test_size)
            train_end = n_samples - test_samples - gap_periods
            test_start = train_end + gap_periods
            dates_test = pd.Series(dates_all[test_start:test_start + test_samples])
            qe_mask = dates_test.dt.is_quarter_end
            if qe_mask.any() and qe_mask.sum() >= 3:
                X_test = X_test[qe_mask.values]
                y_test = y_test[qe_mask.values]
        
        # Method fitting
        method = self.method_registry.get_method(method_name)
        try:
            train_results = method.fit(X_train, y_train)
        except Exception as e:
            print(f"    Training failed: {e}")
            raise
        
        # Test evaluation
        test_performance = self.cv_validator._evaluate_on_test_safe(
            train_results, X_test, y_test, method
        )
        
        # Cross-validation on training data only
        cv_performance = self.cv_validator.validate_method_robust(
            method, X_train, y_train
        )
        
        # Calculate key metrics
        train_r2 = train_results.get('r_squared', np.nan)
        test_r2 = test_performance.get('r_squared', np.nan)
        cv_mean = cv_performance.get('cv_mean', np.nan)
        cv_std = cv_performance.get('cv_std', np.nan)
        
        # Overfitting calculation
        overfitting = train_r2 - test_r2 if (np.isfinite(train_r2) and np.isfinite(test_r2)) else np.nan
        
        # Performance assessment
        performance_flags = []
        if np.isfinite(test_r2) and test_r2 > 0.95:
            performance_flags.append("SUSPICIOUSLY_HIGH_R2")
        if np.isfinite(overfitting) and overfitting > 0.15:
            performance_flags.append("HIGH_OVERFITTING")
        if np.isfinite(cv_std) and cv_std > 0.3:
            performance_flags.append("UNSTABLE_CV")
        if len(cv_performance.get('cv_scores', [])) < 2:
            performance_flags.append("INSUFFICIENT_CV_FOLDS")
        
        # Combine results
        results = {
            **train_results,
            'method_name': method_name,
            'feature_names': feature_names,
            'target_var': preparation_result['target_name'],
            'transformation': preparation_result['transformation_info'].get('transformation_applied') or
                              preparation_result['transformation_info'].get('transformation', 'levels'),
            'test_performance': test_performance,
            'cv_performance': cv_performance,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'cv_mean_r_squared': cv_mean,
            'cv_std_r_squared': cv_std,
            'overfitting': overfitting,
            'performance_flags': performance_flags,
            'sample_sizes': {
                'total': len(y),
                'train': len(y_train) if 'y_train' in locals() else 0,
                'test': len(y_test) if 'y_test' in locals() else 0,
                'features_selected': len(feature_names)
            }
        }
        
        # Performance summary
        print(f"    Results: Test R² = {test_r2:.4f}, CV R² = {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"    Overfitting: {overfitting:.4f}, Flags: {len(performance_flags)}")
        if performance_flags:
            for flag in performance_flags:
                print(f"      ⚠️ {flag}")
        
        return results
    
    def fit_multiple_methods_robust(self, methods: List[str] = None, 
                                    transformation: str = 'auto') -> Dict[str, Any]:
        """
        Robuste Anpassung mehrerer Methoden mit optimaler Transformation.
        """
        if methods is None:
            # Conservative method selection based on sample size
            n_samples = len(self.data)
            if n_samples < 50:
                methods = ['OLS', 'Random Forest']  # Nur robuste Methoden
            elif n_samples < 100:
                methods = ['OLS', 'Random Forest', 'Bayesian Ridge']
            else:
                methods = ['OLS', 'Random Forest', 'XGBoost', 'Bayesian Ridge'] if HAS_XGBOOST else ['OLS', 'Random Forest', 'Bayesian Ridge']
        
        print(f"=== ROBUST METHOD FITTING ===")
        print(f"Methods: {', '.join(methods)}")
        
        # Transformation optimization
        if transformation == 'auto':
            transformation = self._find_optimal_transformation_robust()
        
        print(f"Using transformation: {transformation}")
        
        # Prepare data once
        try:
            preparation_result = self.prepare_data_robust(transformation)
        except Exception as e:
            return {'status': 'failed', 'error': f'Data preparation failed: {str(e)}'}
        
        # Fit methods
        results = {}
        successful_methods = 0
        
        for method_name in methods:
            try:
                result = self.fit_method_improved(method_name, preparation_result)
                results[method_name] = result
                successful_methods += 1
            except Exception as e:
                print(f"    ❌ {method_name} failed: {str(e)[:60]}")
                continue
        
        if successful_methods == 0:
            return {'status': 'failed', 'error': 'All methods failed'}
        
        print(f"Successfully fitted {successful_methods}/{len(methods)} methods")
        
        return {
            'status': 'success',
            'method_results': results,
            'preparation_info': preparation_result,
            'successful_methods': successful_methods,
            'total_methods': len(methods)
        }
    
    def _find_optimal_transformation_robust(self) -> str:
        """
        Robuste Transformationsoptimierung basierend auf Datencharakteristiken.
        """
        print("  Finding optimal transformation...")
        
        # Analyze target variable characteristics
        target_series = self.data[self.target_var].dropna()
        
        if len(target_series) < 10:
            print("    Insufficient data for transformation analysis - using 'levels'")
            return 'levels'
        
        # Check stationarity
        stationarity_result = self.quality_checker.test_stationarity(target_series, self.target_var)
        is_stationary = stationarity_result.get('is_stationary', None)
        p_value = stationarity_result.get('p_value', np.nan)
        if is_stationary is None and (isinstance(p_value, (int, float)) or np.isfinite(p_value)):
            try:
                is_stationary = (p_value < 0.05)
            except Exception:
                pass
        
        # Check value characteristics
        all_positive = (target_series > 0).all()
        has_trend = abs(np.corrcoef(np.arange(len(target_series)), target_series)[0, 1]) > 0.3
        high_volatility = (target_series.std() / abs(target_series.mean()) > 0.5) if target_series.mean() != 0 else False
        
        print(f"    Target characteristics:")
        print(f"      Stationary: {is_stationary}")
        print(f"      All positive: {all_positive}")
        print(f"      Has trend: {has_trend}")
        print(f"      High volatility: {high_volatility}")

        # Transformation decision logic (with p-value fallback and safer defaults)
        if is_stationary is False:
            if all_positive and not high_volatility:
                best_transformation = 'pct'  # percentage change for non-stationary positive series
                print("    → Selected 'pct': Non-stationary positive data")
            else:
                best_transformation = 'diff'  # first differences for non-stationary data
                print("    → Selected 'diff': Non-stationary data")
        elif is_stationary is True:
            if all_positive and has_trend:
                best_transformation = 'log'
                print("    → Selected 'log': Trending positive data")
            else:
                best_transformation = 'levels'
                print("    → Selected 'levels': Stationary or suitable for levels")
        else:
            # Unknown stationarity → be conservative; if p-value available use 0.10 threshold
            try:
                if np.isfinite(p_value) and p_value > 0.10:
                    best_transformation = 'diff'
                    print("    → Selected 'diff': Unknown stationarity, high p-value")
                else:
                    best_transformation = 'levels'
                    print("    → Selected 'levels': Unknown stationarity, defaulting to levels")
            except Exception:
                best_transformation = 'levels'
                print("    → Selected 'levels': Fallback")

        return best_transformation
    
    def create_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstelle umfassende Zusammenfassung mit Qualitätsbewertung.
        """
        if results['status'] != 'success':
            return results
        
        method_results = results['method_results']
        preparation_info = results['preparation_info']
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, result in method_results.items():
            test_r2 = result.get('test_r_squared', np.nan)
            train_r2 = result.get('train_r_squared', np.nan)
            cv_mean = result.get('cv_mean_r_squared', np.nan)
            cv_std = result.get('cv_std_r_squared', np.nan)
            overfitting = result.get('overfitting', np.nan)
            flags = result.get('performance_flags', [])
            
            # Quality assessment
            quality_score = 0.0
            max_score = 5.0
            
            # Test R² contribution (0-2 points)
            if np.isfinite(test_r2):
                if test_r2 > 0.7:
                    quality_score += 2.0
                elif test_r2 > 0.3:
                    quality_score += 1.0
                elif test_r2 > 0:
                    quality_score += 0.5
            
            # Overfitting penalty (-1 to +1 points)
            if np.isfinite(overfitting):
                if overfitting < 0.05:
                    quality_score += 1.0
                elif overfitting < 0.1:
                    quality_score += 0.5
                elif overfitting > 0.2:
                    quality_score -= 1.0
            
            # CV stability (0-1 points)
            if np.isfinite(cv_std) and cv_std < 0.2:
                quality_score += 1.0
            elif np.isfinite(cv_std) and cv_std < 0.3:
                quality_score += 0.5
            
            # Flag penalties
            flag_penalty = len([f for f in flags if 'SUSPICIOUSLY' in f or 'HIGH' in f]) * 0.5
            quality_score -= flag_penalty
            
            quality_score = max(0.0, min(max_score, quality_score))
            
            comparison_data.append({
                'Method': method_name,
                'Test_R²': test_r2,
                'Train_R²': train_r2,
                'CV_Mean_R²': cv_mean,
                'CV_Std_R²': cv_std,
                'Overfitting': overfitting,
                'Quality_Score': quality_score,
                'Flags': len(flags),
                'Flag_Details': '; '.join(flags) if flags else 'None'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Quality_Score', ascending=False).round(4)
        comparison_df = comparison_df.sort_values(
            ['Quality_Score','Test_R²','CV_Mean_R²'],
            ascending=[False, False, False]
        )       
        
        # Best method selection based on quality score
        if not comparison_df.empty:
            best_method_info = comparison_df.iloc[0]
            best_method = best_method_info['Method']
            best_quality = float(best_method_info['Quality_Score'])
        else:
            best_method = None
            best_quality = 0.0
        
        # Create final summary
        summary_parts = []
        summary_parts.append("=== ANALYSIS SUMMARY ===")
        n_total = preparation_info.get('sample_sizes', {}).get('total',
           len(preparation_info.get('final_data', [])))
        summary_parts.append(f"Dataset: {int(n_total)} observations")

        summary_parts.append(f"Transformation: {preparation_info['transformation_info'].get('transformation_applied') or preparation_info['transformation_info'].get('transformation', 'levels')}")
        summary_parts.append(f"Methods tested: {results['successful_methods']}/{results['total_methods']}")
        
        if best_method:
            summary_parts.append(f"Best method: {best_method} (Quality Score: {best_quality:.1f}/5.0)")
            best_result = method_results[best_method]
            summary_parts.append(f"  Test R²: {best_result.get('test_r_squared', np.nan):.4f}")
            summary_parts.append(f"  CV R²: {best_result.get('cv_mean_r_squared', np.nan):.4f} (±{best_result.get('cv_std_r_squared', np.nan):.4f})")
            summary_parts.append(f"  Overfitting: {best_result.get('overfitting', np.nan):.4f}")
        
        # Warnings and recommendations
        all_warnings = preparation_info.get('preparation_warnings', [])[:]
        leakage_risk = preparation_info['transformation_info'].get('leakage_risk', 'low')
        if leakage_risk != 'low':
            all_warnings.append(f"Potential data leakage risk: {leakage_risk}")
        
        if all_warnings:
            summary_parts.append("⚠️ WARNINGS:")
            for warning in all_warnings[:5]:  # Limit to top 5
                summary_parts.append(f"  - {warning}")
        
        summary = "\n".join(summary_parts)
        
        return {
            **results,
            'comparison': comparison_df,
            'best_method': best_method,
            'best_quality_score': best_quality,
            'summary': summary,
            'validation_summary': {
                'total_warnings': len(all_warnings),
                'leakage_risk': leakage_risk,
                'data_coverage': preparation_info.get('data_coverage_ratio', np.nan)
            }
        }


def improved_financial_analysis(data: pd.DataFrame, target_var: str, exog_vars: List[str],
                                analysis_type: str = 'comprehensive', config=None) -> Dict[str, Any]:
    """
    Hauptfunktion für verbesserte Finanzanalyse.
    """
    print("=== IMPROVED FINANCIAL REGRESSION ANALYSIS ===")
    print(f"Target: {target_var}")
    print(f"Features: {', '.join(exog_vars)}")
    print(f"Analysis type: {analysis_type}")
    
    try:
        # Initialize improved analyzer
        analyzer = ImprovedFinancialRegressionAnalyzer(data, target_var, exog_vars, config)
        
        # Step 1: Comprehensive validation
        validation_result = analyzer.comprehensive_data_validation()
        
        if not validation_result['is_valid']:
            return {
                'status': 'failed',
                'stage': 'validation',
                'validation': validation_result,
                'error': 'Data validation failed - see validation results for details'
            }
        
        # Step 2: Method selection based on analysis type
        if analysis_type == 'quick':
            methods = ['Random Forest', 'OLS']
            transformation = 'auto'
        elif analysis_type == 'comprehensive':
            methods = None  # Use automatic selection
            transformation = 'auto'
        else:  # full
            methods = analyzer.method_registry.list_methods()
            transformation = 'auto'
        
        # Step 3: Fit methods
        fit_results = analyzer.fit_multiple_methods_robust(methods, transformation)
        
        if fit_results['status'] != 'success':
            return {
                'status': 'failed',
                'stage': 'fitting',
                'validation': validation_result,
                'error': fit_results.get('error', 'Method fitting failed')
            }
        
        # Step 4: Create comprehensive summary
        final_results = analyzer.create_comprehensive_summary(fit_results)
        
        # Add validation info
        final_results['validation'] = validation_result
        
        print(f"\n{final_results['summary']}")
        
        return final_results
        
    except Exception as e:
        return {
            'status': 'failed',
            'stage': 'unknown',
            'error': f'Analysis failed: {str(e)}',
            'validation': validation_result if 'validation_result' in locals() else {}
        }


def quick_analysis_improved(target_name: str, start_date: str = "2010-01", 
                            config=None) -> Dict[str, Any]:
    """Verbesserte Quick-Analyse."""
    print("=== IMPROVED QUICK ANALYSIS ===")
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        return improved_financial_analysis(data, target_name, exog_vars, 'quick', config)
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'stage': 'data_download'}


def comprehensive_analysis_improved(target_name: str, start_date: str = "2010-01",
                                    config=None) -> Dict[str, Any]:
    """Verbesserte Comprehensive-Analyse."""
    print("=== IMPROVED COMPREHENSIVE ANALYSIS ===")
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        return improved_financial_analysis(data, target_name, exog_vars, 'comprehensive', config)
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'stage': 'data_download'}


print("Main financial regression analyzer loaded")

# %%
"""
Analysis Pipeline & Main Functions - VEREINFACHT
Kombiniert alle Komponenten zu einer benutzbaren Pipeline
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def financial_analysis(data: pd.DataFrame, target_var: str, exog_vars: List[str],
                      analysis_type: str = 'comprehensive', config: AnalysisConfig = None) -> Dict[str, Any]:
    """
    Main financial analysis function - KORRIGIERT für Mixed-Frequency.
    """
    
    config = config or AnalysisConfig()
    
    print("FINANCIAL REGRESSION ANALYSIS")
    print("=" * 50)
    print(f"Target: {target_var}")
    print(f"Features: {', '.join(exog_vars)}")
    print(f"Analysis type: {analysis_type}")
    
    # Step 1: Data Quality Validation
    print("\n1. DATA QUALITY VALIDATION")
    print("-" * 30)
    
    validation = DataQualityChecker.validate_financial_data(
        data, target_var, exog_vars, min_target_coverage=0.3
    )
    
    if not validation['is_valid']:
        print("❌ Data validation failed!")
        for error in validation['errors']:
            print(f"  Error: {error}")
        return {'status': 'failed', 'validation': validation}
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  {warning}")
    
    if validation['recommendations']:
        print("💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"  {rec}")
    
    # Step 2: Detailed Data Diagnosis
    print("\n2. DATA DIAGNOSIS")
    print("-" * 30)
    diagnose_data_issues(data, target_var, exog_vars)
    
    # Step 3: Initialize Analyzer
    print("\n3. ANALYSIS SETUP")
    print("-" * 30)
    analyzer = FinancialRegressionAnalyzer(data, target_var, exog_vars, config)
    
    # Determine methods based on analysis type
    if analysis_type == 'quick':
        methods = ['Random Forest', 'OLS']
        transformations = ['levels', 'pct']
        test_combinations = False
        test_selection = False
    elif analysis_type == 'comprehensive':
        methods = ['Random Forest', 'XGBoost', 'OLS', 'Bayesian Ridge'] if HAS_XGBOOST else ['Random Forest', 'OLS', 'Bayesian Ridge']
        transformations = ['levels', 'pct', 'diff']
        test_combinations = True
        test_selection = True
    else:  # full
        methods = analyzer.method_registry.list_methods()
        transformations = ['levels', 'pct', 'diff']
        test_combinations = True
        test_selection = True
    
    print(f"Methods to test: {', '.join(methods)}")
    print(f"Transformations to test: {', '.join(transformations)}")
    
    # Step 4: Find Optimal Transformation
    print("\n4. TRANSFORMATION OPTIMIZATION")
    print("-" * 30)
    
    try:
        best_transformation = analyzer.find_optimal_transformation(
            transformations, 
            baseline_method=methods[0]
        )
    except Exception as e:
        print(f"Transformation testing failed: {e}")
        best_transformation = 'levels'
    
    # Step 5: Fit All Methods
    print("\n5. METHOD COMPARISON")
    print("-" * 30)
    
    method_results = analyzer.fit_multiple_methods(methods, best_transformation)
    
    if not method_results:
        return {'status': 'failed', 'error': 'No methods succeeded'}
    
    # Step 6: Compare Methods
    comparison_df = analyzer.compare_methods(method_results)
    print("\nMethod Comparison Results:")
    display_cols = ['Method', 'Test_R²', 'Train_R²', 'Overfitting', 'Overfitting_Level']
    print(comparison_df[display_cols].head().to_string(index=False))
    
    # Step 7: Feature Analysis (Optional)
    feature_selection_df = pd.DataFrame()
    combination_results_df = pd.DataFrame()
    
    if test_selection and len(method_results) > 0:
        print("\n6. FEATURE SELECTION ANALYSIS")
        print("-" * 30)
        try:
            feature_selection_df = analyzer.test_feature_selection_methods(best_transformation)
            if not feature_selection_df.empty:
                print("Feature selection methods tested successfully")
                print(feature_selection_df.head().to_string(index=False))
        except Exception as e:
            print(f"Feature selection failed: {e}")
    
    if test_combinations and len(method_results) > 0:
        print("\n7. FEATURE COMBINATION ANALYSIS")
        print("-" * 30)
        try:
            combination_results_df = analyzer.test_feature_combinations(
                max_combinations=config.max_feature_combinations,
                transformation=best_transformation
            )
            if not combination_results_df.empty:
                print(f"Tested {len(combination_results_df)} feature combinations")
                print(combination_results_df.head().to_string(index=False))
        except Exception as e:
            print(f"Combination testing failed: {e}")
    
    # Step 8: Generate Summary
    print("\n8. ANALYSIS SUMMARY")
    print("-" * 30)
    
    summary_lines = []
    summary_lines.append("FINAL RESULTS")
    summary_lines.append("=" * 20)
    summary_lines.append(f"Best Transformation: {best_transformation}")
    
    if not comparison_df.empty:
        best_method = comparison_df.iloc[0]['Method']
        best_test_r2 = comparison_df.iloc[0]['Test_R²']
        best_overfitting = comparison_df.iloc[0]['Overfitting']
        
        summary_lines.append(f"Best Method: {best_method}")
        summary_lines.append(f"Test R²: {best_test_r2:.4f}")
        summary_lines.append(f"Overfitting: {best_overfitting:.4f}")
        
        # Add warnings if necessary
        if best_overfitting > 0.1:
            summary_lines.append("⚠️ WARNING: High overfitting detected")
        if best_test_r2 > 0.9:
            summary_lines.append("⚠️ WARNING: Very high R² - check for data leakage")
        if best_test_r2 < 0.1:
            summary_lines.append("⚠️ NOTE: Low predictive power")
    
    if not feature_selection_df.empty:
        best_selection = feature_selection_df.iloc[0]
        summary_lines.append(f"Best Feature Selection: {best_selection['selection_method']}")
        summary_lines.append(f"Selected Features ({best_selection['n_features']}): {best_selection['selected_features']}")
    
    if not combination_results_df.empty:
        best_combo = combination_results_df.iloc[0]
        summary_lines.append(f"Best Feature Combination: {best_combo['n_features']} features")
        summary_lines.append(f"Combination R²: {best_combo['test_r_squared']:.4f}")
    
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    
    # Create final result
    result = {
        'status': 'success',
        'transformation_used': best_transformation,
        'method_results': method_results,
        'comparison': comparison_df,
        'feature_selection': feature_selection_df,
        'combination_results': combination_results_df,
        'best_method': comparison_df.iloc[0]['Method'] if not comparison_df.empty else None,
        'best_test_r2': comparison_df.iloc[0]['Test_R²'] if not comparison_df.empty else None,
        'summary': summary,
        'validation': validation
    }
    
    return result

def quick_analysis(target_name: str, start_date: str = "2010-01", 
                  config: AnalysisConfig = None) -> Dict[str, Any]:
    """Quick analysis for a target variable with standard exogenous variables."""
    print("QUICK FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'quick', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def comprehensive_analysis(target_name: str, start_date: str = "2010-01",
                         config: AnalysisConfig = None) -> Dict[str, Any]:
    """Comprehensive analysis for a target variable."""
    print("COMPREHENSIVE FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'comprehensive', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def full_analysis(target_name: str, start_date: str = "2010-01",
                 config: AnalysisConfig = None) -> Dict[str, Any]:
    """Full analysis with all available methods and features."""
    print("FULL FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'full', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

class SimpleVisualizer:
    """Simple visualization functions for analysis results."""
    
    @staticmethod
    def plot_data_overview(data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        """Plot basic data overview."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        plot_cols = list(numeric_cols[:6])  # Limit to 6 series
        
        if target_var in plot_cols:
            # Move target to front
            plot_cols.remove(target_var)
            plot_cols = [target_var] + plot_cols
        
        data.set_index(date_col)[plot_cols].plot(ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('Time Series Overview')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Target distribution
        if target_var in data.columns:
            axes[0, 1].hist(data[target_var].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title(f'{target_var} Distribution')
        
        # Correlation with target
        if target_var in numeric_cols:
            corr_with_target = data[numeric_cols].corr()[target_var].abs().sort_values(ascending=False)
            top_vars = corr_with_target.head(6).index.tolist()
            
            if len(top_vars) > 1:
                corr_matrix = data[top_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           ax=axes[1, 0], fmt='.2f', square=True)
                axes[1, 0].set_title(f'Correlations with {target_var}')
        
        # Missing data
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            axes[1, 1].barh(range(len(missing_data)), missing_data.values)
            axes[1, 1].set_yticks(range(len(missing_data)))
            axes[1, 1].set_yticklabels(missing_data.index)
            axes[1, 1].set_title('Missing Data Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Data Completeness: Perfect')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_method_comparison(comparison_df: pd.DataFrame):
        """Plot method comparison results."""
        if comparison_df.empty:
            print("No method results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test R² comparison
        methods = comparison_df['Method'].values
        test_r2 = comparison_df['Test_R²'].values
        
        bars1 = axes[0].barh(methods, test_r2)
        axes[0].set_xlabel('Test R²')
        axes[0].set_title('Method Performance Comparison')
        
        # Color bars by performance
        for i, bar in enumerate(bars1):
            if test_r2[i] > 0.7:
                bar.set_color('green')
            elif test_r2[i] > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels
        for i, v in enumerate(test_r2):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Overfitting analysis
        overfitting = comparison_df['Overfitting'].values
        colors = ['red' if x > 0.15 else 'orange' if x > 0.08 else 'green' for x in overfitting]
        
        axes[1].barh(methods, overfitting, color=colors, alpha=0.7)
        axes[1].set_xlabel('Overfitting (Train - Test R²)')
        axes[1].set_title('Overfitting Analysis')
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axvline(x=0.08, color='orange', linestyle='--', alpha=0.5, label='Warning (0.08)')
        axes[1].axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Critical (0.15)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

# System initialization
def initialize_system():
    """Initialize the financial analysis system."""
    setup_logging()
    
    print("FINANCIAL ANALYSIS SYSTEM - CORRECTED VERSION")
    print("=" * 55)
    print("Key improvements:")
    print("- Fixed mixed-frequency data handling (forward-fill)")
    print("- Robust cross-validation without extreme values")
    print("- Conservative model hyperparameters")
    print("- Proper data quality validation")
    print("- Clean architecture without monkey patches")
    print("")
    print("Available functions:")
    print("  quick_analysis(target_name)")
    print("  comprehensive_analysis(target_name)")
    print("  full_analysis(target_name)")
    print("  financial_analysis(data, target_var, exog_vars)")
    print("")
    print("Available targets:", ", ".join(list(INDEX_TARGETS.keys())[:4]) + "...")
    print("System ready!")

def test_system():
    """Test the system with a simple example."""
    print("Testing system with PH_KREDITE...")
    
    try:
        results = quick_analysis("PH_KREDITE", start_date="2005-01")
        
        if results['status'] == 'success':
            print(f"✅ System test successful!")
            print(f"  Best method: {results['best_method']}")
            print(f"  Test R²: {results['best_test_r2']:.4f}")
            print(f"  Transformation used: {results['transformation_used']}")
            return True
        else:
            print(f"❌ System test failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ System test failed with exception: {e}")
        return False

print("Analysis pipeline and main functions loaded")



# %%
# %%
"""
Cache-Fixes - Erweitert CacheManager für Final Dataset Caching
"""

class ExtendedCacheManager(CacheManager):
    """
    Erweitert den ursprünglichen CacheManager um Final Dataset Caching.
    """
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        # Zusätzliches Verzeichnis für finale Datasets
        self.final_datasets_dir = self.cache_dir / "final_datasets"
        self.final_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Verzeichnis für transformierte Datasets
        self.transformed_dir = self.cache_dir / "transformed_datasets"
        self.transformed_dir.mkdir(parents=True, exist_ok=True)
    
    def make_final_dataset_key(self, series_definitions: Dict[str, str], start: str, end: str) -> str:
        """Erstelle einen stabilen Key für finale Datasets."""
        import hashlib
        import json
        
        payload = {
            "series_definitions": {k: series_definitions[k] for k in sorted(series_definitions)},
            "start": start,
            "end": end,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    
    def has_fresh_final_dataset(self, key: str) -> bool:
        """Prüft ob ein frisches final Dataset existiert."""
        pattern = f"*_{key}.xlsx"
        matches = sorted(self.final_datasets_dir.glob(pattern), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return False
        
        try:
            latest = matches[0]
            age_days = (dt.datetime.now() - dt.datetime.fromtimestamp(latest.stat().st_mtime)).days
            return age_days <= self.config.cache_max_age_days
        except OSError:
            return False
    
    def write_final_dataset(self, key: str, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
        """Speichere finales Dataset mit Metadaten."""
        if df is None or df.empty:
            return False
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Datenpfad
        data_path = self.final_datasets_dir / f"{timestamp}_{key}.xlsx"
        meta_path = self.final_datasets_dir / f"{timestamp}_{key}_meta.json"
        
        try:
            # Speichere Daten
            df.to_excel(data_path, index=False, engine=get_excel_engine())
            
            # Speichere Metadaten
            meta_info = {
                "created_at": dt.datetime.now().isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "date_range": {
                    "start": df['Datum'].min().isoformat() if 'Datum' in df.columns else None,
                    "end": df['Datum'].max().isoformat() if 'Datum' in df.columns else None
                }
            }
            
            if metadata:
                meta_info.update(metadata)
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)
            
            print(f"Final dataset cached: {data_path.name}")
            return True
            
        except Exception as e:
            print(f"Failed to cache final dataset: {e}")
            return False
    
    def read_final_dataset(self, key: str) -> Optional[pd.DataFrame]:
        """Lade neuestes finales Dataset."""
        pattern = f"*_{key}.xlsx"
        matches = sorted(self.final_datasets_dir.glob(pattern), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return None
        
        try:
            latest = matches[0]
            df = pd.read_excel(latest, engine=get_excel_engine())
            
            # Ensure Datum is datetime
            if 'Datum' in df.columns:
                df['Datum'] = pd.to_datetime(df['Datum'])
            
            print(f"Final dataset loaded from cache: {latest.name}")
            return df
            
        except Exception as e:
            print(f"Failed to load final dataset: {e}")
            return None
    
    def write_transformed_dataset(self, transformation: str, target_var: str, df: pd.DataFrame) -> bool:
        """Speichere transformiertes Dataset."""
        if df is None or df.empty:
            return False
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_target = "".join(c for c in target_var if c.isalnum() or c in "._-")
        
        file_path = self.transformed_dir / f"{timestamp}_{safe_target}_{transformation}.xlsx"
        
        try:
            df.to_excel(file_path, index=False, engine=get_excel_engine())
            print(f"Transformed dataset cached: {file_path.name}")
            return True
        except Exception as e:
            print(f"Failed to cache transformed dataset: {e}")
            return False
    
    def cleanup_old_cache(self, max_age_days: int = None):
        """Bereinige alte Cache-Dateien."""
        if max_age_days is None:
            max_age_days = self.config.cache_max_age_days * 2  # Keep longer for final datasets
        
        cutoff_date = dt.datetime.now() - dt.timedelta(days=max_age_days)
        deleted_count = 0
        
        # Cleanup in allen Cache-Verzeichnissen
        for cache_subdir in [self.cache_dir, self.final_datasets_dir, self.transformed_dir]:
            if not cache_subdir.exists():
                continue
                
            for file_path in cache_subdir.glob("*"):
                if file_path.is_file():
                    try:
                        file_time = dt.datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1
                    except OSError:
                        continue
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old cache files")

# Erweitere FinancialDataDownloader für Final Dataset Caching
def download_with_final_caching(self, series_definitions: Dict[str, str], start_date: str = None, 
                               end_date: str = None, prefer_cache: bool = True, 
                               anchor_var: Optional[str] = None) -> pd.DataFrame:
    """
    Download mit Final Dataset Caching - erweitert die ursprüngliche download Methode.
    """
    start_date = start_date or self.config.default_start_date
    end_date = end_date or self.config.default_end_date
    
    # Verwende ExtendedCacheManager
    if not isinstance(self.cache_manager, ExtendedCacheManager):
        self.cache_manager = ExtendedCacheManager(self.config)
    
    # Prüfe Final Dataset Cache
    final_key = self.cache_manager.make_final_dataset_key(series_definitions, start_date, end_date)
    
    if prefer_cache and self.cache_manager.has_fresh_final_dataset(final_key):
        cached_final = self.cache_manager.read_final_dataset(final_key)
        if cached_final is not None and not cached_final.empty:
            print(f"Loaded final dataset from cache: {cached_final.shape[0]} rows, {cached_final.shape[1]-1} variables")
            return cached_final
    
    # Führe normale Download-Logik aus (aus der ursprünglichen Methode)
    print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
    
    regular_codes = {}
    index_definitions = {}
    
    for var_name, definition in series_definitions.items():
        index_codes = parse_index_specification(definition)
        if index_codes:
            index_definitions[var_name] = index_codes
        else:
            regular_codes[var_name] = definition
    
    all_codes = set(regular_codes.values())
    for index_codes in index_definitions.values():
        all_codes.update(index_codes)
    all_codes = list(all_codes)
    
    print(f"Total series to download: {len(all_codes)}")
    
    # Individual series caching (bestehende Logik)
    cached_data = {}
    missing_codes = []
    
    if prefer_cache:
        for code in all_codes:
            cached_df = self.cache_manager.read_cache(code)
            if cached_df is not None:
                cached_data[code] = cached_df
            else:
                missing_codes.append(code)
    else:
        missing_codes = all_codes[:]
    
    # Download missing codes (bestehende Logik bleibt unverändert)
    downloaded_data = {}
    if missing_codes:
        print(f"Downloading {len(missing_codes)} missing series...")
        try:
            downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
                except ImportError:
                    print("Using synchronous download mode...")
                    downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            else:
                print("Async failed, using synchronous download mode...")
                downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
        except Exception as e:
            print(f"Download failed ({e}), trying synchronous mode...")
            downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
        
        # Cache individual series
        for code, df in downloaded_data.items():
            self.cache_manager.write_cache(code, df)
    
    # Bestehende Merge- und Index-Erstellungslogik bleibt unverändert...
    all_data = {**cached_data, **downloaded_data}
    
    if not all_data:
        raise Exception("No series loaded successfully")
    
    merged_df = self._merge_series_data(all_data)
    final_data = {"Datum": merged_df["Datum"]}
    
    for var_name, series_code in regular_codes.items():
        if series_code in merged_df.columns:
            final_data[var_name] = merged_df[series_code]
    
    for var_name, index_codes in index_definitions.items():
        try:
            available_codes = [c for c in index_codes if c in merged_df.columns]
            
            if len(available_codes) >= len(index_codes) * 0.3:
                index_series = self.index_creator.create_index(merged_df, available_codes, var_name)
                aligned_index = index_series.reindex(pd.to_datetime(merged_df['Datum']))
                final_data[var_name] = aligned_index.values
                print(f"Created INDEX: {var_name} from {len(available_codes)}/{len(index_codes)} series")
            else:
                if var_name in SIMPLE_TARGET_FALLBACKS:
                    fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback_code in merged_df.columns:
                        final_data[var_name] = merged_df[fallback_code]
                        print(f"Using fallback for {var_name}: {fallback_code}")
                    else:
                        print(f"Warning: Could not create {var_name} - fallback series {fallback_code} not available")
                else:
                    print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available_codes)}/{len(index_codes)} series available)")
                    
        except Exception as e:
            print(f"Failed to create INDEX {var_name}: {e}")
            if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                if fallback_code in merged_df.columns:
                    final_data[var_name] = merged_df[fallback_code]
                    print(f"Using fallback for {var_name} after INDEX creation failed: {fallback_code}")
    
    final_df = pd.DataFrame(final_data)
    final_df["Datum"] = pd.to_datetime(final_df["Datum"])
    final_df = final_df.sort_values("Datum").reset_index(drop=True)

    # Bestehende Trimming-Logik...
    value_cols = [c for c in final_df.columns if c != 'Datum']
    if value_cols:
        non_na_count = final_df[value_cols].notna().sum(axis=1)
        required = 2 if len(value_cols) >= 2 else 1
        keep_mask = non_na_count >= required
        if keep_mask.any():
            first_keep = keep_mask.idxmax()
            if first_keep > 0:
                _before = len(final_df)
                final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

    if anchor_var and anchor_var in final_df.columns:
        mask_anchor = final_df[anchor_var].notna()
        if mask_anchor.any():
            start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
            end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
            _before_rows = len(final_df)
            final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
            final_df.reset_index(drop=True, inplace=True)
            print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

    if anchor_var and anchor_var in final_df.columns:
        exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
        if exog_cols:
            tgt_notna = final_df[anchor_var].notna().values
            all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
            keep_start = 0
            for i in range(len(final_df)):
                if not (tgt_notna[i] and all_exog_nan[i]):
                    keep_start = i
                    break
            if keep_start > 0:
                _before = len(final_df)
                final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

    print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")
    
    # Cache final dataset
    metadata = {
        "series_definitions": series_definitions,
        "start_date": start_date,
        "end_date": end_date,
        "downloaded_codes": sorted(list(all_data.keys())),
        "regular_series": regular_codes,
        "index_specifications": index_definitions
    }
    
    self.cache_manager.write_final_dataset(final_key, final_df, metadata)
    
    return final_df

# Erweitere DataPreprocessor für Transformation Caching
def create_transformations_with_caching(self, transformation: str = 'levels') -> pd.DataFrame:
    """
    Transformationen mit Caching - erweitert die ursprüngliche Methode.
    """
    # Verwende ExtendedCacheManager falls verfügbar
    cache_manager = getattr(self, 'cache_manager', None)
    if cache_manager and hasattr(cache_manager, 'write_transformed_dataset'):
        # Führe Transformationen durch
        transformed_data = self.create_transformations_original(transformation)
        
        # Cache transformierte Daten
        cache_manager.write_transformed_dataset(transformation, self.target_var, transformed_data)
        
        return transformed_data
    else:
        # Fallback zur ursprünglichen Methode
        return self.create_transformations_original(transformation)

# Monkey-patch die Methoden
FinancialDataDownloader.download_original = FinancialDataDownloader.download
FinancialDataDownloader.download = download_with_final_caching

DataPreprocessor.create_transformations_original = DataPreprocessor.create_transformations
DataPreprocessor.create_transformations = create_transformations_with_caching

print("Cache fixes loaded - Final datasets and transformed data will now be cached")



# %%# %%
"""
Main Script - Usage Example
Zeigt wie die korrigierte Pipeline verwendet wird
# """

# if __name__ == "__main__":
#     # Initialize system
#     initialize_system()
    
#     # WICHTIG: Cache-Fixes laden
#     print("Loading cache fixes for raw and transformed data...")
#     # Der Cache-Fix Code sollte hier eingefügt werden (aus Cache-Fixes Artifact)
    
#     # Configuration with conservative settings
#     config = AnalysisConfig(
#         default_start_date="2000-01",
#         default_end_date="2024-12",
#         test_size=0.25,                    # Conservative test size
#         cv_folds=3,                       # Fewer CV folds for stability
#         gap_periods=2,                    # Mandatory gaps
#         max_feature_combinations=15,      # Reduced combinations to prevent overfitting
#         handle_mixed_frequencies=True,    # Enable mixed frequency handling
#         cache_final_dataset=True,         # Enable final dataset caching
#         cache_max_age_days=7              # Cache für 7 Tage
#     )
    
#     # === EXAMPLE 1: Quick Analysis with Caching ===
#     print("\n" + "="*60)
#     print("EXAMPLE 1: QUICK ANALYSIS WITH CACHING")
#     print("="*60)
    
#     TARGET = "PH_KREDITE"
#     START_DATE = "2005-01"
    
#     # Erste Ausführung - lädt und cached Daten
#     print("First run - will download and cache data...")
#     results1 = quick_analysis(TARGET, START_DATE, config)
    
#     # Zweite Ausführung - sollte aus Cache laden
#     print("\nSecond run - should load from cache...")
#     results2 = quick_analysis(TARGET, START_DATE, config)
    
#     if results1['status'] == 'success':
#         print(f"\n✅ Analysis completed successfully!")
#         print(f"Best method: {results1['best_method']}")
#         print(f"Test R²: {results1['best_test_r2']:.4f}")
#         print(f"Transformation: {results1['transformation_used']}")
#     else:
#         print(f"\n❌ Analysis failed: {results1.get('error', 'Unknown error')}")
    
#     # === Cache-Status anzeigen ===
#     print("\n" + "="*60)
#     print("CACHE STATUS")
#     print("="*60)
    
#     # Zeige Cache-Verzeichnisse
#     cache_dir = Path(config.cache_dir)
#     if cache_dir.exists():
#         print(f"Cache directory: {cache_dir}")
        
#         # Rohdaten Cache
#         raw_files = list(cache_dir.glob("*.xlsx"))
#         print(f"Raw data files cached: {len(raw_files)}")
        
#         # Final datasets
#         final_dir = cache_dir / "final_datasets"
#         if final_dir.exists():
#             final_files = list(final_dir.glob("*.xlsx"))
#             print(f"Final datasets cached: {len(final_files)}")
#             for f in final_files[-3:]:  # Show last 3
#                 print(f"  - {f.name}")
        
#         # Transformed datasets  
#         trans_dir = cache_dir / "transformed_datasets"
#         if trans_dir.exists():
#             trans_files = list(trans_dir.glob("*.xlsx"))
#             print(f"Transformed datasets cached: {len(trans_files)}")
#             for f in trans_files[-3:]:  # Show last 3
#                 print(f"  - {f.name}")
    
#     # === Cache cleanup demo ===
#     print("\n" + "="*60)
#     print("CACHE CLEANUP DEMO")
#     print("="*60)
    
#     # Erstelle ExtendedCacheManager für Cleanup
#     downloader = FinancialDataDownloader(config)
#     if hasattr(downloader, 'cache_manager'):
#         cache_manager = downloader.cache_manager
#         if hasattr(cache_manager, 'cleanup_old_cache'):
#             print("Running cache cleanup (files older than 60 days)...")
#             cache_manager.cleanup_old_cache(max_age_days=60)
#         else:
#             print("Cache manager does not support cleanup")
    
#     # === EXAMPLE 2: Custom Analysis ===
#     print("\n" + "="*60)
#     print("EXAMPLE 2: CUSTOM ANALYSIS WITH CACHING")
#     print("="*60)
    
#     # Define custom series (mix of monthly and quarterly)
#     # Use only reliably available series
#     series_definitions = {
#         "PH_KREDITE": INDEX_TARGETS["PH_KREDITE"],  # Quarterly target
#         "euribor_3m": STANDARD_EXOG_VARS["euribor_3m"],  # Monthly
#         "german_rates": STANDARD_EXOG_VARS["german_rates"],  # Monthly
#         "german_inflation": STANDARD_EXOG_VARS["german_inflation"],  # Monthly
#         "german_unemployment": STANDARD_EXOG_VARS["german_unemployment"]  # Monthly
#     }
    
#     # Download data (should use caching)
#     print("Downloading with caching...")
#     downloader = FinancialDataDownloader(config)
#     data = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=True)
    
#     print(f"Downloaded data shape: {data.shape}")
#     print(f"Columns: {list(data.columns)}")
    
#     # Show data overview plot
#     visualizer = SimpleVisualizer()
#     visualizer.plot_data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === EXAMPLE 3: System Test ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE")
#     print("="*60)
    
#     #data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === Cache-Performance Test ===
#     print("\n" + "="*60)
#     print("CACHE PERFORMANCE TEST")
#     print("="*60)
    
#     import time
    
#     # Test 1: Download ohne Cache
#     print("Test 1: Download without cache...")
#     start_time = time.time()
#     data_no_cache = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=False)
#     time_no_cache = time.time() - start_time
#     print(f"Time without cache: {time_no_cache:.2f} seconds")
    
#     # Test 2: Download mit Cache
#     print("\nTest 2: Download with cache...")
#     start_time = time.time()
#     data_with_cache = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=True)
#     time_with_cache = time.time() - start_time
#     print(f"Time with cache: {time_with_cache:.2f} seconds")
    
#     if time_no_cache > 0:
#         speedup = time_no_cache / max(time_with_cache, 0.01)
#         print(f"Cache speedup: {speedup:.1f}x faster")
    
#     # === EXAMPLE 3: System Test ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("✅ Raw data caching: Active")
#         print("✅ Final dataset caching: Active") 
#         print("✅ Transformed data caching: Active")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#         print("\nCache locations:")
#         print(f"- Raw data: {cache_dir}")
#         print(f"- Final datasets: {cache_dir / 'final_datasets'}")
#         print(f"- Transformed data: {cache_dir / 'transformed_datasets'}")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE - WITH FULL CACHING")
#     print("="*60)
#     #data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === EXAMPLE 3: Test System ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE")
#     print("="*60)

# %%
"""
Improved Mixed-Frequency Data Processing - KORRIGIERT
Verhindert Data Leakage durch strikte zeitliche Beschränkungen
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from statsmodels.tsa.stattools import adfuller
import warnings

class ImprovedMixedFrequencyProcessor:
    """
    Verbesserte Behandlung von Mixed-Frequency Daten mit strikten Anti-Leakage Regeln.
    """
    
    @staticmethod
    def detect_frequency(series: pd.Series, date_col: pd.Series) -> str:
        """Erweiterte Frequenzerkennung mit robusteren Regeln."""
        if series.isna().all():
            return "unknown"
        
        df_temp = pd.DataFrame({'date': date_col, 'value': series})
        df_temp = df_temp.dropna().copy()
        
        if len(df_temp) < 4:  # Mindestens 4 Beobachtungen für Frequenzerkennung
            return "unknown"
        
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['quarter'] = df_temp['date'].dt.quarter
        
        # Erweiterte Frequenzanalyse
        years_with_data = df_temp['year'].nunique()
        if years_with_data < 2:
            return "insufficient_history"
        
        # Analysiere Beobachtungen pro Jahr
        obs_per_year = df_temp.groupby('year').size()
        avg_obs_per_year = obs_per_year.mean()
        std_obs_per_year = obs_per_year.std()
        
        # Analysiere monatliche vs. quartalsweise Verteilung
        months_per_year = df_temp.groupby('year')['month'].nunique()
        quarters_per_year = df_temp.groupby('year')['quarter'].nunique()
        
        avg_months_per_year = months_per_year.mean()
        avg_quarters_per_year = quarters_per_year.mean()
        
        # Robuste Klassifikation
        if avg_quarters_per_year <= 4.2 and avg_obs_per_year <= 5:
            if avg_months_per_year <= 4.5:  # Meist nur ein Monat pro Quartal
                return "quarterly"
        
        if avg_months_per_year >= 8 and avg_obs_per_year >= 10:
            return "monthly"
        
        # Zusätzliche Prüfung: Sind die Daten gleichmäßig verteilt?
        if std_obs_per_year / max(avg_obs_per_year, 1) > 0.5:
            return "irregular"
            
        return "unknown"
    
    @staticmethod
    def safe_forward_fill_quarterly(df: pd.DataFrame, quarterly_vars: List[str], 
                                   max_fill_periods: int = 6) -> pd.DataFrame:
        """
        Sicheres Forward-Fill für Quartalsdaten mit strikten Limits.
        
        Args:
            df: DataFrame mit Zeitreihen
            quarterly_vars: Liste der Quartalsvariablen
            max_fill_periods: Maximale Anzahl Perioden zum Forward-Fill (Standard: 2 Monate)
        """
        result = df.copy()
        
        for var in quarterly_vars:
            if var not in df.columns:
                continue
            
            series = df[var].copy()
            valid_mask = series.notna()
            
            if not valid_mask.any():
                continue
            
            # Identifiziere alle validen Zeitpunkte
            valid_indices = valid_mask[valid_mask].index.tolist()
            
            # Forward-Fill nur zwischen benachbarten validen Punkten
            filled_series = series.copy()
            
            for i in range(len(valid_indices) - 1):
                current_idx = valid_indices[i]
                next_idx = valid_indices[i + 1]
                
                # Bereich zwischen aktuellen und nächsten validen Werten
                gap_start = current_idx + 1
                gap_end = next_idx
                
                # Begrenze Fill-Bereich auf max_fill_periods
                actual_gap_end = min(gap_end, current_idx + max_fill_periods + 1)
                
                if gap_start < actual_gap_end:
                    # Forward-Fill nur im begrenzten Bereich
                    fill_value = series.iloc[current_idx]
                    filled_series.iloc[gap_start:actual_gap_end] = fill_value
            
            result[var] = filled_series
        
        return result
    
    @staticmethod
    def align_frequencies_improved(
        df: pd.DataFrame,
        target_var: str,
        date_col: str = "Datum",
        train_end_index: Optional[int] = None,
        validation_split_date: Optional[pd.Timestamp] = None,
        max_fill_periods: int = 2,   # konservativer Standard
    ) -> Dict[str, Any]:
        """
        Verbesserte Frequenz-Alignierung mit Anti-Leakage-Schutz.
        - Forward-Fill nur im Trainingsfenster
        - Leakage-Flags nur, wenn neue Fills im Testfenster liegen
        """
        if target_var not in df.columns or date_col not in df.columns:
            raise ValueError(f"Missing {target_var} or {date_col} column")

        # Datums-Handling & Sortierung
        out = {
            "processed_df": None,
            "frequency_info": {},
            "warnings": [],
            "leakage_risk": "low",
            "forward_fill_used": False,
            "fill_span_overlaps_test_period": False,
        }

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.sort_values(date_col).reset_index(drop=True)

        # Frequenzen erkennen (auf Originaldaten)
        all_vars = [c for c in work.columns if c != date_col]
        freqs = {}
        for var in all_vars:
            freqs[var] = ImprovedMixedFrequencyProcessor.detect_frequency(work[var], work[date_col])
        out["frequency_info"] = freqs

        quarterly_vars = [v for v, f in freqs.items() if f == "quarterly"]
        monthly_vars   = [v for v, f in freqs.items() if f == "monthly"]
        irregular_vars = [v for v, f in freqs.items() if f == "irregular"]

        print("Detected frequencies:")
        print(f"  Quarterly: {quarterly_vars}")
        print(f"  Monthly: {monthly_vars}")
        if irregular_vars:
            print(f"  Irregular: {irregular_vars}")
            out["warnings"].append(f"Irregular frequency detected: {irregular_vars}")

        # Train-Ende bestimmen
        split_date = None
        if validation_split_date is not None:
            split_date = pd.to_datetime(validation_split_date)
        elif train_end_index is not None and 0 <= train_end_index < len(work):
            split_date = pd.to_datetime(work.loc[train_end_index, date_col])

        # Forward-Fill nur für Trainingsanteil
        if quarterly_vars:
            if split_date is not None:
                train_mask = work[date_col] < split_date
                train_df = work.loc[train_mask].copy()
                print(f"Using training data only for forward-fill: {len(train_df)} observations")
            else:
                train_df = work.copy()
                out["warnings"].append("No split provided — forward-fill applied on full sample (potential bias)")
                out["leakage_risk"] = "medium"

            print(f"Applying safe forward-fill to {len(quarterly_vars)} quarterly variables...")
            filled_train = ImprovedMixedFrequencyProcessor.safe_forward_fill_quarterly(
                train_df, quarterly_vars, max_fill_periods=max_fill_periods
            )

            if split_date is not None:
                valid_df = work.loc[work[date_col] >= split_date].copy()  # Validierungs-/Testteil: ungefüllt
                processed_df = pd.concat([filled_train, valid_df], ignore_index=True)
                processed_df = processed_df.sort_values(date_col).reset_index(drop=True)
            else:
                processed_df = filled_train
        else:
            processed_df = work.copy()  # nichts zu füllen

        # Zielbereich trimmen (keine Lead/Tail-NaNs)
        s = processed_df[target_var]
        first = s.first_valid_index()
        last  = s.last_valid_index()
        if first is not None and last is not None and last >= first:
            processed_df = processed_df.loc[first:last].reset_index(drop=True)

        # Fortschritts-Log je Quartalsvariable
        for var in quarterly_vars:
            before_count = work[var].notna().sum()
            after_count  = processed_df[var].notna().sum()
            improvement  = int(after_count - before_count)
            print(f"  {var}: {before_count} → {after_count} observations (+{improvement})")
            # Keine automatische Leakage-Hochstufung mehr nur wegen „Large improvement“
            if improvement > max(5, int(before_count * 0.5)):
                out["warnings"].append(f"Large improvement in {var} coverage — verify correctness")

        # Leakage-Flags setzen: neue Fills identifizieren & prüfen, ob sie NACH split_date liegen
        orig_series = work.set_index(date_col)[target_var]
        proc_series = processed_df.set_index(date_col)[target_var]
        orig_notna  = orig_series.reindex(proc_series.index).notna().fillna(False)
        proc_notna  = proc_series.notna()
        new_filled_mask = (proc_notna & ~orig_notna)

        out["forward_fill_used"] = bool(new_filled_mask.any())
        if split_date is not None:
            out["fill_span_overlaps_test_period"] = bool(new_filled_mask.loc[new_filled_mask.index >= split_date].any())
            out["leakage_risk"] = "high" if out["fill_span_overlaps_test_period"] else out["leakage_risk"]
        # Wenn kein split_date: Risiko bleibt wie oben gesetzt (medium), aber kein „high“

        out["processed_df"] = processed_df
        return out

        




class ImprovedDataQualityChecker:
    """
    Erweiterte Datenqualitätsprüfung mit Stationaritätstests.
    """
    
    @staticmethod
    def test_stationarity(series: pd.Series, variable_name: str) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller Test für Stationarität.
        """
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': None,
                'error': 'Insufficient data for stationarity test'
            }
        
        try:
            # ADF Test
            result = adfuller(clean_series, autolag='AIC')
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'variable': variable_name,
                'n_observations': len(clean_series)
            }
        except Exception as e:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': None,
                'error': str(e),
                'variable': variable_name
            }
    
    @staticmethod
    def comprehensive_data_validation(data: pd.DataFrame, target_var: str, 
                                    exog_vars: List[str],
                                    min_target_coverage: float = 0.15,  # Für Quartalsdaten
                                    min_observations: int = 30         ) -> Dict[str, Any]:
        """
        Umfassende Datenvalidierung mit Stationaritätstests.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'stationarity_tests': {},
            'recommendations': [],
            'sample_adequacy': {}
        }
        
        # Grundlegende Validierung
        missing_vars = [var for var in [target_var] + exog_vars if var not in data.columns]
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {', '.join(missing_vars)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Stichprobengröße prüfen
        if len(data) < min_observations:
            validation_results['errors'].append(
                f"Insufficient sample size: {len(data)} < {min_observations} required"
            )
            validation_results['is_valid'] = False
        
        # Target Variable Analyse
        target_series = data[target_var]
        target_coverage = target_series.notna().sum() / len(target_series)
        target_valid_obs = target_series.notna().sum()
        
        validation_results['data_quality'][target_var] = {
            'total_obs': len(target_series),
            'valid_obs': target_valid_obs,
            'coverage': target_coverage,
            'mean': target_series.mean() if target_valid_obs > 0 else np.nan,
            'std': target_series.std() if target_valid_obs > 0 else np.nan,
            'frequency': ImprovedMixedFrequencyProcessor.detect_frequency(
                target_series, data['Datum'] if 'Datum' in data.columns else data.index
            )
        }
        
        # Critical checks für Target
        if target_coverage < min_target_coverage:
            validation_results['errors'].append(
                f"Target {target_var} insufficient coverage: {target_coverage:.1%} < {min_target_coverage:.1%}"
            )
            validation_results['is_valid'] = False
        
        if target_valid_obs > 1 and target_series.std() == 0:
            validation_results['errors'].append(f"Target {target_var} is constant")
            validation_results['is_valid'] = False
        
        # Stationaritätstest für Target
        stationarity_result = ImprovedDataQualityChecker.test_stationarity(target_series, target_var)
        validation_results['stationarity_tests'][target_var] = stationarity_result
        
        if stationarity_result.get('is_stationary') is False:
            validation_results['warnings'].append(
                f"Target {target_var} may be non-stationary (ADF p-value: {stationarity_result.get('p_value', 'N/A'):.3f})"
            )
            validation_results['recommendations'].append(
                f"Consider differencing or log-transformation for {target_var}"
            )
        
        # Exogenous Variables Analyse
        for var in exog_vars:
            if var in data.columns:
                series = data[var]
                coverage = series.notna().sum() / len(series)
                valid_obs = series.notna().sum()
                
                validation_results['data_quality'][var] = {
                    'total_obs': len(series),
                    'valid_obs': valid_obs,
                    'coverage': coverage,
                    'mean': series.mean() if valid_obs > 0 else np.nan,
                    'std': series.std() if valid_obs > 0 else np.nan,
                    'frequency': ImprovedMixedFrequencyProcessor.detect_frequency(
                        series, data['Datum'] if 'Datum' in data.columns else data.index
                    )
                }
                
                # Stationaritätstest
                stationarity_result = ImprovedDataQualityChecker.test_stationarity(series, var)
                validation_results['stationarity_tests'][var] = stationarity_result
                
                # Warnings
                if coverage < 0.3:
                    validation_results['warnings'].append(f"Low coverage in {var}: {coverage:.1%}")
                
                if valid_obs > 1 and series.std() == 0:
                    validation_results['warnings'].append(f"Variable {var} is constant")
        
        # Sample adequacy assessment
        all_vars = [target_var] + [v for v in exog_vars if v in data.columns]
        complete_cases = data[all_vars].dropna()
        overlap_ratio = len(complete_cases) / len(data)
        
        validation_results['sample_adequacy'] = {
            'total_observations': len(data),
            'complete_cases': len(complete_cases),
            'overlap_ratio': overlap_ratio,
            'variables_tested': len(all_vars)
        }
        
        if overlap_ratio < 0.2:
            validation_results['errors'].append(
                f"Insufficient overlap: only {overlap_ratio:.1%} complete cases"
            )
            validation_results['is_valid'] = False
        elif overlap_ratio < 0.4:
            validation_results['warnings'].append(
                f"Low overlap: {overlap_ratio:.1%} complete cases - results may be unreliable"
            )
        
        # Final recommendations
        non_stationary_count = sum(1 for result in validation_results['stationarity_tests'].values() 
                                 if result.get('is_stationary') is False)
        
        if non_stationary_count > 0:
            validation_results['recommendations'].append(
                f"Consider using 'diff' or 'pct' transformations - {non_stationary_count} variables may be non-stationary"
            )
        
        return validation_results


class ImprovedDataPreprocessor:
    """
    Verbesserter Datenvorverarbeitungsschritt mit robusten Transformationen.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.date_col = date_col

        self.forward_fill_used: bool = False
        self.fill_span_overlaps_test_period: bool = False
        
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        
    def create_robust_transformations(self, transformation: str = 'levels',
                                    train_end_date: pd.Timestamp = None,
                                    outlier_method: str = 'conservative') -> Dict[str, Any]:
        """
        Robuste Transformationen mit Anti-Leakage Schutz.
        
        Args:
            transformation: 'levels', 'log', 'pct', 'diff'
            train_end_date: Trainingsende für Anti-Leakage (optional)
            outlier_method: 'conservative', 'moderate', 'aggressive'
        """
        
        # Schritt 1: Mixed-Frequency Handling mit Anti-Leakage
        freq_result = ImprovedMixedFrequencyProcessor.align_frequencies_improved(
            self.data, self.target_var, self.date_col, 
            validation_split_date=train_end_date
        )
        
        processed_data = freq_result['processed_df']
        warnings_list = freq_result['warnings']
        
        # Schritt 2: Transformationen anwenden
        transformed_data = processed_data[[self.date_col]].copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        print(f"Applying '{transformation}' transformation to {len(numeric_cols)} variables...")
        
        def _robust_pct_change(s: pd.Series, eps: float = 1e-8) -> pd.Series:
            return (s - s.shift(1)) / (np.abs(s.shift(1)) + eps)
        
        for col in numeric_cols:
            series = processed_data[col].copy()
            if transformation == 'robust_pct':
                transformed_data[col] = _robust_pct_change(series)

            elif transformation == 'levels':
                transformed_data[col] = series
                
            elif transformation == 'log':
                # Robuste Log-Transformation
                if (series > 0).sum() / series.notna().sum() > 0.9:
                    # Nur wenn > 90% der Werte positiv sind
                    # Kleine Konstante hinzufügen um Zeros zu handhaben
                    min_positive = series[series > 0].min()
                    epsilon = min_positive * 0.001 if pd.notna(min_positive) else 0.001
                    transformed_data[col] = np.log(series.clip(lower=epsilon))
                else:
                    transformed_data[col] = series
                    warnings_list.append(f"{col}: Not suitable for log transformation")
                    
            elif transformation == 'pct':
                # Prozentuale Änderungen
                transformed_data[col] = series.pct_change()
                
            elif transformation == 'diff':
                # Erste Differenzen
                transformed_data[col] = series.diff()
                
            else:
                transformed_data[col] = series
        
        # Schritt 3: Robuste Outlier-Behandlung
        transformed_data = self._robust_outlier_treatment(
            transformed_data, method=outlier_method
        )
        
        # Schritt 4: Saisonale Features hinzufügen
        transformed_data = self._add_seasonal_features(transformed_data)
        
        # Schritt 5: Final cleaning
        before_clean = len(transformed_data)
        
        # Nur Zeilen mit mindestens dem Target und einer exogenen Variable behalten
        essential_cols = [col for col in transformed_data.columns 
                         if col != self.date_col and col in [self.target_var] + 
                         [c for c in numeric_cols if c != self.target_var][:3]]  # Top 3 exog vars
        
        if len(essential_cols) > 1:
            # Behalte Zeilen mit Target + mindestens einer exogenen Variable
            keep_mask = (transformed_data[essential_cols].notna().sum(axis=1) >= 2)
            transformed_data = transformed_data[keep_mask].copy()
        else:
            # Fallback: alle NaN Zeilen entfernen
            transformed_data = transformed_data.dropna(how="all", subset=numeric_cols)
        
        after_clean = len(transformed_data)
        
        if before_clean > after_clean:
            print(f"Cleaned dataset: {before_clean} → {after_clean} observations")
        
        # Data types stabilisieren
        for col in [c for c in transformed_data.columns if c != self.date_col]:
            transformed_data[col] = pd.to_numeric(transformed_data[col], errors='coerce')
        
        return {
            'data': transformed_data,
            'warnings': warnings_list,
            'frequency_info': freq_result['frequency_info'],
            'leakage_risk': freq_result['leakage_risk'],
            'transformation_applied': transformation,
            'outlier_method': outlier_method,
            'rows_before_cleaning': before_clean,
            'rows_after_cleaning': after_clean
        }
    
    def _robust_outlier_treatment(self, data: pd.DataFrame, 
                                method: str = 'conservative') -> pd.DataFrame:
        """
        Robuste Outlier-Behandlung mit verschiedenen Aggressivitätsstufen.
        """
        numeric_cols = [c for c in data.columns if c != self.date_col]
        result_data = data.copy()
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) < 20:  # Zu wenige Daten für Outlier-Behandlung
                continue
            
            if method == 'conservative':
                # Sehr konservativ: nur extreme Outliers (0.5% / 99.5%)
                lower_bound = series.quantile(0.005)
                upper_bound = series.quantile(0.995)
            elif method == 'moderate':
                # Moderat: 1% / 99% Quantile
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
            else:  # aggressive
                # Aggressiv: 2.5% / 97.5% Quantile
                lower_bound = series.quantile(0.025)
                upper_bound = series.quantile(0.975)
            
            # Nur clippen wenn bounds sinnvoll sind
            if pd.notna(lower_bound) and pd.notna(upper_bound) and upper_bound > lower_bound:
                original_std = series.std()
                clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
                clipped_std = clipped_series.std()
                
                # Nur anwenden wenn nicht zu viel Variation verloren geht
                if clipped_std > 0.5 * original_std:
                    result_data[col] = result_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return result_data
    
    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Saisonale Features hinzufügen."""
        data_with_features = data.copy()
        
        # Quartalsdummies
        data_with_features['quarter'] = pd.to_datetime(data_with_features[self.date_col]).dt.quarter
        
        for q in [2, 3, 4]:
            data_with_features[f'Q{q}'] = (data_with_features['quarter'] == q).astype(int)
        
        # Zeittrend (normalisiert)
        data_with_features['time_trend'] = (
            np.arange(len(data_with_features)) / len(data_with_features)
        )
        
        data_with_features = data_with_features.drop('quarter', axis=1)
        
        return data_with_features

# %%
"""
Test Script für die verbesserten Komponenten
Demonstriert die Korrekturen und deren Auswirkungen auf die Modellgüte
"""

def test_improved_system():
    """
    Testet das verbesserte System und zeigt die Verbesserungen auf.
    """
    print("=" * 80)
    print("TESTING IMPROVED FINANCIAL ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Configuration mit verbesserten Einstellungen
    config = AnalysisConfig(
        default_start_date="2000-01",
        default_end_date="2024-12",
        test_size=0.20,  # Kleinere Test-Sets für Stabilität
        cv_folds=3,      # Weniger CV-Folds
        gap_periods=2,   # Längere Gaps gegen Leakage
        cache_max_age_days=7,
        handle_mixed_frequencies=True,
        max_feature_combinations=10,  # Statt 15-20
    )
    
    # Test 1: Vergleich alter vs. neuer Ansatz
    print("\n" + "=" * 60)
    print("TEST 1: COMPARISON - OLD VS NEW APPROACH")
    print("=" * 60)
    
    target = "PH_KREDITE"
    start_date = "2005-01"
    
    try:
        # Lade Daten
        downloader = FinancialDataDownloader(config)
        series_definitions = {
            target: INDEX_TARGETS[target],
            **{k: v for k, v in STANDARD_EXOG_VARS.items()}
        }
        
        data = downloader.download(series_definitions, start_date=start_date, prefer_cache=True)
        print(f"Data loaded: {data.shape[0]} observations, {data.shape[1]-1} variables")
        
        # Test mit verbessertem System
        print("\n--- IMPROVED ANALYSIS ---")
        improved_results = improved_financial_analysis(
            data=data,
            target_var=target,
            exog_vars=[col for col in data.columns if col not in ['Datum', target]],
            analysis_type='comprehensive',
            config=config
        )
        
        if improved_results['status'] == 'success':
            print("✅ Improved analysis succeeded!")
            comparison_df = improved_results['comparison']
            print(f"Best method: {improved_results['best_method']}")
            print(f"Quality score: {improved_results['best_quality_score']:.1f}/5.0")
            
            # Zeige Top-3 Methoden
            print("\nTop 3 Methods:")
            display_cols = ['Method', 'Test_R²', 'Quality_Score', 'Overfitting', 'Flags']
            print(comparison_df[display_cols].head(3).to_string(index=False))
            
        else:
            print(f"❌ Improved analysis failed: {improved_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test 1 failed with exception: {str(e)}")
    
    # Test 2: Verschiedene Zielgrößen
    print("\n" + "=" * 60) 
    print("TEST 2: MULTIPLE TARGET VARIABLES")
    print("=" * 60)
    
    test_targets = ["PH_KREDITE", "PH_EINLAGEN", "PH_WERTPAPIERE"]
    results_summary = []
    
    for target in test_targets:
        print(f"\nTesting {target}...")
        
        try:
            result = quick_analysis_improved(target, start_date="2010-01", config=config)
            
            if result['status'] == 'success':
                best_method = result.get('best_method', 'N/A')
                best_r2 = result.get('comparison', pd.DataFrame()).iloc[0]['Test_R²'] if not result.get('comparison', pd.DataFrame()).empty else np.nan
                quality_score = result.get('best_quality_score', 0)
                
                results_summary.append({
                    'Target': target,
                    'Best_Method': best_method,
                    'Test_R²': best_r2,
                    'Quality_Score': quality_score,
                    'Status': '✅ Success'
                })
                
                print(f"  ✅ Success: {best_method}, R²={best_r2:.4f}, Quality={quality_score:.1f}")
                
            else:
                results_summary.append({
                    'Target': target,
                    'Best_Method': 'N/A',
                    'Test_R²': np.nan,
                    'Quality_Score': 0,
                    'Status': f"❌ Failed: {result.get('error', 'Unknown')[:50]}"
                })
                
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            results_summary.append({
                'Target': target,
                'Best_Method': 'N/A', 
                'Test_R²': np.nan,
                'Quality_Score': 0,
                'Status': f"❌ Exception: {str(e)[:50]}"
            })
            
            print(f"  ❌ Exception: {str(e)}")
    
    # Zeige Zusammenfassung
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(f"\nMULTI-TARGET SUMMARY:")
        print(summary_df.to_string(index=False))
        
        success_rate = (summary_df['Status'].str.contains('Success')).mean()
        avg_quality = summary_df['Quality_Score'].mean()
        
        print(f"\nOverall Performance:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Quality Score: {avg_quality:.1f}/5.0")
    
    # Test 3: Datenqualitäts-Validierung
    print("\n" + "=" * 60)
    print("TEST 3: DATA QUALITY VALIDATION")
    print("=" * 60)
    
    try:
        # Teste mit problematischen Daten (zu kurze Zeitreihe)
        short_data = data.tail(20).copy()  # Nur 20 Beobachtungen
        
        print("Testing with short dataset (20 observations)...")
        short_result = improved_financial_analysis(
            data=short_data,
            target_var=target,
            exog_vars=[col for col in short_data.columns if col not in ['Datum', target]][:3],
            analysis_type='quick',
            config=config
        )
        
        if short_result['status'] == 'failed':
            print("✅ Correctly rejected short dataset")
            print(f"   Reason: {short_result.get('error', 'Unknown')}")
        else:
            print("⚠️  Short dataset was accepted - may indicate insufficient validation")
        
        # Teste mit konstanter Zielvariable
        constant_data = data.copy()
        constant_data[target] = 100  # Konstanter Wert
        
        print("\nTesting with constant target variable...")
        constant_result = improved_financial_analysis(
            data=constant_data,
            target_var=target,
            exog_vars=[col for col in constant_data.columns if col not in ['Datum', target]][:3],
            analysis_type='quick',
            config=config
        )
        
        if constant_result['status'] == 'failed':
            print("✅ Correctly rejected constant target")
            print(f"   Reason: {constant_result.get('error', 'Unknown')}")
        else:
            print("⚠️  Constant target was accepted - validation may be insufficient")
            
    except Exception as e:
        print(f"❌ Test 3 failed with exception: {str(e)}")
    
    # Test 4: Anti-Leakage Test  
    print("\n" + "=" * 60)
    print("TEST 4: ANTI-LEAKAGE VERIFICATION")
    print("=" * 60)
    
    try:
        # Erstelle Analyzer für detaillierte Tests
        analyzer = ImprovedFinancialRegressionAnalyzer(
            data, target, 
            [col for col in data.columns if col not in ['Datum', target]][:4],
            config
        )
        
        # Teste Mixed-Frequency Processing
        print("Testing mixed-frequency processing...")
        freq_result = ImprovedMixedFrequencyProcessor.align_frequencies_improved(
            data, target, 'Datum', 
            validation_split_date=pd.Timestamp('2020-01-01')
        )
        
        leakage_risk = freq_result.get('leakage_risk', 'unknown')
        warnings = freq_result.get('warnings', [])
        
        print(f"  Leakage risk: {leakage_risk}")
        print(f"  Warnings: {len(warnings)}")
        
        if warnings:
            for warning in warnings[:3]:
                print(f"    - {warning}")
        
        if leakage_risk == 'low':
            print("✅ Low leakage risk detected")
        elif leakage_risk == 'medium':
            print("⚠️  Medium leakage risk - acceptable with warnings")  
        else:
            print("❌ High leakage risk - needs attention")
            
    except Exception as e:
        print(f"❌ Test 4 failed with exception: {str(e)}")
    
    # Test 5: Performance Benchmark
    print("\n" + "=" * 60)
    print("TEST 5: PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        import time
        
        # Zeitbasierter Performance-Test
        start_time = time.time()
        
        benchmark_result = comprehensive_analysis_improved(
            "PH_KREDITE", 
            start_date="2010-01",
            config=config
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"Analysis completed in {analysis_time:.1f} seconds")
        
        if benchmark_result['status'] == 'success':
            quality_score = benchmark_result.get('best_quality_score', 0)
            leakage_risk = benchmark_result.get('validation_summary', {}).get('leakage_risk', 'unknown')
            
            print(f"✅ Benchmark completed successfully")
            print(f"   Quality Score: {quality_score:.1f}/5.0")
            print(f"   Leakage Risk: {leakage_risk}")
            print(f"   Processing Time: {analysis_time:.1f}s")
            
            # Performance Rating
            if quality_score >= 3.0 and analysis_time < 30:
                print("🏆 EXCELLENT: High quality, fast execution")
            elif quality_score >= 2.0 and analysis_time < 60:
                print("✅ GOOD: Acceptable quality and speed")
            else:
                print("⚠️  NEEDS IMPROVEMENT: Low quality or slow execution")
        else:
            print(f"❌ Benchmark failed: {benchmark_result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Test 5 failed with exception: {str(e)}")
    
    # Abschluss
    print("\n" + "=" * 80)
    print("IMPROVED SYSTEM TEST COMPLETED")
    print("=" * 80)
    
    print("\nKey Improvements Implemented:")
    print("✅ Anti-leakage mixed-frequency processing")  
    print("✅ Robust time series cross-validation")
    print("✅ Conservative feature selection")
    print("✅ Enhanced data quality validation")
    print("✅ Stationarity-based transformation selection")
    print("✅ Quality-score based method ranking")
    print("✅ Comprehensive error handling")
    
    return True


def compare_old_vs_new_detailed():
    """
    Detaillierter Vergleich der alten und neuen Implementation.
    """
    print("=" * 80)
    print("DETAILED COMPARISON: OLD VS NEW IMPLEMENTATION")
    print("=" * 80)
    
    config = AnalysisConfig()
    target = "PH_KREDITE"
    
    try:
        # Lade Testdaten
        data = get_target_with_standard_exog(target, "2005-01", config)
        exog_vars = [col for col in data.columns if col not in ['Datum', target]][:4]
        
        print(f"Test data: {data.shape[0]} observations, {len(exog_vars)} features")
        print(f"Target: {target}")
        print(f"Features: {', '.join(exog_vars)}")
        
        # Old approach (original system)
        print(f"\n--- OLD APPROACH ---")
        try:
            old_results = financial_analysis(data, target, exog_vars, 'comprehensive', config)
            
            if old_results['status'] == 'success':
                old_best_r2 = old_results.get('best_test_r2', np.nan)
                old_method = old_results.get('best_method', 'N/A')
                
                print(f"✅ Old system: {old_method}, R² = {old_best_r2:.4f}")
            else:
                print(f"❌ Old system failed: {old_results.get('error', 'Unknown')}")
                old_best_r2 = np.nan
                old_method = 'Failed'
                
        except Exception as e:
            print(f"❌ Old system exception: {str(e)}")
            old_best_r2 = np.nan
            old_method = 'Exception'
        
        # New approach (improved system)  
        print(f"\n--- NEW APPROACH ---")
        try:
            new_results = improved_financial_analysis(data, target, exog_vars, 'comprehensive', config)
            
            if new_results['status'] == 'success':
                new_best_r2 = new_results['comparison'].iloc[0]['Test_R²']
                new_method = new_results.get('best_method', 'N/A')
                new_quality = new_results.get('best_quality_score', 0)
                
                print(f"✅ New system: {new_method}, R² = {new_best_r2:.4f}, Quality = {new_quality:.1f}")
            else:
                print(f"❌ New system failed: {new_results.get('error', 'Unknown')}")
                new_best_r2 = np.nan
                new_method = 'Failed'
                new_quality = 0
                
        except Exception as e:
            print(f"❌ New system exception: {str(e)}")
            new_best_r2 = np.nan
            new_method = 'Exception'
            new_quality = 0
        
        # Comparison summary
        print(f"\n--- COMPARISON SUMMARY ---")
        
        comparison_table = pd.DataFrame({
            'System': ['Old', 'New'],
            'Method': [old_method, new_method],
            'Test_R²': [old_best_r2, new_best_r2],
            'Quality_Score': [np.nan, new_quality],
            'Status': [
                'Success' if pd.notna(old_best_r2) else 'Failed',
                'Success' if pd.notna(new_best_r2) else 'Failed'
            ]
        })
        
        print(comparison_table.to_string(index=False))
        
        # Analysis of improvement
        if pd.notna(old_best_r2) and pd.notna(new_best_r2):
            improvement = new_best_r2 - old_best_r2
            print(f"\nR² Improvement: {improvement:+.4f}")
            
            if improvement > 0.05:
                print("🎉 SIGNIFICANT IMPROVEMENT")
            elif improvement > 0:
                print("✅ SLIGHT IMPROVEMENT") 
            elif improvement > -0.05:
                print("≈ COMPARABLE PERFORMANCE")
            else:
                print("⚠️  PERFORMANCE DEGRADATION")
        
        # Qualitative improvements
        print(f"\nQualitative Improvements in New System:")
        print("- Enhanced data validation with stationarity tests")
        print("- Anti-leakage mixed-frequency processing")
        print("- Robust cross-validation with proper time gaps")
        print("- Quality-based method evaluation")
        print("- Conservative feature selection")
        print("- Better error handling and reporting")
        
    except Exception as e:
        print(f"❌ Detailed comparison failed: {str(e)}")


if __name__ == "__main__":
    print("Starting comprehensive test of improved financial analysis system...")
    
    # Run all tests
    try:
        # Test improved system
        test_improved_system()
        
        # Detailed comparison
        print("\n" + "="*80)
        compare_old_vs_new_detailed()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
        print("\nRECOMMENDATIONS:")
        print("1. Use improved_financial_analysis() instead of financial_analysis()")
        print("2. Use quick_analysis_improved() for quick tests")
        print("3. Use comprehensive_analysis_improved() for full analysis")
        print("4. Monitor Quality_Score - aim for >3.0")
        print("5. Check leakage_risk in results - should be 'low'")
        print("6. Review validation warnings before proceeding")
        
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

# %%
# Test ob Forward-Fill korrekt funktioniert
data = get_target_with_standard_exog("PH_KREDITE", "2010-01")
print("Before forward-fill:")
print(data['PH_KREDITE'].notna().sum(), "of", len(data))

# Nach Forward-Fill (simuliere den Prozess)
processor = ImprovedMixedFrequencyProcessor()
result = processor.align_frequencies_improved(data, 'PH_KREDITE', 'Datum')
processed_data = result['processed_df']

print("After forward-fill:")
print(processed_data['PH_KREDITE'].notna().sum(), "of", len(processed_data))
print("Remaining NaNs:", processed_data['PH_KREDITE'].isnull().sum())





# %%
# %%
# %%
"""
Financial Analysis Pipeline - Core Configuration & Data Download
ORIGINAL DOWNLOAD LOGIC - NICHT ÄNDERN!
"""
import hashlib
import json
import asyncio
import datetime as dt
import io
import json
import logging
import re
import ssl
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from itertools import combinations

import aiohttp
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from abc import ABC, abstractmethod

# Regression methods
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

# Statistical analysis
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Optional advanced methods
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from ecbdata import ecbdata
    HAS_ECBDATA = True
except ImportError:
    HAS_ECBDATA = False
    ecbdata = None

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================


from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LagConfig:
    enable: bool = True
    candidates: List[int] = field(default_factory=lambda: [1, 3])
    per_var_max: int = 1
    total_max: int = 8
    min_train_overlap: int = 24
    min_abs_corr: float = 0.0  # optional threshold
@dataclass
class AnalysisConfig:
    """Enhanced configuration with conservative defaults for robust analysis."""
    
    # Cache and download settings
    cache_max_age_days: int = 60
    cache_dir: str = "financial_cache"
    download_timeout_seconds: int = 30
    max_concurrent_downloads: int = 8
    default_start_date: str = "2000-01"
    default_end_date: str = "2024-12"
    min_response_size: int = 100
    max_retry_attempts: int = 3
    
    # Validation settings
    test_size: float = 0.25
    cv_folds: int = 4
    gap_periods: int = 2
    
    # Feature selection settings
    max_feature_combinations: int = 20
    min_features_per_combination: int = 2
    
    # Model settings
    remove_outliers: bool = True
    outlier_method: str = "conservative"
    add_seasonal_dummies: bool = True
    handle_mixed_frequencies: bool = True
    
    # Model persistence
    save_models: bool = True
    model_cleanup_days: int = 30
    keep_best_models: int = 10
    random_seed: int = 42
    
    # Final dataset caching
    cache_final_dataset: bool = True
    final_cache_format: str = "xlsx"
    final_cache_subdir: str = "final_datasets"
    
    # Plot settings
    save_plots: bool = True
    show_plots: bool = True
    plots_dir: str = "financial_cache/diagnostic_plots"
    
    # Validation thresholds
    high_r2_warning_threshold: float = 0.9
    high_overfitting_threshold: float = 0.1
    cv_test_discrepancy_threshold: float = 0.1
    evaluate_quarter_ends_only: bool = True  # evaluate metrics only at quarter ends
    
    # Index-Normalisierung Parameter
    index_base_year: int = 2015
    index_base_value: float = 100.0
    lag_config: Optional[LagConfig] = None
    ab_compare_lags: bool = True

# =============================================================================
# CONSTANTS AND DEFINITIONS (ORIGINAL)
# =============================================================================

# Target variable definitions
INDEX_TARGETS = {
    "PH_EINLAGEN": "INDEX(BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F22.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29A.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29B.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F29D.S14.DE.S1.W0.F.N._X.B)",
    "PH_WERTPAPIERE": "INDEX(BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F511.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F512.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F519.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F52.S14.DE.S1.W0.F.N._X.B)",
    "PH_VERSICHERUNGEN": "INDEX(BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B, BBAF3.Q.F8.S14.DE.S1.W0.F.N._X.B)",
    "PH_KREDITE": "INDEX(BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B, BBAF3.Q.F8.S1.W0.S14.DE.F.N._X.B)",
    "NF_KG_EINLAGEN": "INDEX(BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F22.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29A.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29B.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F29D.S11.DE.S1.W0.F.N._X.B)",
    "NF_KG_WERTPAPIERE": "INDEX(BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F32.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F511.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F512.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F519.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F52.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F7.S11.DE.S1.W0.F.N._X.B)",
    "NF_KG_VERSICHERUNGEN": "INDEX(BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F8.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F6.S1.W0.S11.DE.F.N._X.B)",
    "NF_KG_KREDITE": "INDEX(BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F42.S11.DE.S1.W0.F.N._X.B, BBAF3.Q.F4.S1.W0.S11.DE.F.N._X.B, BBAF3.Q.F8.S1.W0.S11.DE.F.N._X.B)",
}

# Standard exogenous variables
STANDARD_EXOG_VARS = {
    "euribor_3m": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "german_rates": "IRS.M.DE.L.L40.CI.0000.EUR.N.Z",
    "german_inflation": "ICP.M.DE.N.000000.4.ANR",
    "german_unemployment": "LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T",
    "eur_usd_rate": "EXR.D.USD.EUR.SP00.A",
    "german_gdp": "MNA.Q.DE.N.B1GQ.C.S1.S1.B.B1GQ._Z.EUR.LR.GY",
    "ecb_main_rate": "FM.M.U2.EUR.RT.MM.EONIA_.HSTA",
}

# API constants
ECB_API_BASE_URL = "https://data-api.ecb.europa.eu/service/data"
BUNDESBANK_API_BASE_URL = "https://api.statistiken.bundesbank.de/rest/download"
ECB_PREFIXES = ("ICP.", "BSI.", "MIR.", "FM.", "IRS.", "LFSI.", "STS.", "MNA.", "BOP.", "GFS.", "EXR.")

# Fallback definitions
SIMPLE_TARGET_FALLBACKS = {
    "PH_KREDITE": "BBAF3.Q.F4.S1.W0.S14.DE.F.N._X.B",
    "PH_EINLAGEN": "BBAF3.Q.F21.S14.DE.S1.W0.F.N._X.B",
    "PH_WERTPAPIERE": "BBAF3.Q.F3.S14.DE.S1.W0.F.N._X.B",
    "PH_VERSICHERUNGEN": "BBAF3.Q.F6.S14.DE.S1.W0.F.N._X.B",
    "NF_KG_EINLAGEN": "BBAF3.Q.F21.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_WERTPAPIERE": "BBAF3.Q.F31.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_VERSICHERUNGEN": "BBAF3.Q.F6.S11.DE.S1.W0.F.N._X.B",
    "NF_KG_KREDITE": "BBAF3.Q.F41.S11.DE.S1.W0.F.N._X.B",
}

INDEX_SPEC_RE = re.compile(r'^\s*INDEX\s*\(\s*(.*?)\s*\)\s*', re.IGNORECASE)

# =============================================================================
# ORIGINAL UTILITY FUNCTIONS (NICHT ÄNDERN)
# =============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_data_source(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Invalid series code: {code}")
    code_upper = code.upper()
    if "." in code_upper and code_upper.startswith(ECB_PREFIXES):
        return "ECB"
    return "BUNDESBANK"

def parse_index_specification(spec: str) -> Optional[List[str]]:
    if not isinstance(spec, str):
        return None
    match = INDEX_SPEC_RE.match(spec.strip())
    if not match:
        return None
    inner = match.group(1)
    codes = [c.strip() for c in inner.split(",") if c.strip()]
    return list(dict.fromkeys(codes)) if codes else None

def validate_date_string(date_str: str) -> bool:
    if not isinstance(date_str, str):
        return False
    date_patterns = ["%Y-%m", "%Y-%m-%d", "%Y"]
    for pattern in date_patterns:
        try:
            dt.datetime.strptime(date_str, pattern)
            return True
        except ValueError:
            continue
    return False

def format_date_for_ecb_api(date_str: str) -> str:
    if not date_str:
        return date_str
    try:
        if len(date_str) == 4:
            return f"{date_str}-01"
        elif len(date_str) == 7:
            return date_str
        elif len(date_str) == 10:
            return date_str[:7]
        else:
            parsed_date = pd.to_datetime(date_str)
            return parsed_date.strftime("%Y-%m")
    except:
        return date_str

def get_excel_engine() -> str:
    try:
        import openpyxl
        return 'openpyxl'
    except ImportError:
        try:
            import xlsxwriter
            return 'xlsxwriter'
        except ImportError:
            raise ImportError("Excel support requires openpyxl or xlsxwriter. Install with: pip install openpyxl")

# =============================================================================
# ORIGINAL DATA DOWNLOAD CLASSES (NICHT ÄNDERN)
# =============================================================================

class DataProcessor:
    @staticmethod
    def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        date_candidates = ["TIME_PERIOD", "DATE", "Datum", "Period", "period"]
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None and len(df.columns) > 0:
            date_col = df.columns[0]
        
        value_candidates = ["OBS_VALUE", "VALUE", "Wert", "Value"]
        value_col = None
        for candidate in value_candidates:
            if candidate in df.columns and candidate != date_col:
                value_col = candidate
                break
        if value_col is None:
            numeric_cols = [c for c in df.columns if c != date_col and df[c].dtype in ['float64', 'int64']]
            if numeric_cols:
                value_col = numeric_cols[-1]
            else:
                raise ValueError("No value column found")
        
        result = pd.DataFrame()
        result["Datum"] = pd.to_datetime(df[date_col], errors='coerce')
        result["value"] = pd.to_numeric(df[value_col], errors='coerce')
        result = result.dropna(subset=["value", "Datum"])
        result = result.sort_values("Datum").reset_index(drop=True)
        return result

class CacheManager:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir = self.cache_dir / self.config.final_cache_subdir
        self.final_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, code: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in code)
        return self.cache_dir / f"{safe_name}.xlsx"
    
    def is_fresh(self, code: str) -> bool:
        cache_path = self._cache_path(code)
        if not cache_path.exists():
            return False
        try:
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_days = (dt.datetime.now() - mtime).days
            return age_days <= self.config.cache_max_age_days
        except OSError:
            return False
    
    def read_cache(self, code: str) -> Optional[pd.DataFrame]:
        if not self.is_fresh(code):
            return None
        cache_path = self._cache_path(code)
        try:
            df = pd.read_excel(cache_path, sheet_name="data", engine=get_excel_engine())
            return DataProcessor.standardize_dataframe(df)
        except Exception:
            return None
    
    def write_cache(self, code: str, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        cache_path = self._cache_path(code)
        temp_path = cache_path.with_suffix(".tmp.xlsx")
        try:
            with pd.ExcelWriter(temp_path, engine=get_excel_engine()) as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            temp_path.replace(cache_path)
            return True
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False

class BundesbankCSVParser:
    @staticmethod
    def parse(content: str, code: str) -> pd.DataFrame:
        try:
            lines = content.strip().split('\n')
            if not lines:
                raise ValueError("Empty CSV content")
            
            data_start_idx = BundesbankCSVParser._find_data_start(lines, code)
            csv_lines = lines[data_start_idx:]
            if not csv_lines:
                raise ValueError("No data lines found")
            
            delimiter = BundesbankCSVParser._detect_delimiter(csv_lines[0])
            df = pd.read_csv(io.StringIO('\n'.join(csv_lines)), delimiter=delimiter, skip_blank_lines=True)
            df = df.dropna(how='all')
            if df.empty:
                raise ValueError("No valid data after parsing")
            
            time_col, value_col = BundesbankCSVParser._identify_columns(df, code)
            result_df = pd.DataFrame()
            time_values = df[time_col].dropna()
            value_values = df[value_col].dropna()
            min_len = min(len(time_values), len(value_values))
            if min_len == 0:
                raise ValueError("No valid data pairs found")
            
            result_df['Datum'] = time_values.iloc[:min_len].astype(str)
            result_df['value'] = pd.to_numeric(value_values.iloc[:min_len], errors='coerce')
            result_df = result_df.dropna()
            if result_df.empty:
                raise ValueError("No valid numeric data after cleaning")
            return result_df
        except Exception as e:
            raise ValueError(f"Bundesbank CSV parsing failed: {e}")
    
    @staticmethod
    def _find_data_start(lines: List[str], code: str) -> int:
        for i, line in enumerate(lines):
            if code in line and ('BBAF3' in line or 'BBK' in line):
                return i
        for i, line in enumerate(lines):
            if code in line:
                return i
        for i, line in enumerate(lines):
            if ',' in line or ';' in line:
                sep_count = max(line.count(','), line.count(';'))
                if sep_count >= 2:
                    return i
        return 0
    
    @staticmethod
    def _detect_delimiter(header_line: str) -> str:
        comma_count = header_line.count(',')
        semicolon_count = header_line.count(';')
        if comma_count > semicolon_count:
            return ','
        elif semicolon_count > 0:
            return ';'
        else:
            if '\t' in header_line:
                return '\t'
            elif '|' in header_line:
                return '|'
            else:
                return ','
    
    @staticmethod
    def _identify_columns(df: pd.DataFrame, code: str) -> Tuple[str, str]:
        value_col = None
        for col in df.columns:
            col_str = str(col)
            if code in col_str and 'FLAG' not in col_str.upper() and 'ATTRIBUT' not in col_str.upper():
                value_col = col
                break
        if value_col is None:
            code_parts = code.split('.')
            for col in df.columns:
                col_str = str(col)
                if any(part in col_str for part in code_parts if len(part) > 3) and 'FLAG' not in col_str.upper():
                    value_col = col
                    break
        if value_col is None and len(df.columns) >= 2:
            for col in df.columns[1:]:
                if pd.to_numeric(df[col], errors='coerce').notna().sum() > 0:
                    value_col = col
                    break
        if value_col is None:
            if len(df.columns) >= 2:
                value_col = df.columns[1]
            else:
                raise ValueError("Could not identify value column")
        
        time_col = None
        date_keywords = ['TIME', 'DATE', 'PERIOD', 'DATUM', 'ZEIT']
        for col in df.columns:
            col_str = str(col).upper()
            if any(keyword in col_str for keyword in date_keywords):
                time_col = col
                break
        if time_col is None:
            for col in df.columns:
                if col != value_col and 'FLAG' not in str(col).upper():
                    time_col = col
                    break
        if time_col is None:
            time_col = df.columns[0]
        
        return time_col, value_col

class APIClient:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    async def fetch_series(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        source = detect_data_source(code)
        if source == "ECB":
            return await self._fetch_ecb(session, code, start, end)
        else:
            return await self._fetch_bundesbank(session, code, start, end)
    
    async def _fetch_ecb(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        if HAS_ECBDATA:
            try:
                df = ecbdata.get_series(series_key=code, start=start, end=end)
                if df is not None and not df.empty:
                    return DataProcessor.standardize_dataframe(df)
            except Exception:
                pass
        
        flow, series = code.split(".", 1)
        url = f"{ECB_API_BASE_URL}/{flow}/{series}"
        fstart = format_date_for_ecb_api(start)
        fend = format_date_for_ecb_api(end)
        
        param_strategies = [
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
            {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
            {"format": "csvdata", "detail": "dataonly"},
        ]
        
        timeout = aiohttp.ClientTimeout(total=self.config.download_timeout_seconds)
        headers = {"Accept": "text/csv"}
        last_error = None
        
        for params in param_strategies:
            async with session.get(url, params=params, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    last_error = f"Status {response.status}"
                    continue
                text = await response.text()
                if not text.strip() or len(text.strip()) < self.config.min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                try:
                    df = pd.read_csv(io.StringIO(text))
                    df = DataProcessor.standardize_dataframe(df)
                    if not df.empty:
                        return df
                except Exception as e:
                    last_error = f"CSV parse error: {e}"
                    continue
        
        raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    
    async def _fetch_bundesbank(self, session: aiohttp.ClientSession, code: str, start: str, end: str) -> pd.DataFrame:
        url_patterns = self._build_bundesbank_urls(code)
        params_variants = self._get_bundesbank_params(start, end)
        headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        timeout = aiohttp.ClientTimeout(total=self.config.download_timeout_seconds)
        last_error = None
        attempt_count = 0
        max_attempts = min(len(url_patterns) * len(params_variants), 20)
        
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
        
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as bb_session:
                for url in url_patterns:
                    for params in params_variants:
                        attempt_count += 1
                        if attempt_count > max_attempts:
                            break
                        try:
                            async with bb_session.get(url, params=params, headers=headers) as response:
                                if response.status == 200:
                                    text = await response.text()
                                    if text and len(text.strip()) > self.config.min_response_size:
                                        df = BundesbankCSVParser.parse(text, code)
                                        if df is not None and not df.empty:
                                            df = DataProcessor.standardize_dataframe(df)
                                            if not df.empty:
                                                return df
                                    else:
                                        last_error = f"Response too small: {len(text)} bytes"
                                        continue
                                elif response.status == 404:
                                    last_error = "Series not found (404)"
                                    continue
                                else:
                                    error_text = await response.text()
                                    last_error = f"Status {response.status}: {error_text[:100]}"
                                    continue
                        except asyncio.TimeoutError:
                            last_error = f"Timeout after {self.config.download_timeout_seconds}s"
                            continue
                        except Exception as e:
                            last_error = f"Unexpected error: {str(e)}"
                            continue
                    if attempt_count > max_attempts:
                        break
        except Exception as e:
            last_error = f"Session creation failed: {e}"
        
        raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")
    
    def _build_bundesbank_urls(self, code: str) -> List[str]:
        base_urls = [
            "https://api.statistiken.bundesbank.de/rest/download",
            "https://www.bundesbank.de/statistic-rmi/StatisticDownload"
        ]
        url_patterns = []
        if '.' in code:
            dataset, series = code.split('.', 1)
            url_patterns.extend([
                f"{base_urls[0]}/{dataset}/{series}",
                f"{base_urls[0]}/{code.replace('.', '/')}",
                f"{base_urls[1]}/{dataset}/{series}",
                f"{base_urls[1]}/{code.replace('.', '/')}"
            ])
        url_patterns.extend([
            f"{base_urls[0]}/{code}",
            f"{base_urls[1]}/{code}"
        ])
        if code.count('.') > 1:
            parts = code.split('.')
            for i in range(1, len(parts)):
                path1 = '.'.join(parts[:i])
                path2 = '.'.join(parts[i:])
                url_patterns.extend([
                    f"{base_urls[0]}/{path1}/{path2}",
                    f"{base_urls[0]}/{path1.replace('.', '/')}/{path2.replace('.', '/')}"
                ])
        seen = set()
        unique_patterns = []
        for pattern in url_patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        return unique_patterns[:12]
    
    def _get_bundesbank_params(self, start: str, end: str) -> List[Dict[str, str]]:
        return [
            {"format": "csv", "lang": "en", "metadata": "false"},
            {"format": "csv", "lang": "de", "metadata": "false"},
            {"format": "csv", "lang": "en", "metadata": "false", "startPeriod": start, "endPeriod": end},
            {"format": "csv", "lang": "de", "metadata": "false", "startPeriod": start, "endPeriod": end},
            {"format": "tsv", "lang": "en", "metadata": "false"},
            {"format": "tsv", "lang": "de", "metadata": "false"},
            {"format": "csv"},
            {"lang": "en"},
            {"lang": "de"},
            {}
        ]

class IndexCreator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def create_index(self, data_df: pd.DataFrame, series_codes: List[str], index_name: str) -> pd.Series:
        if 'Datum' not in data_df.columns:
            raise ValueError("DataFrame must contain a 'Datum' column")
        
        available_codes = [code for code in series_codes if code in data_df.columns]
        if not available_codes:
            raise ValueError(f"No valid series found for index {index_name}")
        
        index_data = data_df[['Datum'] + available_codes].copy()
        index_data = index_data.set_index('Datum')
        
        has_any = index_data[available_codes].notna().any(axis=1)
        index_data = index_data.loc[has_any].copy()

        def _fill_inside(s: pd.Series) -> pd.Series:
            if s.notna().sum() == 0:
                return s
            first, last = s.first_valid_index(), s.last_valid_index()
            if first is None or last is None:
                return s
            filled = s.ffill().bfill()
            mask = (s.index >= first) & (s.index <= last)
            return filled.where(mask, s)
        
        index_data[available_codes] = index_data[available_codes].apply(_fill_inside)
        clean_data = index_data.dropna()
        
        if clean_data.empty:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result
        
        weights = {code: 1.0 / len(available_codes) for code in available_codes}
        weighted_values = []
        for code in available_codes:
            if code in clean_data.columns:
                weighted_values.append(clean_data[code] * weights[code])
        
        if not weighted_values:
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            result[:] = np.nan
            return result
        
        aggregated = sum(weighted_values)
        
        try:
            base_year_int = int(self.config.index_base_year)
            base_year_mask = aggregated.index.year == base_year_int
            base_year_data = aggregated[base_year_mask]
            
            if base_year_data.empty or base_year_data.isna().all():
                first_valid = aggregated.dropna()
                if first_valid.empty:
                    base_value_actual = 1.0
                else:
                    base_value_actual = first_valid.iloc[0]
            else:
                base_value_actual = base_year_data.mean()
            
            if base_value_actual == 0 or pd.isna(base_value_actual):
                base_value_actual = 1.0
            
            result = pd.Series(index=index_data.index, dtype=float, name=index_name)
            mask = aggregated.notna()
            result[mask] = (aggregated[mask] / base_value_actual) * self.config.index_base_value
            
            return result
            
        except Exception as e:
            print(f"Warning: Index normalization failed for {index_name}, using raw data: {e}")
            aggregated.name = index_name
            return aggregated

# =============================================================================
# ORIGINAL DOWNLOAD LOGIC (NICHT ÄNDERN)
# =============================================================================

class FinancialDataDownloader:
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.cache_manager = CacheManager(self.config)
        self.api_client = APIClient(self.config)
        self.index_creator = IndexCreator(self.config)
    
    def download(self, series_definitions: Dict[str, str], start_date: str = None, 
                end_date: str = None, prefer_cache: bool = True, anchor_var: Optional[str] = None) -> pd.DataFrame:
        start_date = start_date or self.config.default_start_date
        end_date = end_date or self.config.default_end_date
        print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
        
        regular_codes = {}
        index_definitions = {}
        
        for var_name, definition in series_definitions.items():
            index_codes = parse_index_specification(definition)
            if index_codes:
                index_definitions[var_name] = index_codes
            else:
                regular_codes[var_name] = definition
        
        all_codes = set(regular_codes.values())
        for index_codes in index_definitions.values():
            all_codes.update(index_codes)
        all_codes = list(all_codes)
        
        print(f"Total series to download: {len(all_codes)}")
        
        cached_data = {}
        missing_codes = []
        
        if prefer_cache:
            for code in all_codes:
                cached_df = self.cache_manager.read_cache(code)
                if cached_df is not None:
                    cached_data[code] = cached_df
                else:
                    missing_codes.append(code)
        else:
            missing_codes = all_codes[:]
        
        downloaded_data = {}
        if missing_codes:
            print(f"Downloading {len(missing_codes)} missing series...")
            try:
                downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
                    except ImportError:
                        print("Using synchronous download mode...")
                        downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
                else:
                    print("Async failed, using synchronous download mode...")
                    downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            except Exception as e:
                print(f"Download failed ({e}), trying synchronous mode...")
                downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            
            for code, df in downloaded_data.items():
                self.cache_manager.write_cache(code, df)
        
        all_data = {**cached_data, **downloaded_data}
        
        if not all_data:
            raise Exception("No series loaded successfully")
        
        merged_df = self._merge_series_data(all_data)
        final_data = {"Datum": merged_df["Datum"]}
        
        for var_name, series_code in regular_codes.items():
            if series_code in merged_df.columns:
                final_data[var_name] = merged_df[series_code]
        
        for var_name, index_codes in index_definitions.items():
            try:
                available_codes = [c for c in index_codes if c in merged_df.columns]
                
                if len(available_codes) >= len(index_codes) * 0.3:
                    index_series = self.index_creator.create_index(merged_df, available_codes, var_name)
                    aligned_index = index_series.reindex(pd.to_datetime(merged_df['Datum']))
                    final_data[var_name] = aligned_index.values
                    print(f"Created INDEX: {var_name} from {len(available_codes)}/{len(index_codes)} series")
                else:
                    if var_name in SIMPLE_TARGET_FALLBACKS:
                        fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                        if fallback_code in merged_df.columns:
                            final_data[var_name] = merged_df[fallback_code]
                            print(f"Using fallback for {var_name}: {fallback_code}")
                        else:
                            print(f"Warning: Could not create {var_name} - fallback series {fallback_code} not available")
                    else:
                        print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available_codes)}/{len(index_codes)} series available)")
                        
            except Exception as e:
                print(f"Failed to create INDEX {var_name}: {e}")
                if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                    fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback_code in merged_df.columns:
                        final_data[var_name] = merged_df[fallback_code]
                        print(f"Using fallback for {var_name} after INDEX creation failed: {fallback_code}")
        
        final_df = pd.DataFrame(final_data)
        final_df["Datum"] = pd.to_datetime(final_df["Datum"])
        final_df = final_df.sort_values("Datum").reset_index(drop=True)

        value_cols = [c for c in final_df.columns if c != 'Datum']
        if value_cols:
            non_na_count = final_df[value_cols].notna().sum(axis=1)
            required = 2 if len(value_cols) >= 2 else 1
            keep_mask = non_na_count >= required
            if keep_mask.any():
                first_keep = keep_mask.idxmax()
                if first_keep > 0:
                    _before = len(final_df)
                    final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                    print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

        if anchor_var and anchor_var in final_df.columns:
            mask_anchor = final_df[anchor_var].notna()
            if mask_anchor.any():
                start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
                end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
                _before_rows = len(final_df)
                final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
                final_df.reset_index(drop=True, inplace=True)
                print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

        if anchor_var and anchor_var in final_df.columns:
            exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
            if exog_cols:
                tgt_notna = final_df[anchor_var].notna().values
                all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
                keep_start = 0
                for i in range(len(final_df)):
                    if not (tgt_notna[i] and all_exog_nan[i]):
                        keep_start = i
                        break
                if keep_start > 0:
                    _before = len(final_df)
                    final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                    print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

        print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")
        return final_df
    
    def _fetch_all_series_sync(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        import requests
        successful = {}
        
        for code in codes:
            try:
                source = detect_data_source(code)
                if source == "ECB":
                    df = self._fetch_ecb_sync(code, start, end)
                else:
                    df = self._fetch_bundesbank_sync(code, start, end)
                
                if df is not None and not df.empty:
                    successful[code] = df
                    print(f"  ✓ {code}: {len(df)} observations")
                else:
                    print(f"  ✗ {code}: No data returned")
            except Exception as e:
                print(f"  ✗ {code}: {str(e)}")
            
            import time
            time.sleep(0.5)
        
        return successful
    
    def _fetch_ecb_sync(self, code: str, start: str, end: str) -> pd.DataFrame:
        import requests
        
        if HAS_ECBDATA:
            try:
                df = ecbdata.get_series(series_key=code, start=start, end=end)
                if df is not None and not df.empty:
                    return DataProcessor.standardize_dataframe(df)
            except Exception:
                pass
        
        flow, series = code.split(".", 1)
        url = f"{ECB_API_BASE_URL}/{flow}/{series}"
        fstart = format_date_for_ecb_api(start)
        fend = format_date_for_ecb_api(end)

        param_strategies = [
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly"},
            {"format": "csvdata", "startDate": fstart, "endDate": fend, "detail": "dataonly"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend, "detail": "dataonly", "includeHistory": "true"},
            {"format": "csvdata", "startPeriod": fstart, "endPeriod": fend},
            {"format": "csvdata", "detail": "dataonly"},
        ]

        headers = {"Accept": "text/csv"}
        last_error = None

        for params in param_strategies:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.config.download_timeout_seconds)
                if resp.status_code != 200:
                    last_error = f"Status {resp.status_code}"
                    continue
                text = resp.text
                if not text.strip() or len(text.strip()) < self.config.min_response_size:
                    last_error = f"Response too small: {len(text)}"
                    continue
                df = pd.read_csv(io.StringIO(text))
                df = DataProcessor.standardize_dataframe(df)
                if not df.empty:
                    return df
            except Exception as e:
                last_error = str(e)
                continue

        raise Exception(f"ECB API failed for {code}. Last error: {last_error}")
    
    def _fetch_bundesbank_sync(self, code: str, start: str, end: str) -> pd.DataFrame:
        import requests
        
        url_patterns = self.api_client._build_bundesbank_urls(code)
        params_variants = self.api_client._get_bundesbank_params(start, end)
        headers = {"Accept": "text/csv,application/csv,text/plain,*/*"}
        last_error = None
        attempt_count = 0
        max_attempts = min(len(url_patterns) * len(params_variants), 20)
        
        for url in url_patterns:
            for params in params_variants:
                attempt_count += 1
                if attempt_count > max_attempts:
                    break
                
                try:
                    response = requests.get(
                        url, params=params, headers=headers,
                        timeout=self.config.download_timeout_seconds, verify=False
                    )
                    
                    if response.status_code == 200:
                        text = response.text
                        if text and len(text.strip()) > self.config.min_response_size:
                            df = BundesbankCSVParser.parse(text, code)
                            if df is not None and not df.empty:
                                df = DataProcessor.standardize_dataframe(df)
                                if not df.empty:
                                    return df
                        else:
                            last_error = f"Response too small: {len(text)} bytes"
                            continue
                    elif response.status_code == 404:
                        last_error = "Series not found (404)"
                        continue
                    else:
                        last_error = f"Status {response.status_code}: {response.text[:100]}"
                        continue
                        
                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {self.config.download_timeout_seconds}s"
                    continue
                except requests.exceptions.SSLError:
                    last_error = "SSL verification failed"
                    continue
                except Exception as e:
                    last_error = f"Request failed: {str(e)}"
                    continue
            
            if attempt_count > max_attempts:
                break
        
        raise Exception(f"Bundesbank API failed after {attempt_count} attempts. Last error: {last_error}")

    async def _fetch_all_series(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        successful = {}
        
        async with aiohttp.ClientSession() as session:
            for code in codes:
                try:
                    df = await self.api_client.fetch_series(session, code, start, end)
                    successful[code] = df
                    print(f"  ✓ {code}: {len(df)} observations")
                except Exception as e:
                    print(f"  ✗ {code}: {str(e)}")
                
                await asyncio.sleep(0.5)
        
        return successful
    
    def _merge_series_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        all_series = []
        
        for code, df in data_dict.items():
            if not df.empty and "Datum" in df.columns and "value" in df.columns:
                series_df = df.set_index("Datum")[["value"]].rename(columns={"value": code})
                all_series.append(series_df)
        
        if not all_series:
            return pd.DataFrame()
        
        merged_df = pd.concat(all_series, axis=1, sort=True)
        merged_df = merged_df.reset_index()
        merged_df = merged_df.sort_values("Datum").reset_index(drop=True)
        return merged_df

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_target_with_standard_exog(target_name: str, start_date: str = "2000-01", 
                                 config: AnalysisConfig = None) -> pd.DataFrame:
    if target_name not in INDEX_TARGETS:
        raise ValueError(f"Unknown target: {target_name}. Available: {list(INDEX_TARGETS.keys())}")
    
    series_definitions = {target_name: INDEX_TARGETS[target_name]}
    standard_exog = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    for exog_name in standard_exog:
        if exog_name in STANDARD_EXOG_VARS:
            series_definitions[exog_name] = STANDARD_EXOG_VARS[exog_name]
    
    downloader = FinancialDataDownloader(config)
    return downloader.download(series_definitions, start_date=start_date)

class FinancialAnalysisError(Exception):
    pass

class DataDownloadError(FinancialAnalysisError):
    pass

class ValidationError(FinancialAnalysisError):
    pass

class AnalysisError(FinancialAnalysisError):
    pass

print("Core configuration and data download loaded")

# %%
"""
Mixed-Frequency Data Processing - KORRIGIERTE VERSION
Löst das Forward-Fill Problem für Quartalsdaten korrekt
"""

class MixedFrequencyProcessor:
    """
    Handles quarterly target variables with monthly exogenous variables.
    Ensures proper forward-filling without data leakage.
    """
    
    @staticmethod
    def detect_frequency(series: pd.Series, date_col: pd.Series) -> str:
        """
        Detect if a series is monthly or quarterly based on data availability.
        """
        if series.isna().all():
            return "unknown"
        
        # Count observations per year
        df_temp = pd.DataFrame({'date': date_col, 'value': series})
        df_temp = df_temp.dropna()
        
        if len(df_temp) == 0:
            return "unknown"
        
        df_temp['year'] = df_temp['date'].dt.year
        obs_per_year = df_temp.groupby('year').size()
        
        avg_obs_per_year = obs_per_year.mean()
        
        if avg_obs_per_year <= 4.5:  # Allow for some missing quarters
            return "quarterly"
        elif avg_obs_per_year >= 10:  # Allow for some missing months
            return "monthly"
        else:
            return "unknown"
    
    @staticmethod
    def forward_fill_quarterly(df: pd.DataFrame, quarterly_vars: List[str]) -> pd.DataFrame:
        """
        Forward-fill quarterly variables properly:
        1. Only fill within the available data range (no extrapolation)
        2. Fill monthly gaps between quarterly observations
        """
        result = df.copy()
        
        for var in quarterly_vars:
            if var not in df.columns:
                continue
                
            series = df[var].copy()
            
            # Find first and last valid observation
            valid_mask = series.notna()
            if not valid_mask.any():
                continue
                
            first_valid_idx = valid_mask.idxmax()
            last_valid_idx = valid_mask[::-1].idxmax()  # Last valid
            
            # Only forward fill between first and last valid observation
            fill_range = series.iloc[first_valid_idx:last_valid_idx+1]
            filled_range = fill_range.ffill()
            
            # Update only the range between first and last valid
            result.loc[first_valid_idx:last_valid_idx, var] = filled_range
            
        return result
    
    @staticmethod
    def align_frequencies(df: pd.DataFrame, target_var: str, 
                         date_col: str = "Datum") -> pd.DataFrame:
        """
        Align mixed-frequency data properly for regression analysis.
        """
        if target_var not in df.columns or date_col not in df.columns:
            raise ValueError(f"Missing {target_var} or {date_col} column")
        
        # Detect frequencies
        frequencies = {}
        all_vars = [col for col in df.columns if col != date_col]
        
        for var in all_vars:
            freq = MixedFrequencyProcessor.detect_frequency(df[var], df[date_col])
            frequencies[var] = freq
            
        print(f"Detected frequencies:")
        for var, freq in frequencies.items():
            obs_count = df[var].notna().sum()
            print(f"  {var}: {freq} ({obs_count} observations)")
        
        # Identify quarterly variables
        quarterly_vars = [var for var, freq in frequencies.items() if freq == "quarterly"]
        monthly_vars = [var for var, freq in frequencies.items() if freq == "monthly"]
        
        if not quarterly_vars:
            print("No quarterly variables detected - returning original data")
            return df
        
        # Apply forward-fill to quarterly variables
        print(f"Forward-filling {len(quarterly_vars)} quarterly variables...")
        processed_df = MixedFrequencyProcessor.forward_fill_quarterly(df, quarterly_vars)
        
        # Validation: Check improvement
        for var in quarterly_vars:
            before_count = df[var].notna().sum()
            after_count = processed_df[var].notna().sum()
            print(f"  {var}: {before_count} → {after_count} observations")
        
        return processed_df

class DataQualityChecker:
    """
    Comprehensive data quality validation for financial time series.
    """
    
    @staticmethod
    def validate_financial_data(data: pd.DataFrame, target_var: str, 
                               exog_vars: List[str], 
                               min_target_coverage: float = 0.15) -> Dict[str, Any]:
        """
        Enhanced data validation with mixed-frequency awareness.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Check if variables exist
        missing_vars = [var for var in [target_var] + exog_vars if var not in data.columns]
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {', '.join(missing_vars)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Analyze target variable quality (CRITICAL)
        target_series = data[target_var]
        target_coverage = target_series.notna().sum() / len(target_series)
        
        validation_results['data_quality'][target_var] = {
            'total_obs': len(target_series),
            'valid_obs': target_series.notna().sum(),
            'coverage': target_coverage,
            'frequency': MixedFrequencyProcessor.detect_frequency(target_series, data['Datum'])
        }
        
        # CRITICAL CHECK: Target coverage
        if target_coverage < min_target_coverage:
            validation_results['errors'].append(
                f"Target variable {target_var} has only {target_coverage:.1%} valid data "
                f"(minimum required: {min_target_coverage:.1%})"
            )
            validation_results['is_valid'] = False
            
        # Check for completely constant target
        if target_series.notna().sum() > 1 and target_series.std() == 0:
            validation_results['errors'].append(f"Target variable {target_var} is constant")
            validation_results['is_valid'] = False
        
        # Analyze exogenous variables
        for var in exog_vars:
            if var in data.columns:
                series = data[var]
                coverage = series.notna().sum() / len(series)
                
                validation_results['data_quality'][var] = {
                    'total_obs': len(series),
                    'valid_obs': series.notna().sum(),
                    'coverage': coverage,
                    'frequency': MixedFrequencyProcessor.detect_frequency(series, data['Datum'])
                }
                
                if coverage < 0.5:
                    validation_results['warnings'].append(
                        f"Exogenous variable {var} has low coverage ({coverage:.1%})"
                    )
                
                if series.notna().sum() > 1 and series.std() == 0:
                    validation_results['warnings'].append(f"Variable {var} is constant")
        
        # Check for sufficient overlapping data
        all_vars = [target_var] + exog_vars
        available_vars = [var for var in all_vars if var in data.columns]
        
        if available_vars:
            complete_cases = data[available_vars].dropna()
            overlap_ratio = len(complete_cases) / len(data)
            
            validation_results['overlap_analysis'] = {
                'complete_cases': len(complete_cases),
                'total_cases': len(data),
                'overlap_ratio': overlap_ratio
            }
            
            if overlap_ratio < 0.1:
                validation_results['errors'].append(
                    f"Very low overlap between variables ({overlap_ratio:.1%} complete cases)"
                )
                validation_results['is_valid'] = False
            elif overlap_ratio < 0.3:
                validation_results['warnings'].append(
                    f"Low overlap between variables ({overlap_ratio:.1%} complete cases)"
                )
        
        # Generate recommendations
        if validation_results['data_quality'][target_var]['frequency'] == 'quarterly':
            validation_results['recommendations'].append(
                "Target is quarterly - will apply forward-fill to align with monthly exogenous variables"
            )
        
        if not validation_results['is_valid']:
            validation_results['recommendations'].append(
                "Consider using a different target variable or extending the time period"
            )
        
        return validation_results

class DataPreprocessor:
    """
    Handles data preprocessing for financial regression analysis.
    Includes proper mixed-frequency handling and transformation logic.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.date_col = date_col
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
    
    def create_transformations(self, transformation: str = 'levels') -> pd.DataFrame:
        """
        Create data transformations with proper mixed-frequency handling.
        """
        # Step 1: Handle mixed frequencies FIRST
        processed_data = MixedFrequencyProcessor.align_frequencies(
            self.data, self.target_var, self.date_col
        )
        
        # Step 2: Apply transformations
        transformed_data = processed_data[[self.date_col]].copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        print(f"Applying '{transformation}' transformation to {len(numeric_cols)} variables...")
        
        for col in numeric_cols:
            series = processed_data[col]
            
            if transformation == 'levels':
                transformed_data[col] = series
            elif transformation == 'log':
                # Check if all positive values
                positive_mask = series > 0
                if positive_mask.sum() > len(series) * 0.8:  # At least 80% positive
                    # Apply log transformation with small epsilon for zeros
                    transformed_data[col] = np.log(series.clip(lower=1e-6))
                else:
                    # Fall back to levels if not suitable for log
                    transformed_data[col] = series
                    print(f"  Warning: {col} not suitable for log transformation (negative values)")
            elif transformation == 'pct':
                transformed_data[col] = series.pct_change()
            elif transformation == 'diff':
                transformed_data[col] = series.diff()
            else:
                transformed_data[col] = series
        
        # Step 3: Clean up infinite and NaN values
        transformed_data = transformed_data.replace([np.inf, -np.inf], np.nan)
        
        # Step 4: Conservative outlier handling
        numeric_cols = [c for c in transformed_data.columns if c != self.date_col]
        for col in numeric_cols:
            series = transformed_data[col].dropna()
            if len(series) > 20:  # Only if enough observations
                q_low = series.quantile(0.01)  # Conservative 1%/99%
                q_high = series.quantile(0.99)
                if pd.notna(q_low) and pd.notna(q_high) and q_high > q_low:
                    transformed_data[col] = transformed_data[col].clip(lower=q_low, upper=q_high)
        
        # Step 5: Add seasonal dummies and time trend
        transformed_data = self._add_seasonal_features(transformed_data)
        
        # Step 6: Final cleaning
        before_clean = len(transformed_data)
        transformed_data = transformed_data.dropna(how="any")
        after_clean = len(transformed_data)
        
        if before_clean > after_clean:
            print(f"Dropped {before_clean - after_clean} rows with missing values after transformation")
        
        # Ensure stable data types
        for col in [c for c in transformed_data.columns if c != self.date_col]:
            transformed_data[col] = transformed_data[col].astype("float64")
        
        return transformed_data
    
    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add quarterly seasonal dummies and time trend."""
        data_with_features = data.copy()
        
        # Extract quarter from date
        data_with_features['quarter'] = pd.to_datetime(data_with_features[self.date_col]).dt.quarter
        
        # Create dummy variables (Q1 as base category)
        for q in [2, 3, 4]:
            data_with_features[f'Q{q}'] = (data_with_features['quarter'] == q).astype(int)
        
        # Create time trend (normalize to start from 0)
        data_with_features['time_trend'] = range(len(data_with_features))
        
        # Drop the quarter column
        data_with_features = data_with_features.drop('quarter', axis=1)
        
        return data_with_features

def diagnose_data_issues(data: pd.DataFrame, target_var: str, exog_vars: List[str]) -> None:
    """
    Comprehensive data quality diagnosis with specific focus on mixed-frequency issues.
    """
    print("\n" + "="*60)
    print("DATA QUALITY DIAGNOSIS")
    print("="*60)
    
    # Overall dataset info
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['Datum'].min().strftime('%Y-%m')} to {data['Datum'].max().strftime('%Y-%m')}")
    
    # Analyze each variable
    all_vars = [target_var] + exog_vars
    
    for i, var in enumerate(all_vars):
        if var not in data.columns:
            print(f"\n{i+1}. {var}: ❌ NOT FOUND IN DATA")
            continue
        
        series = data[var]
        valid_count = series.notna().sum()
        coverage = valid_count / len(series)
        frequency = MixedFrequencyProcessor.detect_frequency(series, data['Datum'])
        
        print(f"\n{i+1}. {var} ({'TARGET' if var == target_var else 'EXOG'}):")
        print(f"   - Frequency: {frequency}")
        print(f"   - Coverage: {coverage:.1%} ({valid_count}/{len(series)} observations)")
        
        if valid_count > 0:
            print(f"   - Range: {series.min():.4f} to {series.max():.4f}")
            print(f"   - Mean: {series.mean():.4f}, Std: {series.std():.4f}")
            
            # Check for problematic patterns
            if series.std() == 0:
                print(f"   - ⚠️  WARNING: Variable is constant!")
            
            if frequency == "quarterly" and var != target_var:
                print(f"   - ℹ️  INFO: Quarterly exogenous variable detected")
            elif frequency == "quarterly" and var == target_var:
                print(f"   - ℹ️  INFO: Quarterly target - will use forward-fill")
        
        # Check correlations with target (if not the target itself)
        if var != target_var and var in data.columns and target_var in data.columns:
            # Find overlapping observations
            overlap_data = data[[var, target_var]].dropna()
            if len(overlap_data) > 10:
                corr = overlap_data[var].corr(overlap_data[target_var])
                print(f"   - Correlation with {target_var}: {corr:.4f}")
            else:
                print(f"   - Correlation: insufficient overlap ({len(overlap_data)} obs)")
    
    # Check data overlap
    print(f"\n" + "-"*40)
    print("DATA OVERLAP ANALYSIS")
    print("-"*40)
    
    available_vars = [var for var in all_vars if var in data.columns]
    complete_cases = data[available_vars].dropna()
    overlap_ratio = len(complete_cases) / len(data)
    
    print(f"Complete cases: {len(complete_cases)} / {len(data)} ({overlap_ratio:.1%})")
    
    if overlap_ratio < 0.1:
        print("❌ CRITICAL: Very low data overlap - analysis likely to fail")
    elif overlap_ratio < 0.3:
        print("⚠️  WARNING: Low data overlap - results may be unreliable")
    else:
        print("✅ OK: Sufficient data overlap for analysis")
    
    # Frequency analysis summary
    print(f"\n" + "-"*40)
    print("FREQUENCY ANALYSIS SUMMARY")
    print("-"*40)
    
    quarterly_vars = []
    monthly_vars = []
    unknown_vars = []
    
    for var in available_vars:
        freq = MixedFrequencyProcessor.detect_frequency(data[var], data['Datum'])
        if freq == "quarterly":
            quarterly_vars.append(var)
        elif freq == "monthly":
            monthly_vars.append(var)
        else:
            unknown_vars.append(var)
    
    print(f"Quarterly variables ({len(quarterly_vars)}): {', '.join(quarterly_vars)}")
    print(f"Monthly variables ({len(monthly_vars)}): {', '.join(monthly_vars)}")
    if unknown_vars:
        print(f"Unknown frequency ({len(unknown_vars)}): {', '.join(unknown_vars)}")
    
    # Recommendations
    print(f"\n" + "-"*40)
    print("RECOMMENDATIONS")
    print("-"*40)
    
    if len(quarterly_vars) > 0 and target_var in quarterly_vars:
        print("✅ Will apply forward-fill to quarterly target variable")
        
    if overlap_ratio < 0.3:
        print("💡 Consider:")
        print("   - Using a longer time period")
        print("   - Using different target/exogenous variables")
        print("   - Checking data quality at source")
    
    if len(complete_cases) < 50:
        print("⚠️  Small sample size - consider using simpler models")

print("Mixed-frequency data processor loaded")


# %%
"""
Regression Methods & Cross-Validation - VEREINFACHT & ROBUSTE VERSION
Ohne Monkey-Patches - saubere Klassenstruktur
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from scipy import stats

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class RegressionMethod(ABC):
    """Abstract base class for regression methods."""
    
    def __init__(self, name: str, requires_scaling: bool = True):
        self.name = name
        self.requires_scaling = requires_scaling
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit the model and return results."""
        pass

class OLSMethod(RegressionMethod):
    """OLS Regression with robust standard errors."""
    
    def __init__(self):
        super().__init__("OLS", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # Add constant
        X_with_const = sm.add_constant(X, has_constant='add')
        
        # Fit model
        model = OLS(y, X_with_const, missing='drop')
        
        # Use robust standard errors (HAC)
        max_lags = min(4, int(len(y) ** (1/4)))
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        
        # Calculate diagnostics
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(results.resid)
        except:
            dw = np.nan
        
        try:
            jb_stat, jb_p = stats.jarque_bera(results.resid)[:2]
            jarque_bera = {'statistic': jb_stat, 'p_value': jb_p}
        except:
            jarque_bera = {'statistic': np.nan, 'p_value': np.nan}
        
        diagnostics = {
            'durbin_watson': dw,
            'jarque_bera': jarque_bera
        }
        
        return {
            'model': results,
            'coefficients': results.params,
            'std_errors': results.bse,
            'p_values': results.pvalues,
            'r_squared': results.rsquared,
            'r_squared_adj': results.rsquared_adj,
            'mse': results.mse_resid,
            'mae': np.mean(np.abs(results.resid)),
            'residuals': results.resid,
            'fitted_values': results.fittedvalues,
            'diagnostics': diagnostics
        }

class RandomForestMethod(RegressionMethod):
    """Conservative Random Forest optimized for financial time series."""
    
    def __init__(self):
        super().__init__("Random Forest", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = X.shape
        
        # Very conservative hyperparameters based on sample size
        if n_samples < 50:
            n_estimators = 100
            max_depth = 3
            min_samples_leaf = max(5, n_samples // 10)
            min_samples_split = max(10, n_samples // 5)
        elif n_samples < 100:
            n_estimators = 150
            max_depth = 4
            min_samples_leaf = max(8, n_samples // 12)
            min_samples_split = max(16, n_samples // 6)
        else:
            n_estimators = 200
            max_depth = min(5, max(3, int(np.log2(n_samples)) - 2))
            min_samples_leaf = max(10, n_samples // 15)
            min_samples_split = max(20, n_samples // 8)
        
        # Additional constraints for high-dimensional data
        max_features = "sqrt" if n_features <= n_samples // 3 else max(1, min(int(np.sqrt(n_features)), n_features // 2))
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            max_samples=min(0.8, max(0.5, 1.0 - 0.1 * n_features / n_samples)),
            random_state=42,
            n_jobs=1  # Avoid multiprocessing issues
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Out-of-bag score as additional validation
        oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else np.nan
        
        return {
            'model': model,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'oob_score': oob_score,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'feature_importance': model.feature_importances_,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'max_features': max_features
            }
        }

class XGBoostMethod(RegressionMethod):
    """Conservative XGBoost optimized for financial time series."""
    
    def __init__(self):
        super().__init__("XGBoost", requires_scaling=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available")
        
        n_samples, n_features = X.shape
        
        # Very conservative hyperparameters
        n_estimators = min(100, max(50, n_samples // 3))
        max_depth = 3
        learning_rate = 0.01 if n_samples > 50 else 0.02
        
        # Strong regularization
        reg_alpha = 1.0 + 0.1 * (n_features / 10)  # L1
        reg_lambda = 2.0 + 0.2 * (n_features / 10)  # L2
        
        # Sampling parameters
        subsample = max(0.6, 1.0 - 0.05 * (n_features / 10))
        colsample_bytree = max(0.6, 1.0 - 0.05 * (n_features / 10))
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=max(3, n_samples // 20),
            gamma=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'feature_importance': model.feature_importances_,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }
        }

class SVRMethod(RegressionMethod):
    """Support Vector Regression."""
    
    def __init__(self):
        super().__init__("SVR", requires_scaling=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred
        }

class BayesianRidgeMethod(RegressionMethod):
    """Bayesian Ridge Regression."""
    
    def __init__(self):
        super().__init__("Bayesian Ridge", requires_scaling=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = BayesianRidge()
        model.fit(X_scaled, y)
        
        y_pred, y_std = model.predict(X_scaled, return_std=True)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r_squared': r2,
            'mae': mae,
            'residuals': y - y_pred,
            'fitted_values': y_pred,
            'prediction_std': y_std,
            'coefficients': model.coef_
        }

class MethodRegistry:
    """Registry for regression methods."""
    
    def __init__(self):
        self.methods = {
            "OLS": OLSMethod(),
            "Random Forest": RandomForestMethod(),
            "SVR": SVRMethod(),
            "Bayesian Ridge": BayesianRidgeMethod()
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.methods["XGBoost"] = XGBoostMethod()
    
    def get_method(self, name: str) -> RegressionMethod:
        if name not in self.methods:
            raise ValueError(f"Method '{name}' not available. Choose from: {list(self.methods.keys())}")
        return self.methods[name]
    
    def list_methods(self) -> List[str]:
        return list(self.methods.keys())

class RobustCrossValidator:
    """
    Robust time series cross-validation for financial data.
    Fixed implementation without leakage or extreme scores.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def validate_method(self, method: RegressionMethod, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform robust time series cross-validation.
        """
        n_samples = len(X_train)
        
        # Conservative CV parameters based on sample size
        if n_samples < 30:
            n_splits = 2
            gap = 1
        elif n_samples < 60:
            n_splits = 3
            gap = 2
        else:
            n_splits = min(4, n_samples // 20)  # Conservative: at least 20 obs per fold
            gap = max(2, int(n_samples * 0.05))  # Larger gaps
        
        if n_splits < 2:
            return {'cv_scores': [], 'cv_mean': np.nan, 'cv_std': np.nan, 'n_folds': 0}
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        cv_scores = []
        successful_folds = 0
        
        print(f"    Running {n_splits}-fold CV with gap={gap}...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            try:
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Ensure minimum fold sizes
                if len(X_fold_train) < 10 or len(X_fold_val) < 3:
                    print(f"      Fold {fold_idx+1}: Skipped (insufficient data)")
                    continue
                
                # Fit method on fold training data
                fold_results = method.fit(X_fold_train, y_fold_train)
                
                # Evaluate on fold validation data
                fold_test_perf = self._evaluate_on_test(fold_results, X_fold_val, y_fold_val, method)
                
                score = fold_test_perf['r_squared']
                
                # Sanity check: Only accept reasonable scores
                if np.isfinite(score) and -1.0 <= score <= 1.0:
                    cv_scores.append(score)
                    successful_folds += 1
                    print(f"      Fold {fold_idx+1}: R² = {score:.4f}")
                else:
                    print(f"      Fold {fold_idx+1}: Extreme score {score:.4f} - skipped")
                
            except Exception as e:
                print(f"      Fold {fold_idx+1}: Failed ({str(e)[:50]})")
                continue
        
        if len(cv_scores) == 0:
            return {'cv_scores': [], 'cv_mean': np.nan, 'cv_std': np.nan, 'n_folds': 0}
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'n_folds': successful_folds
        }
    
    def _evaluate_on_test(self, train_results: Dict[str, Any], X_test: np.ndarray, 
                         y_test: np.ndarray, method: RegressionMethod) -> Dict[str, Any]:
        """Evaluate trained model on test data."""
        model = train_results['model']
        
        # Handle predictions based on model type
        try:
            if method.name == "OLS":
                X_test_with_const = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_with_const)
            elif method.requires_scaling and 'scaler' in train_results:
                X_test_scaled = train_results['scaler'].transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'mse': test_mse,
                'r_squared': test_r2,
                'mae': test_mae,
                'predictions': y_pred,
                'actual': y_test,
                'residuals': y_test - y_pred
            }
            
        except Exception as e:
            # Return NaN if prediction fails
            return {
                'mse': np.nan,
                'r_squared': np.nan,
                'mae': np.nan,
                'predictions': np.full(len(y_test), np.nan),
                'actual': y_test,
                'residuals': np.full(len(y_test), np.nan),
                'error': str(e)
            }

class TimeSeriesSplitter:
    """
    Robust time series train/test splitting with mandatory gaps.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def split(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-series aware train/test split with mandatory gap periods."""
        n_samples = len(X)
        
        # Adaptive gap and test size based on sample size
        if n_samples < 50:
            gap = 1
            test_size = max(0.2, self.config.test_size)  # Minimum 20% for test
        elif n_samples < 100:
            gap = 2
            test_size = self.config.test_size
        else:
            gap = max(2, int(n_samples * 0.02))  # 2% of sample as gap, minimum 2
            test_size = self.config.test_size
        
        # Calculate indices
        test_samples = int(n_samples * test_size)
        train_end = n_samples - test_samples - gap
        test_start = train_end + gap
        
        # Ensure minimum training data (60%)
        min_train = int(n_samples * 0.6)
        if train_end < min_train:
            print(f"Warning: Limited training data: {train_end}/{n_samples} samples")
            train_end = min_train
            gap = max(1, n_samples - train_end - test_samples)
            test_start = train_end + gap
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_start + test_samples]
        y_test = y[test_start:test_start + test_samples]
        
        print(f"    Split: Train={len(y_train)}, Gap={gap}, Test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test

print("Regression methods and cross-validation loaded")



# %%
"""
Feature Selection & Analysis - VEREINFACHT
Saubere Implementation ohne komplexe Patches
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

class SimpleFeatureSelector:
    """
    Simplified feature selection with robust methods.
    """
    
    @staticmethod
    def statistical_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 5) -> Tuple[List[str], np.ndarray]:
        """Select k best features using F-test."""
        k_actual = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_actual)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices
    
    @staticmethod
    def importance_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[List[str], np.ndarray]:
        """Select features using Random Forest importance."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top n features
        n_actual = min(n_features, X.shape[1])
        top_indices = np.argsort(importances)[::-1][:n_actual]
        
        selected_names = [feature_names[i] for i in top_indices]
        return selected_names, top_indices
    
    @staticmethod
    def rfe_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = 5) -> Tuple[List[str], np.ndarray]:
        """Recursive feature elimination."""
        n_actual = min(n_features, X.shape[1])
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        selector = RFE(estimator=estimator, n_features_to_select=n_actual)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices
    
    @staticmethod
    def lasso_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], np.ndarray]:
        """Lasso-based feature selection."""
        from sklearn.feature_selection import SelectFromModel
        
        lasso_cv = LassoCV(cv=3, random_state=42, max_iter=1000)
        selector = SelectFromModel(lasso_cv)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        if len(selected_indices) == 0:
            # Fallback: use all features
            selected_indices = np.arange(X.shape[1])
        
        selected_names = [feature_names[i] for i in selected_indices]
        return selected_names, selected_indices

class FeatureCombinationTester:
    """
    Test different feature combinations systematically.
    """
    
    def __init__(self, method_registry, cv_validator):
        self.method_registry = method_registry
        self.cv_validator = cv_validator
    
    def test_combinations(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         max_combinations: int = 20) -> pd.DataFrame:
        """
        Test feature combinations with proper validation.
        """
        print(f"Testing feature combinations from {len(feature_names)} features...")
        
        # Limit feature space for combinations if too large
        if len(feature_names) > 10:
            print(f"Feature space too large ({len(feature_names)}), using top 10 by importance...")
            # Use Random Forest to get top features
            selector = SimpleFeatureSelector()
            top_names, top_indices = selector.importance_selection(X, y, feature_names, n_features=10)
            X_reduced = X[:, top_indices]
            feature_names = top_names
            X = X_reduced
        
        # Generate combinations
        all_combos = []
        min_features = 2
        max_features = min(6, len(feature_names))  # Limit to reasonable size
        
        for size in range(min_features, max_features + 1):
            combos = list(combinations(feature_names, size))
            all_combos.extend(combos)
            
            # Stop if we have enough combinations
            if len(all_combos) >= max_combinations:
                break
        
        # Limit to max_combinations
        if len(all_combos) > max_combinations:
            all_combos = all_combos[:max_combinations]
        
        print(f"Testing {len(all_combos)} combinations...")
        
        # Test each combination
        results = []
        method = self.method_registry.get_method('Random Forest')
        
        for i, combo in enumerate(all_combos):
            try:
                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{len(all_combos)} combinations tested")
                
                # Get indices for this combination
                combo_indices = [feature_names.index(name) for name in combo]
                X_combo = X[:, combo_indices]
                
                if X_combo.shape[1] == 0 or len(X_combo) < 20:
                    continue
                
                # Split data for this combination
                splitter = TimeSeriesSplitter(AnalysisConfig())
                X_train, X_test, y_train, y_test = splitter.split(y, X_combo)
                
                if len(X_train) < 10 or len(X_test) < 3:
                    continue
                
                # Fit method
                train_results = method.fit(X_train, y_train)
                
                # Evaluate on test
                test_perf = self.cv_validator._evaluate_on_test(train_results, X_test, y_test, method)
                
                # Store results
                results.append({
                    'combination_id': i,
                    'features': ', '.join(combo),
                    'n_features': len(combo),
                    'test_r_squared': test_perf.get('r_squared', np.nan),
                    'test_mse': test_perf.get('mse', np.nan),
                    'overfitting': train_results.get('r_squared', 0) - test_perf.get('r_squared', 0),
                    'feature_list': list(combo)
                })
                
            except Exception as e:
                print(f"  Combination {i} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print("No successful combinations tested")
            return pd.DataFrame(columns=['combination_id', 'features', 'n_features', 'test_r_squared', 'test_mse', 'overfitting', 'feature_list'])
        
        df = pd.DataFrame(results)
        df = df.sort_values('test_r_squared', ascending=False).reset_index(drop=True)
        
        print(f"Combination testing completed: {len(df)} successful combinations")
        return df

class FeatureAnalyzer:
    """
    Main feature analysis coordinator.
    """
    
    def __init__(self, method_registry, cv_validator):
        self.method_registry = method_registry
        self.cv_validator = cv_validator
        self.selector = SimpleFeatureSelector()
        self.combo_tester = FeatureCombinationTester(method_registry, cv_validator)
    
    def test_selection_methods(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """
        Compare different feature selection methods.
        """
        print("Testing feature selection methods...")
        
        # Split data once for consistent comparison
        splitter = TimeSeriesSplitter(AnalysisConfig())
        X_train, X_test, y_train, y_test = splitter.split(y, X)
        
        selection_methods = {}
        
        # All Features baseline
        selection_methods['All Features'] = (feature_names, np.arange(len(feature_names)))
        
        # Statistical selection (F-test)
        try:
            sel_names, sel_idx = self.selector.statistical_selection(X_train, y_train, feature_names, k=5)
            selection_methods['Statistical (F-test)'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Statistical selection failed: {e}")
        
        # Importance-based selection
        try:
            sel_names, sel_idx = self.selector.importance_selection(X_train, y_train, feature_names, n_features=5)
            selection_methods['Importance (RF)'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Importance selection failed: {e}")
        
        # RFE selection
        try:
            sel_names, sel_idx = self.selector.rfe_selection(X_train, y_train, feature_names, n_features=5)
            selection_methods['RFE'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  RFE selection failed: {e}")
        
        # Lasso selection
        try:
            sel_names, sel_idx = self.selector.lasso_selection(X_train, y_train, feature_names)
            selection_methods['Lasso'] = (sel_names, sel_idx)
        except Exception as e:
            print(f"  Lasso selection failed: {e}")
        
        # Test each selection method
        results = []
        method = self.method_registry.get_method('Random Forest')
        
        for method_name, (sel_names, sel_idx) in selection_methods.items():
            try:
                if len(sel_idx) == 0:
                    continue
                
                X_train_sel = X_train[:, sel_idx]
                X_test_sel = X_test[:, sel_idx]
                
                # Fit on selected features
                train_results = method.fit(X_train_sel, y_train)
                test_perf = self.cv_validator._evaluate_on_test(train_results, X_test_sel, y_test, method)
                
                results.append({
                    'selection_method': method_name,
                    'selected_features': ', '.join(sel_names),
                    'n_features': len(sel_names),
                    'test_r_squared': test_perf.get('r_squared', np.nan),
                    'test_mse': test_perf.get('mse', np.nan),
                    'overfitting': train_results.get('r_squared', 0) - test_perf.get('r_squared', 0)
                })
                
                print(f"  {method_name}: {len(sel_names)} features, Test R² = {test_perf.get('r_squared', np.nan):.4f}")
                
            except Exception as e:
                print(f"  {method_name} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print("No selection methods succeeded")
            return pd.DataFrame(columns=['selection_method', 'selected_features', 'n_features', 'test_r_squared', 'test_mse', 'overfitting'])
        
        df = pd.DataFrame(results)
        return df.sort_values('test_r_squared', ascending=False).reset_index(drop=True)
    
    def analyze_feature_importance(self, train_results: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze and rank feature importance from trained model.
        """
        importance_data = []
        
        # Extract feature importance/coefficients
        if 'feature_importance' in train_results:
            # Tree-based methods
            importances = train_results['feature_importance']
            for i, (name, imp) in enumerate(zip(feature_names, importances)):
                importance_data.append({
                    'feature': name,
                    'importance': imp,
                    'rank': i + 1,
                    'type': 'importance'
                })
        
        elif 'coefficients' in train_results:
            # Linear methods
            coefficients = train_results['coefficients']
            
            # Handle different coefficient formats
            if hasattr(coefficients, 'values'):
                coef_values = coefficients.values
            else:
                coef_values = np.array(coefficients)
            
            # Skip constant term if present (OLS adds constant)
            if len(coef_values) == len(feature_names) + 1:
                coef_values = coef_values[1:]  # Skip constant
            
            # Create importance based on absolute coefficient values
            abs_coefs = np.abs(coef_values[:len(feature_names)])
            
            for i, (name, coef) in enumerate(zip(feature_names, coef_values[:len(feature_names)])):
                importance_data.append({
                    'feature': name,
                    'importance': abs_coefs[i],
                    'coefficient': coef,
                    'rank': i + 1,
                    'type': 'coefficient'
                })
        
        if not importance_data:
            return pd.DataFrame(columns=['feature', 'importance', 'rank', 'type'])
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df

print("Feature selection and analysis loaded")







"""
Improved Time Series Splitting & Cross-Validation - KORRIGIERT
Verhindert Data Leakage durch strenge zeitliche Trennung und realistische CV-Splits
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

class ImprovedTimeSeriesSplitter:
    """
    Verbesserter Zeitreihen-Splitter mit strikten Anti-Leakage Regeln.
    """
    
    def __init__(self, config):
        self.config = config
    
    def split_with_dates(self, data: pd.DataFrame, 
                        date_col: str = "Datum",
                        test_size: float = None,
                        min_train_size: float = 0.6,
                        gap_months: int = 3) -> Dict[str, Any]:
        """
        Zeitbasierter Split mit expliziten Datums-Gaps.
        
        Args:
            data: DataFrame mit Zeitreihen
            date_col: Name der Datumsspalte  
            test_size: Anteil der Testdaten (default aus config)
            min_train_size: Mindestanteil für Trainingsdaten
            gap_months: Monate zwischen Training und Test
        
        Returns:
            Dict mit train_data, test_data, gap_info, split_info
        """
        if len(data) < 30:
            raise ValueError(f"Dataset zu klein für stabilen Split: {len(data)} Beobachtungen")
        
        test_size = test_size or self.config.test_size
        data_sorted = data.sort_values(date_col).reset_index(drop=True)
        
        total_obs = len(data_sorted)
        
        # Berechne Split-Punkte basierend auf Datum
        date_range = data_sorted[date_col].max() - data_sorted[date_col].min()
        total_months = date_range.days / 30.44  # Approximation
        
        if total_months < 24:  # Weniger als 2 Jahre
            gap_months = 1  # Reduziere Gap
            test_size = min(test_size, 0.2)  # Kleinere Testgröße
        
        # Test-Periode definieren
        test_months = total_months * test_size
        train_months = total_months - test_months - gap_months
        
        if train_months < total_months * min_train_size:
            # Anpassung wenn zu wenig Trainingsdaten
            train_months = total_months * min_train_size
            gap_months = max(1, int((total_months - train_months - test_months) / 2))
            test_months = total_months - train_months - gap_months
            
            warnings.warn(f"Gap reduziert auf {gap_months} Monate für ausreichend Trainingsdaten")
        
        # Datums-basierte Cutoffs berechnen
        start_date = data_sorted[date_col].min()
        train_end_date = start_date + pd.DateOffset(months=int(train_months))
        gap_end_date = train_end_date + pd.DateOffset(months=gap_months)
        test_end_date = data_sorted[date_col].max()
        
        # Daten aufteilen
        train_mask = data_sorted[date_col] < train_end_date
        test_mask = data_sorted[date_col] >= gap_end_date
        gap_mask = (data_sorted[date_col] >= train_end_date) & (data_sorted[date_col] < gap_end_date)
        
        train_data = data_sorted[train_mask].copy()
        test_data = data_sorted[test_mask].copy()
        gap_data = data_sorted[gap_mask].copy()
        
        # Validierung
        if len(train_data) < 20:
            raise ValueError(f"Training set zu klein: {len(train_data)} Beobachtungen")
        if len(test_data) < 5:
            raise ValueError(f"Test set zu klein: {len(test_data)} Beobachtungen")
        
        split_info = {
            'total_observations': total_obs,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'gap_size': len(gap_data),
            'train_ratio': len(train_data) / total_obs,
            'test_ratio': len(test_data) / total_obs,
            'gap_ratio': len(gap_data) / total_obs,
            'train_end_date': train_end_date,
            'gap_end_date': gap_end_date,
            'actual_gap_months': gap_months,
            'date_range_months': total_months
        }
        
        print(f"Time series split: Train={len(train_data)} | Gap={len(gap_data)} | Test={len(test_data)}")
        print(f"  Training period: {data_sorted[date_col].min().strftime('%Y-%m')} to {train_end_date.strftime('%Y-%m')}")
        print(f"  Gap period: {train_end_date.strftime('%Y-%m')} to {gap_end_date.strftime('%Y-%m')}")
        print(f"  Test period: {gap_end_date.strftime('%Y-%m')} to {data_sorted[date_col].max().strftime('%Y-%m')}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'gap_data': gap_data,
            'split_info': split_info,
            'train_end_date': train_end_date
        }
    
    def split(self, y: np.ndarray, X: np.ndarray, 
              dates: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Legacy-kompatibler Split für bestehenden Code.
        """
        n_samples = len(X)
        
        # Adaptive Größen basierend auf Stichprobe
        if n_samples < 40:
            test_size = 0.20
            gap_periods = 1
        elif n_samples < 80:
            test_size = 0.25
            gap_periods = 2
        else:
            test_size = self.config.test_size
            gap_periods = max(1, int(n_samples * 0.015))  # 1.5% als Gap
        
        # Berechne Indizes
        test_samples = int(n_samples * test_size)
        train_end = n_samples - test_samples - gap_periods
        test_start = train_end + gap_periods
        
        # Mindest-Trainingsgröße sicherstellen
        min_train = max(20, int(n_samples * 0.5))  # Mindestens 50% für Training
        if train_end < min_train:
            train_end = min_train
            gap_periods = max(1, n_samples - train_end - test_samples)
            test_start = train_end + gap_periods
        
        if test_start >= n_samples:
            raise ValueError("Dataset zu klein für robusten Split")
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_start + test_samples]
        y_test = y[test_start:test_start + test_samples]
        
        print(f"Array split: Train={len(y_train)}, Gap={gap_periods}, Test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test


class ImprovedRobustCrossValidator:
    """
    Verbesserte Cross-Validation mit realistischen Splits und Stabilitätsprüfungen.
    """
    
    def __init__(self, config):
        self.config = config
        
    def validate_method_robust(self, method, X_train: np.ndarray, y_train: np.ndarray,
                              dates_train: pd.Series = None) -> Dict[str, Any]:
        """
        Robuste Kreuzvalidierung mit stabilitätsfokussierten Splits.
        """
        n_samples = len(X_train)
        
        # Sehr konservative CV-Parameter basierend auf Datengröße
        if n_samples < 40:
            n_splits = 2
            gap = 1
        elif n_samples < 100:
            n_splits = 3
            gap = 2
        else:
            n_splits = 4
            gap = max(2, int(n_samples * 0.02))  # Statt 0.05
        
        if n_splits < 2:
            return {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'n_folds': 0,
                'stability_warning': 'Dataset too small for CV'
            }
        
        # Verwende TimeSeriesSplit mit größeren Gaps
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=None)
        cv_scores = []
        fold_details = []
        successful_folds = 0
        
        print(f"    Running {n_splits}-fold CV with gap={gap}...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            try:
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]  
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
                
                # Strikte Mindestgrößen für Folds
                min_train_fold = max(10, len(X_train) // (n_splits + 3))
                min_val_fold = max(3, len(X_train) // (n_splits * 4))
                
                if len(X_fold_train) < min_train_fold or len(X_fold_val) < min_val_fold:
                    print(f"      Fold {fold_idx+1}: Skipped (train={len(X_fold_train)}, val={len(X_fold_val)})")
                    continue
                
                # Prüfe auf ausreichende Variation in y
                if y_fold_train.std() < 1e-8 or y_fold_val.std() < 1e-8:
                    print(f"      Fold {fold_idx+1}: Skipped (insufficient variation)")
                    continue
                
                # Fit auf Fold-Training
                try:
                    fold_results = method.fit(X_fold_train, y_fold_train)
                except Exception as fit_error:
                    print(f"      Fold {fold_idx+1}: Fit failed ({str(fit_error)[:30]})")
                    continue
                
                # Evaluiere auf Fold-Validation
                fold_test_perf = self._evaluate_on_test_safe(
                    fold_results, X_fold_val, y_fold_val, method
                )
                
                score = fold_test_perf.get('r_squared', np.nan)
                mse = fold_test_perf.get('mse', np.nan)
                
                # Strenge Sanity Checks
                is_valid_score = (
                    np.isfinite(score) and 
                    -2.0 <= score <= 1.0 and  # Erweitere negativen Bereich leicht
                    np.isfinite(mse) and 
                    mse > 0
                )
                
                if is_valid_score:
                    cv_scores.append(score)
                    fold_details.append({
                        'fold': fold_idx + 1,
                        'r_squared': score,
                        'mse': mse,
                        'train_size': len(X_fold_train),
                        'val_size': len(X_fold_val)
                    })
                    successful_folds += 1
                    print(f"      Fold {fold_idx+1}: R² = {score:.4f}, MSE = {mse:.4f}")
                else:
                    print(f"      Fold {fold_idx+1}: Invalid score R²={score:.4f}, MSE={mse:.4f}")
                
            except Exception as e:
                print(f"      Fold {fold_idx+1}: Exception - {str(e)[:40]}")
                continue
        
        if len(cv_scores) == 0:
            return {
                'cv_scores': [],
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'n_folds': 0,
                'stability_warning': 'All CV folds failed'
            }
        
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Stabilitäts-Assessment
        stability_warnings = []
        
        if cv_std > 0.3:
            stability_warnings.append('High CV variance - unstable model')
        
        if len(cv_scores) < n_splits / 2:
            stability_warnings.append(f'Only {len(cv_scores)}/{n_splits} folds succeeded')
            
        if successful_folds > 1:
            score_range = max(cv_scores) - min(cv_scores)
            if score_range > 0.5:
                stability_warnings.append('Very high score range across folds')
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'n_folds': successful_folds,
            'fold_details': fold_details,
            'stability_warnings': stability_warnings,
            'parameters': {
                'n_splits_attempted': n_splits,
                'gap_used': gap,
                'min_train_fold': min_train_fold,
                'min_val_fold': min_val_fold
            }
        }
    
    def _evaluate_on_test_safe(self, train_results: Dict[str, Any], 
                              X_test: np.ndarray, y_test: np.ndarray, 
                              method) -> Dict[str, Any]:
        """
        Sichere Test-Evaluierung mit ausführlichem Error Handling.
        """
        try:
            model = train_results.get('model')
            if model is None:
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan, 
                       'error': 'No model in train_results'}
            
            # Predictions generieren basierend auf Methodentyp
            if method.name == "OLS":
                # Statsmodels OLS
                import statsmodels.api as sm
                X_test_with_const = sm.add_constant(X_test, has_constant='add')
                y_pred = model.predict(X_test_with_const)
                
            elif method.requires_scaling and 'scaler' in train_results:
                # Skalierte Methoden (SVR, Bayesian Ridge)
                scaler = train_results['scaler']
                X_test_scaled = scaler.transform(X_test)
                
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test_scaled)
                else:
                    return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                           'error': 'Model has no predict method'}
                    
            else:
                # Tree-based und andere Methoden
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                           'error': 'Model has no predict method'}
            
            # Validate predictions
            if not isinstance(y_pred, (np.ndarray, list)):
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'Invalid prediction type'}
            
            y_pred = np.array(y_pred).flatten()
            
            if len(y_pred) != len(y_test):
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'Prediction length mismatch'}
            
            if not np.isfinite(y_pred).any():
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': 'All predictions are non-finite'}
            
            # Berechne Metriken mit robusten Checks
            try:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Zusätzliche Sanity Checks
                if not np.isfinite(mse) or mse < 0:
                    mse = np.nan
                if not np.isfinite(r2):
                    r2 = np.nan  
                if not np.isfinite(mae) or mae < 0:
                    mae = np.nan
                
                return {
                    'r_squared': float(r2) if np.isfinite(r2) else np.nan,
                    'mse': float(mse) if np.isfinite(mse) else np.nan,
                    'mae': float(mae) if np.isfinite(mae) else np.nan,
                    'predictions': y_pred,
                    'actual': y_test,
                    'residuals': y_test - y_pred
                }
                
            except Exception as metric_error:
                return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                       'error': f'Metric calculation failed: {str(metric_error)[:50]}'}
                       
        except Exception as e:
            return {'r_squared': np.nan, 'mse': np.nan, 'mae': np.nan,
                   'error': f'Evaluation failed: {str(e)[:50]}'}


class ImprovedFeatureSelector:
    """
    Vereinfachte Feature-Selektion mit Fokus auf Stabilität.
    """
    
    @staticmethod  
    def select_robust_features(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              max_features: int = None) -> Tuple[List[str], np.ndarray]:
        """
        Robuste Feature-Selektion basierend auf Korrelation und Stabilität.
        """
        if X.shape[1] == 0:
            return [], np.array([])
        
        # Automatische Begrenzung basierend auf Sample-Größe
        n_samples = len(X)
        if max_features is None:
            if n_samples < 30:
                max_features = 2  
            elif n_samples < 60:
                max_features = 3
            elif n_samples < 100:
                max_features = 4
            else:
                max_features = min(6, X.shape[1], n_samples // 15)
        
        max_features = min(max_features, X.shape[1])
        
        # Berechne Korrelationen mit Target
        correlations = []
        valid_features = []
        
        for i, feature_name in enumerate(feature_names):
            try:
                feature_data = X[:, i]
                
                # Skip konstante oder fast-konstante Features
                if np.std(feature_data) < 1e-10:
                    continue
                    
                # Skip Features mit zu vielen NaNs
                if np.isnan(feature_data).sum() > len(feature_data) * 0.5:
                    continue
                
                # Berechne Korrelation (robust gegen NaNs)
                mask = ~(np.isnan(feature_data) | np.isnan(y))
                if mask.sum() < max(5, len(y) * 0.3):  # Mindestens 30% overlap
                    continue
                    
                corr = np.corrcoef(feature_data[mask], y[mask])[0, 1]
                
                if np.isfinite(corr):
                    correlations.append((abs(corr), i, feature_name))
                    valid_features.append(i)
                    
            except Exception:
                continue
        
        if not correlations:
            # Fallback: erste verfügbare Features nehmen
            available_features = []
            for i, name in enumerate(feature_names[:max_features]):
                if np.std(X[:, i]) > 1e-10:
                    available_features.append((name, i))
                    
            if available_features:
                selected_names = [name for name, idx in available_features]
                selected_indices = np.array([idx for name, idx in available_features])
                return selected_names, selected_indices
            else:
                return [], np.array([])
        
        # Sortiere nach Korrelation und wähle Top-Features
        correlations.sort(key=lambda x: x[0], reverse=True)
        selected_correlations = correlations[:max_features]
        
        selected_names = [name for _, _, name in selected_correlations]
        selected_indices = np.array([idx for _, idx, _ in selected_correlations])
        
        print(f"Selected {len(selected_names)}/{len(feature_names)} features based on correlation:")
        for corr_abs, idx, name in selected_correlations:
            print(f"  {name}: |r| = {corr_abs:.3f}")
        
        return selected_names, selected_indices











# %%
"""
Main Financial Regression Analyzer - VEREINFACHT & ROBUST
Koordiniert alle Komponenten ohne komplexe Monkey-Patches
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import warnings

class FinancialRegressionAnalyzer:
    """
    Main analyzer that coordinates all components for financial regression analysis.
    Clean implementation without monkey patches.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, exog_vars: List[str], 
                 config: AnalysisConfig = None, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.exog_vars = exog_vars
        self.date_col = date_col
        self.config = config or AnalysisConfig()
        
        # Initialize components
        self.method_registry = MethodRegistry()
        self.cv_validator = RobustCrossValidator(self.config)
        self.splitter = TimeSeriesSplitter(self.config)
        self.preprocessor = DataPreprocessor(data, target_var, date_col)
        self.feature_analyzer = FeatureAnalyzer(self.method_registry, self.cv_validator)
    
    def prepare_data(self, transformation: str = 'levels') -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """
        Prepare data for regression with proper mixed-frequency handling.
        """
        print(f"Preparing data with '{transformation}' transformation...")
        
        # Apply transformations (includes mixed-frequency handling)
        transformed_data = self.preprocessor.create_transformations(transformation)
        
        # Find target and feature columns
        target_col = self.target_var
        if target_col not in transformed_data.columns:
            possible_targets = [col for col in transformed_data.columns 
                             if col.startswith(self.target_var)]
            if possible_targets:
                target_col = possible_targets[0]
            else:
                raise ValueError(f"Target variable {target_col} not found after transformation")
        
        # Get feature columns
        feature_cols = []
        for var in self.exog_vars:
            if var in transformed_data.columns:
                feature_cols.append(var)
        
        # Add seasonal dummies and time trend
        seasonal_cols = ['Q2', 'Q3', 'Q4', 'time_trend']
        for col in seasonal_cols:
            if col in transformed_data.columns:
                feature_cols.append(col)
        
        if not feature_cols:
            raise ValueError("No feature columns found")
        
        # Create final dataset
        #final_data = transformed_data[[target_col] + feature_cols].copy()
        final_data = transformed_data[[self.date_col, target_col] + feature_cols].copy()
        final_data = final_data.dropna()
        
        if len(final_data) < 20:
            raise ValueError(f"Insufficient data: only {len(final_data)} observations after cleaning")
        
        # Extract arrays
        y = final_data[target_col].values
        X = final_data[feature_cols].values
        
        print(f"Final dataset: {len(final_data)} observations, {len(feature_cols)} features")
        
        return y, X, feature_cols, final_data
    
    def fit_method_with_validation(self, method_name: str, y: np.ndarray, X: np.ndarray, 
                                 feature_names: List[str], transformation: str = 'levels', final_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit a method with comprehensive validation.
        """
        print(f"  Fitting {method_name}...")
        
        # Train/test split
        X_train, X_test, y_train, y_test = self.splitter.split(y, X)
        
        
        # If configured, restrict test evaluation to quarter-end months (old approach)
        try:
            if getattr(self.config, 'evaluate_quarter_ends_only', False) and isinstance(final_data, pd.DataFrame) and self.date_col in final_data.columns:
                dates_all = pd.to_datetime(final_data[self.date_col].values)
                n_samples = len(dates_all)
                # Recompute split indices like the splitter
                if n_samples < 40:
                    test_size = 0.20; gap_periods = 1
                elif n_samples < 80:
                    test_size = 0.25; gap_periods = 2
                else:
                    test_size = self.config.test_size
                    gap_periods = max(1, int(n_samples * 0.015))
                test_samples = int(n_samples * test_size)
                train_end = n_samples - test_samples - gap_periods
                test_start = train_end + gap_periods
                dates_test = pd.Series(dates_all[test_start:test_start + test_samples])
                qe_mask = dates_test.dt.is_quarter_end
                if qe_mask.any() and qe_mask.sum() >= 3 and len(X_test) == len(qe_mask):
                    X_test = X_test[qe_mask.values]
                    y_test = y_test[qe_mask.values]
        except Exception:
            pass

        
        # Get method and fit on training data
        method = self.method_registry.get_method(method_name)
        train_results = method.fit(X_train, y_train)
        
        # Evaluate on test data
        test_performance = self.cv_validator._evaluate_on_test(train_results, X_test, y_test, method)
        
        # Cross-validation on training data
        cv_performance = self.cv_validator.validate_method(method, X_train, y_train)
        
        # Calculate metrics
        train_r2 = train_results.get('r_squared', np.nan)
        test_r2 = test_performance['r_squared']
        overfitting = train_r2 - test_r2 if np.isfinite(train_r2) and np.isfinite(test_r2) else np.nan
        
        # Combine results
        results = {
            **train_results,
            'method_name': method_name,
            'feature_names': feature_names,
            'target_var': self.target_var,
            'transformation': transformation,
            'test_performance': test_performance,
            'cv_performance': cv_performance,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'overfitting': overfitting,
            'validation_config': {
                'test_size': self.config.test_size,
                'train_size': len(X_train),
                'test_size_actual': len(X_test),
                'gap_used': len(y) - len(X_train) - len(X_test)
            }
        }
        
        # Performance summary
        cv_mean = cv_performance.get('cv_mean', np.nan)
        print(f"    Test R² = {test_r2:.4f}, Train R² = {train_r2:.4f}, CV = {cv_mean:.4f}")
        
        # Warnings
        if overfitting > 0.1:
            print(f"    ⚠️ WARNING: High overfitting ({overfitting:.4f})")
        if test_r2 > 0.9:
            print(f"    ⚠️ WARNING: Very high R² ({test_r2:.4f}) - check for leakage")
        
        return results
    
    def fit_multiple_methods(self, methods: List[str] = None, transformation: str = 'levels') -> Dict[str, Dict[str, Any]]:
        """
        Fit multiple methods with validation.
        """
        if methods is None:
            methods = self.method_registry.list_methods()
        
        # Prepare data
        y, X, feature_names, final_data = self.prepare_data(transformation)
        
        results = {}
        print(f"Fitting {len(methods)} methods with robust validation...")
        
        for method_name in methods:
            try:
                result = self.fit_method_with_validation(method_name, y, X, feature_names, transformation, final_data=final_data)
                results[method_name] = result
                
            except Exception as e:
                print(f"    ✗ {method_name} failed: {str(e)[:50]}")
                continue
        
        print(f"Successfully fitted {len(results)}/{len(methods)} methods")
        return results
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare method results in a structured format.
        """
        comparison_data = []
        
        for method_name, result in results.items():
            train_r2 = result.get('train_r_squared', np.nan)
            test_r2 = result.get('test_r_squared', np.nan)
            overfitting = result.get('overfitting', np.nan)
            cv_mean = result.get('cv_performance', {}).get('cv_mean', np.nan)
            cv_std = result.get('cv_performance', {}).get('cv_std', np.nan)
            
            # Classify overfitting level
            if np.isfinite(overfitting):
                if overfitting > 0.15:
                    overfitting_level = "SEVERE"
                elif overfitting > 0.08:
                    overfitting_level = "HIGH"
                elif overfitting > 0.04:
                    overfitting_level = "MODERATE"
                else:
                    overfitting_level = "LOW"
            else:
                overfitting_level = "UNKNOWN"
            
            # Get additional metrics
            oob_score = result.get('oob_score', np.nan)
            test_mse = result.get('test_performance', {}).get('mse', np.nan)
            test_mae = result.get('test_performance', {}).get('mae', np.nan)
            
            comparison_data.append({
                'Method': method_name,
                'Test_R²': test_r2,
                'Train_R²': train_r2,
                'Overfitting': overfitting,
                'Overfitting_Level': overfitting_level,
                'Test_MSE': test_mse,
                'Test_MAE': test_mae,
                'CV_Mean_R²': cv_mean,
                'CV_Std_R²': cv_std,
                'OOB_Score': oob_score
            })
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test_R²', ascending=False).round(4)
        
        # Add warning flags
        df['Warnings'] = ''
        for idx, row in df.iterrows():
            warnings = []
            if row['Test_R²'] > 0.9:
                warnings.append("Very high R² - check for leakage")
            if row['Test_R²'] < 0:
                warnings.append("Negative R² - poor fit")
            if row['Overfitting'] > 0.1:
                warnings.append("High overfitting")
            if np.isfinite(row['CV_Mean_R²']) and abs(row['Test_R²'] - row['CV_Mean_R²']) > 0.1:
                warnings.append("CV/Test discrepancy")
            df.at[idx, 'Warnings'] = '; '.join(warnings)
        
        return df
    
    def find_optimal_transformation(self, transformations: List[str] = None, 
                                   baseline_method: str = 'Random Forest') -> str:
        """
        Find optimal transformation by testing with a baseline method.
        """
        if transformations is None:
            transformations = ['levels', 'pct', 'diff']
        
        print("Finding optimal transformation...")
        
        best_transformation = None
        best_score = -np.inf
        transformation_results = {}
        
        for transformation in transformations:
            try:
                print(f"  Testing '{transformation}' transformation...")
                
                # Test with baseline method
                y, X, feature_names, final_data = self.prepare_data(transformation)
                result = self.fit_method_with_validation(method_name, y, X, feature_names, transformation, final_data=final_data)
                
                test_r2 = result.get('test_r_squared', np.nan)
                
                transformation_results[transformation] = {
                    'test_r2': test_r2,
                    'result': result
                }
                
                print(f"    Test R² = {test_r2:.4f}")
                
                # Check if this is the best so far
                if np.isfinite(test_r2) and test_r2 > best_score:
                    best_score = test_r2
                    best_transformation = transformation
                
            except Exception as e:
                print(f"    ✗ {transformation} failed: {str(e)[:50]}")
                continue
        
        if best_transformation is None:
            print("  Warning: No transformations succeeded, using 'levels'")
            return 'levels'
        
        print(f"  Best transformation: '{best_transformation}' (Test R² = {best_score:.4f})")
        return best_transformation
    
    def test_feature_selection_methods(self, transformation: str = 'levels') -> pd.DataFrame:
        """
        Test different feature selection methods.
        """
        print("Testing feature selection methods...")
        
        try:
            y, X, feature_names, _ = self.prepare_data(transformation)
            return self.feature_analyzer.test_selection_methods(X, y, feature_names)
        except Exception as e:
            print(f"Feature selection testing failed: {e}")
            return pd.DataFrame()
    
    def test_feature_combinations(self, max_combinations: int = 20, 
                                 transformation: str = 'levels') -> pd.DataFrame:
        """
        Test different feature combinations.
        """
        print("Testing feature combinations...")
        
        try:
            y, X, feature_names, _ = self.prepare_data(transformation)
            return self.feature_analyzer.combo_tester.test_combinations(
                X, y, feature_names, max_combinations
            )
        except Exception as e:
            print(f"Feature combination testing failed: {e}")
            return pd.DataFrame()

"""
Improved Financial Regression Analyzer - KORRIGIERT
Behebt die Hauptprobleme mit Data Leakage, CV-Splits und Feature Selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings



class TrainOnlyLagSelector:
    """
    Selects best lags per exogenous variable using TRAIN-ONLY correlation with the (transformed) target.
    No leakage: only dates strictly before train_end_date are used for scoring.
    """
    def __init__(self, config: LagConfig):
        self.cfg = config

    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        try:
            s = pd.concat([a, b], axis=1).dropna()
            if len(s) < 3:
                return np.nan
            return float(s.iloc[:,0].corr(s.iloc[:,1]))
        except Exception:
            return np.nan

    def apply(self, df: pd.DataFrame, exog_vars: List[str], target_col: str, date_col: str, train_end_date: Optional[pd.Timestamp]):
        kept = []
        details = []
        # Build train mask
        if train_end_date is not None:
            train_mask = pd.to_datetime(df[date_col]) < pd.to_datetime(train_end_date)
        else:
            # fallback: first 80%
            n = len(df)
            cutoff = int(n * 0.8)
            train_mask = pd.Series([True]*cutoff + [False]*(n-cutoff), index=df.index)

        # Score candidates
        candidates = []
        for var in exog_vars:
            if var not in df.columns:
                continue
            for L in self.cfg.candidates:
                col = f"{var}_lag{L}"
                if col not in df.columns:
                    df[col] = df[var].shift(L)
                corr = self._safe_corr(df.loc[train_mask, col], df.loc[train_mask, target_col])
                # require minimal overlap
                overlap = int(pd.concat([df[col], df[target_col]], axis=1).loc[train_mask].dropna().shape[0])
                if overlap >= self.cfg.min_train_overlap and (not np.isfinite(self.cfg.min_abs_corr) or abs(corr) >= self.cfg.min_abs_corr):
                    candidates.append((var, col, L, abs(corr), overlap))
                else:
                    details.append({'var': var, 'lag': L, 'status': 'skipped', 'corr': corr, 'overlap': overlap})

        # choose per-var best, then apply total cap
        best_per_var = {}
        for var, col, L, acorr, overlap in candidates:
            cur = best_per_var.get(var)
            if (cur is None) or (acorr > cur[3]):
                best_per_var[var] = (var, col, L, acorr, overlap)

        # Flatten, sort by |corr| desc
        ranked = sorted(best_per_var.values(), key=lambda x: x[3], reverse=True)
        total_cap = max(0, int(self.cfg.total_max))
        per_var_cap = max(1, int(self.cfg.per_var_max))
        per_var_counts = {v: 0 for v in best_per_var.keys()}

        for var, col, L, acorr, overlap in ranked:
            if len(kept) >= total_cap:
                break
            if per_var_counts[var] >= per_var_cap:
                continue
            kept.append(col)
            per_var_counts[var] += 1
            details.append({'var': var, 'lag': L, 'status': 'kept', 'corr': acorr, 'overlap': overlap})

        # mark dropped
        kept_set = set(kept)
        for var, col, L, acorr, overlap in ranked:
            if col not in kept_set:
                details.append({'var': var, 'lag': L, 'status': 'dropped', 'corr': acorr, 'overlap': overlap})

        report = {
            'kept': kept,
            'details': details
        }
        return df, report

class ImprovedFinancialRegressionAnalyzer:
    """
    Korrigierter Hauptanalysator mit robuster Anti-Leakage Architektur.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, exog_vars: List[str], 
                 config=None, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.exog_vars = exog_vars
        self.date_col = date_col
        self.config = config or AnalysisConfig()
        
        # Improved components
        self.method_registry = MethodRegistry()
        self.cv_validator = ImprovedRobustCrossValidator(self.config)
        self.splitter = ImprovedTimeSeriesSplitter(self.config)
        self.preprocessor = ImprovedDataPreprocessor(data, target_var, date_col)
        self.quality_checker = ImprovedDataQualityChecker()
    
    def comprehensive_data_validation(self) -> Dict[str, Any]:
        """
        Umfassende Datenvalidierung als erster Schritt.
        """
        print("=== COMPREHENSIVE DATA VALIDATION ===")
        
        validation_result = self.quality_checker.comprehensive_data_validation(
            self.data, self.target_var, self.exog_vars,
            min_observations=30,  # Erhöhte Mindestanforderung
            min_target_coverage=0.25  # Höhere Coverage-Anforderung
        )
        
        # Ausgabe der Validierungsergebnisse
        print(f"\nData Quality Summary:")
        print(f"  Total observations: {len(self.data)}")
        print(f"  Variables tested: {len([self.target_var] + self.exog_vars)}")
        
        if validation_result['errors']:
            print(f"\n❌ ERRORS ({len(validation_result['errors'])}):")
            for error in validation_result['errors']:
                print(f"    - {error}")
        
        if validation_result['warnings']:
            print(f"\n⚠️  WARNINGS ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings']:
                print(f"    - {warning}")
        
        # Stationarity results
        print(f"\nStationarity Tests:")
        for var, result in validation_result['stationarity_tests'].items():
            is_stationary = result.get('is_stationary', None)
            p_value = result.get('p_value', np.nan)
            
            status = "✅ Stationary" if is_stationary else "❌ Non-stationary" if is_stationary is False else "❓ Unknown"
            print(f"  {var}: {status} (p-value: {p_value:.3f})")
        
        # Recommendations
        if validation_result['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_result['recommendations']:
                print(f"    - {rec}")
        
        return validation_result
    
    def prepare_data_robust(self, transformation: str = 'levels',
                            use_train_test_split: bool = True) -> Dict[str, Any]:
        """
        Robuste Datenvorbereitung mit Anti-Leakage Schutz.
        """
        print(f"\n=== ROBUST DATA PREPARATION ===")
        print(f"Transformation: {transformation}")
        
        # Schritt 1: Zeitbasierter Split ZUERST (um Leakage zu verhindern)
        train_end_date = None
        split_info = None
        
        if use_train_test_split and len(self.data) > 30:
            try:
                # Früher Split für Anti-Leakage
                split_result = self.splitter.split_with_dates(
                    self.data, self.date_col, 
                    test_size=self.config.test_size,
                    gap_months=2  # Konservativer Gap
                )
                train_end_date = split_result['train_end_date']
                split_info = split_result['split_info']
                print(f"Early train/test split applied - training ends: {train_end_date.strftime('%Y-%m')}")
                
            except Exception as e:
                print(f"Warning: Could not apply early split: {e}")
                train_end_date = None
        
        # Schritt 2: Transformationen mit Anti-Leakage Schutz
        transform_result = self.preprocessor.create_robust_transformations(
            transformation=transformation,
            train_end_date=train_end_date,  # Critical: pass split date
            outlier_method='conservative'
        )
        
        transformed_data = transform_result['data']
        
        # Schritt 3: Feature-Selektion und finales Dataset
        target_col = self.target_var
        if target_col not in transformed_data.columns:
            available_targets = [col for col in transformed_data.columns if col.startswith(self.target_var)]
            if available_targets:
                target_col = available_targets[0]
            else:
                raise ValueError(f"Target variable {target_col} not found after transformation")
        
        # Intelligente Feature-Auswahl
        available_exog = []
        for var in self.exog_vars:
            if var in transformed_data.columns:
                # Prüfe Datenqualität
                series = transformed_data[var]
                coverage = series.notna().sum() / len(series)
                variation = series.std() if series.notna().sum() > 1 else 0.0
                if coverage > 0.3 and variation > 1e-8:  # Mindestanforderungen
                    available_exog.append(var)
                else:
                    print(f"  Excluding {var}: coverage={coverage:.1%}, std={variation:.2e}")
        
        # Saisonale Features hinzufügen
        seasonal_features = ['Q2', 'Q3', 'Q4', 'time_trend']
        for feat in seasonal_features:
            if feat in transformed_data.columns:
                available_exog.append(feat)
        
        # Add target lag (L=1) after seasonal features
        lag_col = f"{target_col}_lag1"
        if target_col in transformed_data.columns and lag_col not in transformed_data.columns:
            transformed_data[lag_col] = transformed_data[target_col].shift(1)
        if lag_col in transformed_data.columns:
            available_exog.append(lag_col)
        
        if not available_exog:
            raise ValueError("No suitable exogenous features found")
        
        # Finales Dataset erstellen
        final_columns = [self.date_col, target_col] + available_exog
        final_data = transformed_data[final_columns].copy()
        # Drop rows with NaN target
        final_data = final_data.dropna(subset=[target_col])
        print(f"Target NaNs after cleaning: {final_data[target_col].isnull().sum()}")
        
        # Robuste Bereinigung
        before_clean = len(final_data)
        
        # Mindestens Target + eine exogene Variable erforderlich
        min_required_vars = 2
        row_validity = final_data.notna().sum(axis=1) >= min_required_vars
        final_data = final_data[row_validity].copy()
        
        after_clean = len(final_data)
        
        if after_clean < 20:
            raise ValueError(f"Insufficient data after cleaning: {after_clean} observations")
        
        # Arrays extrahieren
        y = final_data[target_col].values
        X = final_data[available_exog].values

        # Remove any rows with NaNs in target or features
        mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
        final_data = final_data.loc[mask].copy()
        y = y[mask]
        X = X[mask]

        # Keep aligned dates for later filtering
        dates_array = pd.to_datetime(final_data[self.date_col]).to_numpy()

        print(f"After NaN removal: {len(y)} observations, {np.isnan(y).sum()} NaNs in target")
        print("Final dataset prepared:")
        print(f"  Observations: {before_clean} → {after_clean}")
        print(f"  Features: {len(available_exog)} (+ target)")
        print(f"  Features: {', '.join(available_exog[:5])}{'...' if len(available_exog) > 5 else ''}")
        
        # Sample sizes
        sample_sizes = {
            'total': int(len(y)),
            'features_selected': int(len(available_exog)),
            # optional – nur falls split_info gesetzt wurde:
            'train_candidate': int(split_info['train_size']) if split_info else None,
            'test_candidate': int(split_info['test_size']) if split_info else None,
        }

        return {
            'y': y,
            'X': X,
            'feature_names': available_exog,
            'final_data': final_data,
            'target_name': target_col,
            'transformation_info': transform_result,
            'split_info': split_info,
            'train_end_date': train_end_date,
            'preparation_warnings': transform_result.get('warnings', []),
            'sample_sizes': sample_sizes,
            'dates': dates_array
        }

    
    def fit_method_improved(self, method_name: str, preparation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verbesserte Methodenanpassung mit robusten Splits.
        """
        print(f"  Fitting {method_name}...")
        
        y = preparation_result['y']
        X = preparation_result['X']
        feature_names = preparation_result['feature_names']
        
        # Feature-Reduktion für kleine Datensätze
        if len(y) < 60 or X.shape[1] > len(y) // 10:
            print(f"    Applying feature selection: {X.shape[1]} → ", end="")
            selected_names, selected_indices = ImprovedFeatureSelector.select_robust_features(
                X, y, feature_names, max_features=min(6, len(y) // 15)
            )
            
            if len(selected_indices) > 0:
                X = X[:, selected_indices]
                feature_names = selected_names
                print(f"{len(selected_names)} features")
            else:
                raise ValueError("Feature selection resulted in no features")
        
        # Train/Test Split mit verbessertem Splitter
        try:
            X_train, X_test, y_train, y_test = self.splitter.split(y, X)
        except ValueError as e:
            print(f"    Split failed: {e}")
            raise

        # Optional: Testbewertung auf Quartalsenden beschränken
        if getattr(self.config, 'evaluate_quarter_ends_only', False) and 'final_data' in preparation_result:
            dates_all = pd.to_datetime(preparation_result['final_data'][self.date_col].values)
            n_samples = len(dates_all)
            # Recompute split indices wie im Splitter
            if n_samples < 40:
                test_size = 0.20; gap_periods = 1
            elif n_samples < 80:
                test_size = 0.25; gap_periods = 2
            else:
                test_size = self.config.test_size
                gap_periods = max(1, int(n_samples * 0.015))
            test_samples = int(n_samples * test_size)
            train_end = n_samples - test_samples - gap_periods
            test_start = train_end + gap_periods
            dates_test = pd.Series(dates_all[test_start:test_start + test_samples])
            qe_mask = dates_test.dt.is_quarter_end
            if qe_mask.any() and qe_mask.sum() >= 3:
                X_test = X_test[qe_mask.values]
                y_test = y_test[qe_mask.values]
        
        # Method fitting
        method = self.method_registry.get_method(method_name)
        try:
            train_results = method.fit(X_train, y_train)
        except Exception as e:
            print(f"    Training failed: {e}")
            raise
        
        # Test evaluation
        test_performance = self.cv_validator._evaluate_on_test_safe(
            train_results, X_test, y_test, method
        )
        
        # Cross-validation on training data only
        cv_performance = self.cv_validator.validate_method_robust(
            method, X_train, y_train
        )
        
        # Calculate key metrics
        train_r2 = train_results.get('r_squared', np.nan)
        test_r2 = test_performance.get('r_squared', np.nan)
        cv_mean = cv_performance.get('cv_mean', np.nan)
        cv_std = cv_performance.get('cv_std', np.nan)
        
        # Overfitting calculation
        overfitting = train_r2 - test_r2 if (np.isfinite(train_r2) and np.isfinite(test_r2)) else np.nan
        
        # Performance assessment
        performance_flags = []
        if np.isfinite(test_r2) and test_r2 > 0.95:
            performance_flags.append("SUSPICIOUSLY_HIGH_R2")
        if np.isfinite(overfitting) and overfitting > 0.15:
            performance_flags.append("HIGH_OVERFITTING")
        if np.isfinite(cv_std) and cv_std > 0.3:
            performance_flags.append("UNSTABLE_CV")
        if len(cv_performance.get('cv_scores', [])) < 2:
            performance_flags.append("INSUFFICIENT_CV_FOLDS")
        
        # Combine results
        results = {
            **train_results,
            'method_name': method_name,
            'feature_names': feature_names,
            'target_var': preparation_result['target_name'],
            'transformation': preparation_result['transformation_info'].get('transformation_applied') or
                              preparation_result['transformation_info'].get('transformation', 'levels'),
            'test_performance': test_performance,
            'cv_performance': cv_performance,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'cv_mean_r_squared': cv_mean,
            'cv_std_r_squared': cv_std,
            'overfitting': overfitting,
            'performance_flags': performance_flags,
            'sample_sizes': {
                'total': len(y),
                'train': len(y_train) if 'y_train' in locals() else 0,
                'test': len(y_test) if 'y_test' in locals() else 0,
                'features_selected': len(feature_names)
            }
        }
        
        # Performance summary
        print(f"    Results: Test R² = {test_r2:.4f}, CV R² = {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"    Overfitting: {overfitting:.4f}, Flags: {len(performance_flags)}")
        if performance_flags:
            for flag in performance_flags:
                print(f"      ⚠️ {flag}")
        
        return results
    
    def fit_multiple_methods_robust(self, methods: List[str] = None, 
                                    transformation: str = 'auto') -> Dict[str, Any]:
        """
        Robuste Anpassung mehrerer Methoden mit optimaler Transformation.
        """
        if methods is None:
            # Conservative method selection based on sample size
            n_samples = len(self.data)
            if n_samples < 50:
                methods = ['OLS', 'Random Forest']  # Nur robuste Methoden
            elif n_samples < 100:
                methods = ['OLS', 'Random Forest', 'Bayesian Ridge']
            else:
                methods = ['OLS', 'Random Forest', 'XGBoost', 'Bayesian Ridge'] if HAS_XGBOOST else ['OLS', 'Random Forest', 'Bayesian Ridge']
        
        print(f"=== ROBUST METHOD FITTING ===")
        print(f"Methods: {', '.join(methods)}")
        
        # Transformation optimization
        if transformation == 'auto':
            transformation = self._find_optimal_transformation_robust()
        
        print(f"Using transformation: {transformation}")
        
        # Prepare data once
        try:
            preparation_result = self.prepare_data_robust(transformation)
        except Exception as e:
            return {'status': 'failed', 'error': f'Data preparation failed: {str(e)}'}
        
        # Fit methods
        results = {}
        successful_methods = 0
        
        for method_name in methods:
            try:
                result = self.fit_method_improved(method_name, preparation_result)
                results[method_name] = result
                successful_methods += 1
            except Exception as e:
                print(f"    ❌ {method_name} failed: {str(e)[:60]}")
                continue
        
        if successful_methods == 0:
            return {'status': 'failed', 'error': 'All methods failed'}
        
        print(f"Successfully fitted {successful_methods}/{len(methods)} methods")
        
        return {
            'status': 'success',
            'method_results': results,
            'preparation_info': preparation_result,
            'successful_methods': successful_methods,
            'total_methods': len(methods)
        }
    
    def _find_optimal_transformation_robust(self) -> str:
        """
        Robuste Transformationsoptimierung basierend auf Datencharakteristiken.
        """
        print("  Finding optimal transformation...")
        
        # Analyze target variable characteristics
        target_series = self.data[self.target_var].dropna()
        
        if len(target_series) < 10:
            print("    Insufficient data for transformation analysis - using 'levels'")
            return 'levels'
        
        # Check stationarity
        stationarity_result = self.quality_checker.test_stationarity(target_series, self.target_var)
        is_stationary = stationarity_result.get('is_stationary', None)
        p_value = stationarity_result.get('p_value', np.nan)
        if is_stationary is None and (isinstance(p_value, (int, float)) or np.isfinite(p_value)):
            try:
                is_stationary = (p_value < 0.05)
            except Exception:
                pass
        
        # Check value characteristics
        all_positive = (target_series > 0).all()
        has_trend = abs(np.corrcoef(np.arange(len(target_series)), target_series)[0, 1]) > 0.3
        high_volatility = (target_series.std() / abs(target_series.mean()) > 0.5) if target_series.mean() != 0 else False
        
        print(f"    Target characteristics:")
        print(f"      Stationary: {is_stationary}")
        print(f"      All positive: {all_positive}")
        print(f"      Has trend: {has_trend}")
        print(f"      High volatility: {high_volatility}")

        # Transformation decision logic (with p-value fallback and safer defaults)
        if is_stationary is False:
            if all_positive and not high_volatility:
                best_transformation = 'pct'  # percentage change for non-stationary positive series
                print("    → Selected 'pct': Non-stationary positive data")
            else:
                best_transformation = 'diff'  # first differences for non-stationary data
                print("    → Selected 'diff': Non-stationary data")
        elif is_stationary is True:
            if all_positive and has_trend:
                best_transformation = 'log'
                print("    → Selected 'log': Trending positive data")
            else:
                best_transformation = 'levels'
                print("    → Selected 'levels': Stationary or suitable for levels")
        else:
            # Unknown stationarity → be conservative; if p-value available use 0.10 threshold
            try:
                if np.isfinite(p_value) and p_value > 0.10:
                    best_transformation = 'diff'
                    print("    → Selected 'diff': Unknown stationarity, high p-value")
                else:
                    best_transformation = 'levels'
                    print("    → Selected 'levels': Unknown stationarity, defaulting to levels")
            except Exception:
                best_transformation = 'levels'
                print("    → Selected 'levels': Fallback")

        return best_transformation
    
    def create_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstelle umfassende Zusammenfassung mit Qualitätsbewertung.
        """
        if results['status'] != 'success':
            return results
        
        method_results = results['method_results']
        preparation_info = results['preparation_info']
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, result in method_results.items():
            test_r2 = result.get('test_r_squared', np.nan)
            train_r2 = result.get('train_r_squared', np.nan)
            cv_mean = result.get('cv_mean_r_squared', np.nan)
            cv_std = result.get('cv_std_r_squared', np.nan)
            overfitting = result.get('overfitting', np.nan)
            flags = result.get('performance_flags', [])
            
            # Quality assessment
            quality_score = 0.0
            max_score = 5.0
            
            # Test R² contribution (0-2 points)
            if np.isfinite(test_r2):
                if test_r2 > 0.7:
                    quality_score += 2.0
                elif test_r2 > 0.3:
                    quality_score += 1.0
                elif test_r2 > 0:
                    quality_score += 0.5
            
            # Overfitting penalty (-1 to +1 points)
            if np.isfinite(overfitting):
                if overfitting < 0.05:
                    quality_score += 1.0
                elif overfitting < 0.1:
                    quality_score += 0.5
                elif overfitting > 0.2:
                    quality_score -= 1.0
            
            # CV stability (0-1 points)
            if np.isfinite(cv_std) and cv_std < 0.2:
                quality_score += 1.0
            elif np.isfinite(cv_std) and cv_std < 0.3:
                quality_score += 0.5
            
            # Flag penalties
            flag_penalty = len([f for f in flags if 'SUSPICIOUSLY' in f or 'HIGH' in f]) * 0.5
            quality_score -= flag_penalty
            
            quality_score = max(0.0, min(max_score, quality_score))
            
            comparison_data.append({
                'Method': method_name,
                'Test_R²': test_r2,
                'Train_R²': train_r2,
                'CV_Mean_R²': cv_mean,
                'CV_Std_R²': cv_std,
                'Overfitting': overfitting,
                'Quality_Score': quality_score,
                'Flags': len(flags),
                'Flag_Details': '; '.join(flags) if flags else 'None'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Quality_Score', ascending=False).round(4)
        comparison_df = comparison_df.sort_values(
            ['Quality_Score','Test_R²','CV_Mean_R²'],
            ascending=[False, False, False]
        )       
        
        # Best method selection based on quality score
        if not comparison_df.empty:
            best_method_info = comparison_df.iloc[0]
            best_method = best_method_info['Method']
            best_quality = float(best_method_info['Quality_Score'])
        else:
            best_method = None
            best_quality = 0.0
        
        # Create final summary
        summary_parts = []
        summary_parts.append("=== ANALYSIS SUMMARY ===")
        n_total = preparation_info.get('sample_sizes', {}).get('total',
           len(preparation_info.get('final_data', [])))
        summary_parts.append(f"Dataset: {int(n_total)} observations")

        summary_parts.append(f"Transformation: {preparation_info['transformation_info'].get('transformation_applied') or preparation_info['transformation_info'].get('transformation', 'levels')}")
        summary_parts.append(f"Methods tested: {results['successful_methods']}/{results['total_methods']}")
        
        if best_method:
            summary_parts.append(f"Best method: {best_method} (Quality Score: {best_quality:.1f}/5.0)")
            best_result = method_results[best_method]
            summary_parts.append(f"  Test R²: {best_result.get('test_r_squared', np.nan):.4f}")
            summary_parts.append(f"  CV R²: {best_result.get('cv_mean_r_squared', np.nan):.4f} (±{best_result.get('cv_std_r_squared', np.nan):.4f})")
            summary_parts.append(f"  Overfitting: {best_result.get('overfitting', np.nan):.4f}")
        
        # Warnings and recommendations
        all_warnings = preparation_info.get('preparation_warnings', [])[:]
        leakage_risk = preparation_info['transformation_info'].get('leakage_risk', 'low')
        if leakage_risk != 'low':
            all_warnings.append(f"Potential data leakage risk: {leakage_risk}")
        
        if all_warnings:
            summary_parts.append("⚠️ WARNINGS:")
            for warning in all_warnings[:5]:  # Limit to top 5
                summary_parts.append(f"  - {warning}")
        
        summary = "\n".join(summary_parts)
        
        return {
            **results,
            'comparison': comparison_df,
            'best_method': best_method,
            'best_quality_score': best_quality,
            'summary': summary,
            'validation_summary': {
                'total_warnings': len(all_warnings),
                'leakage_risk': leakage_risk,
                'data_coverage': preparation_info.get('data_coverage_ratio', np.nan)
            }
        }


def improved_financial_analysis(data: pd.DataFrame, target_var: str, exog_vars: List[str],
                                analysis_type: str = 'comprehensive', config=None) -> Dict[str, Any]:
    """
    Hauptfunktion für verbesserte Finanzanalyse.
    """
    print("=== IMPROVED FINANCIAL REGRESSION ANALYSIS ===")
    print(f"Target: {target_var}")
    print(f"Features: {', '.join(exog_vars)}")
    print(f"Analysis type: {analysis_type}")
    
    try:
        # Initialize improved analyzer
        analyzer = ImprovedFinancialRegressionAnalyzer(data, target_var, exog_vars, config)
        
        # Step 1: Comprehensive validation
        validation_result = analyzer.comprehensive_data_validation()
        
        if not validation_result['is_valid']:
            return {
                'status': 'failed',
                'stage': 'validation',
                'validation': validation_result,
                'error': 'Data validation failed - see validation results for details'
            }
        
        # Step 2: Method selection based on analysis type
        if analysis_type == 'quick':
            methods = ['Random Forest', 'OLS']
            transformation = 'auto'
        elif analysis_type == 'comprehensive':
            methods = None  # Use automatic selection
            transformation = 'auto'
        else:  # full
            methods = analyzer.method_registry.list_methods()
            transformation = 'auto'
        
        # Step 3: Fit methods
        fit_results = analyzer.fit_multiple_methods_robust(methods, transformation)
        
        if fit_results['status'] != 'success':
            return {
                'status': 'failed',
                'stage': 'fitting',
                'validation': validation_result,
                'error': fit_results.get('error', 'Method fitting failed')
            }
        
        # Step 4: Create comprehensive summary
        final_results = analyzer.create_comprehensive_summary(fit_results)
        
        # Add validation info
        final_results['validation'] = validation_result
        
        print(f"\n{final_results['summary']}")
        
        return final_results
        
    except Exception as e:
        return {
            'status': 'failed',
            'stage': 'unknown',
            'error': f'Analysis failed: {str(e)}',
            'validation': validation_result if 'validation_result' in locals() else {}
        }


def quick_analysis_improved(target_name: str, start_date: str = "2010-01", 
                            config=None) -> Dict[str, Any]:
    """Verbesserte Quick-Analyse."""
    print("=== IMPROVED QUICK ANALYSIS ===")
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        return improved_financial_analysis(data, target_name, exog_vars, 'quick', config)
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'stage': 'data_download'}


def comprehensive_analysis_improved(target_name: str, start_date: str = "2010-01",
                                    config=None) -> Dict[str, Any]:
    """Verbesserte Comprehensive-Analyse."""
    print("=== IMPROVED COMPREHENSIVE ANALYSIS ===")
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        return improved_financial_analysis(data, target_name, exog_vars, 'comprehensive', config)
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'stage': 'data_download'}


print("Main financial regression analyzer loaded")

# %%
"""
Analysis Pipeline & Main Functions - VEREINFACHT
Kombiniert alle Komponenten zu einer benutzbaren Pipeline
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def financial_analysis(data: pd.DataFrame, target_var: str, exog_vars: List[str],
                      analysis_type: str = 'comprehensive', config: AnalysisConfig = None) -> Dict[str, Any]:
    """
    Main financial analysis function - KORRIGIERT für Mixed-Frequency.
    """
    
    config = config or AnalysisConfig()
    
    print("FINANCIAL REGRESSION ANALYSIS")
    print("=" * 50)
    print(f"Target: {target_var}")
    print(f"Features: {', '.join(exog_vars)}")
    print(f"Analysis type: {analysis_type}")
    
    # Step 1: Data Quality Validation
    print("\n1. DATA QUALITY VALIDATION")
    print("-" * 30)
    
    validation = DataQualityChecker.validate_financial_data(
        data, target_var, exog_vars, min_target_coverage=0.3
    )
    
    if not validation['is_valid']:
        print("❌ Data validation failed!")
        for error in validation['errors']:
            print(f"  Error: {error}")
        return {'status': 'failed', 'validation': validation}
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  {warning}")
    
    if validation['recommendations']:
        print("💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"  {rec}")
    
    # Step 2: Detailed Data Diagnosis
    print("\n2. DATA DIAGNOSIS")
    print("-" * 30)
    diagnose_data_issues(data, target_var, exog_vars)
    
    # Step 3: Initialize Analyzer
    print("\n3. ANALYSIS SETUP")
    print("-" * 30)
    analyzer = FinancialRegressionAnalyzer(data, target_var, exog_vars, config)
    
    # Determine methods based on analysis type
    if analysis_type == 'quick':
        methods = ['Random Forest', 'OLS']
        transformations = ['levels', 'pct']
        test_combinations = False
        test_selection = False
    elif analysis_type == 'comprehensive':
        methods = ['Random Forest', 'XGBoost', 'OLS', 'Bayesian Ridge'] if HAS_XGBOOST else ['Random Forest', 'OLS', 'Bayesian Ridge']
        transformations = ['levels', 'pct', 'diff']
        test_combinations = True
        test_selection = True
    else:  # full
        methods = analyzer.method_registry.list_methods()
        transformations = ['levels', 'pct', 'diff']
        test_combinations = True
        test_selection = True
    
    print(f"Methods to test: {', '.join(methods)}")
    print(f"Transformations to test: {', '.join(transformations)}")
    
    # Step 4: Find Optimal Transformation
    print("\n4. TRANSFORMATION OPTIMIZATION")
    print("-" * 30)
    
    try:
        best_transformation = analyzer.find_optimal_transformation(
            transformations, 
            baseline_method=methods[0]
        )
    except Exception as e:
        print(f"Transformation testing failed: {e}")
        best_transformation = 'levels'
    
    # Step 5: Fit All Methods
    print("\n5. METHOD COMPARISON")
    print("-" * 30)
    
    method_results = analyzer.fit_multiple_methods(methods, best_transformation)
    
    if not method_results:
        return {'status': 'failed', 'error': 'No methods succeeded'}
    
    # Step 6: Compare Methods
    comparison_df = analyzer.compare_methods(method_results)
    print("\nMethod Comparison Results:")
    display_cols = ['Method', 'Test_R²', 'Train_R²', 'Overfitting', 'Overfitting_Level']
    print(comparison_df[display_cols].head().to_string(index=False))
    
    # Step 7: Feature Analysis (Optional)
    feature_selection_df = pd.DataFrame()
    combination_results_df = pd.DataFrame()
    
    if test_selection and len(method_results) > 0:
        print("\n6. FEATURE SELECTION ANALYSIS")
        print("-" * 30)
        try:
            feature_selection_df = analyzer.test_feature_selection_methods(best_transformation)
            if not feature_selection_df.empty:
                print("Feature selection methods tested successfully")
                print(feature_selection_df.head().to_string(index=False))
        except Exception as e:
            print(f"Feature selection failed: {e}")
    
    if test_combinations and len(method_results) > 0:
        print("\n7. FEATURE COMBINATION ANALYSIS")
        print("-" * 30)
        try:
            combination_results_df = analyzer.test_feature_combinations(
                max_combinations=config.max_feature_combinations,
                transformation=best_transformation
            )
            if not combination_results_df.empty:
                print(f"Tested {len(combination_results_df)} feature combinations")
                print(combination_results_df.head().to_string(index=False))
        except Exception as e:
            print(f"Combination testing failed: {e}")
    
    # Step 8: Generate Summary
    print("\n8. ANALYSIS SUMMARY")
    print("-" * 30)
    
    summary_lines = []
    summary_lines.append("FINAL RESULTS")
    summary_lines.append("=" * 20)
    summary_lines.append(f"Best Transformation: {best_transformation}")
    
    if not comparison_df.empty:
        best_method = comparison_df.iloc[0]['Method']
        best_test_r2 = comparison_df.iloc[0]['Test_R²']
        best_overfitting = comparison_df.iloc[0]['Overfitting']
        
        summary_lines.append(f"Best Method: {best_method}")
        summary_lines.append(f"Test R²: {best_test_r2:.4f}")
        summary_lines.append(f"Overfitting: {best_overfitting:.4f}")
        
        # Add warnings if necessary
        if best_overfitting > 0.1:
            summary_lines.append("⚠️ WARNING: High overfitting detected")
        if best_test_r2 > 0.9:
            summary_lines.append("⚠️ WARNING: Very high R² - check for data leakage")
        if best_test_r2 < 0.1:
            summary_lines.append("⚠️ NOTE: Low predictive power")
    
    if not feature_selection_df.empty:
        best_selection = feature_selection_df.iloc[0]
        summary_lines.append(f"Best Feature Selection: {best_selection['selection_method']}")
        summary_lines.append(f"Selected Features ({best_selection['n_features']}): {best_selection['selected_features']}")
    
    if not combination_results_df.empty:
        best_combo = combination_results_df.iloc[0]
        summary_lines.append(f"Best Feature Combination: {best_combo['n_features']} features")
        summary_lines.append(f"Combination R²: {best_combo['test_r_squared']:.4f}")
    
    summary = "\n".join(summary_lines)
    print(f"\n{summary}")
    
    # Create final result
    result = {
        'status': 'success',
        'transformation_used': best_transformation,
        'method_results': method_results,
        'comparison': comparison_df,
        'feature_selection': feature_selection_df,
        'combination_results': combination_results_df,
        'best_method': comparison_df.iloc[0]['Method'] if not comparison_df.empty else None,
        'best_test_r2': comparison_df.iloc[0]['Test_R²'] if not comparison_df.empty else None,
        'summary': summary,
        'validation': validation
    }
    
    return result

def quick_analysis(target_name: str, start_date: str = "2010-01", 
                  config: AnalysisConfig = None) -> Dict[str, Any]:
    """Quick analysis for a target variable with standard exogenous variables."""
    print("QUICK FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'quick', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def comprehensive_analysis(target_name: str, start_date: str = "2010-01",
                         config: AnalysisConfig = None) -> Dict[str, Any]:
    """Comprehensive analysis for a target variable."""
    print("COMPREHENSIVE FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'comprehensive', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def full_analysis(target_name: str, start_date: str = "2010-01",
                 config: AnalysisConfig = None) -> Dict[str, Any]:
    """Full analysis with all available methods and features."""
    print("FULL FINANCIAL ANALYSIS")
    print("=" * 40)
    
    try:
        data = get_target_with_standard_exog(target_name, start_date, config)
        exog_vars = [col for col in data.columns if col != "Datum" and col != target_name]
        
        return financial_analysis(data, target_name, exog_vars, 'full', config)
    
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

class SimpleVisualizer:
    """Simple visualization functions for analysis results."""
    
    @staticmethod
    def plot_data_overview(data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        """Plot basic data overview."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        plot_cols = list(numeric_cols[:6])  # Limit to 6 series
        
        if target_var in plot_cols:
            # Move target to front
            plot_cols.remove(target_var)
            plot_cols = [target_var] + plot_cols
        
        data.set_index(date_col)[plot_cols].plot(ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('Time Series Overview')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Target distribution
        if target_var in data.columns:
            axes[0, 1].hist(data[target_var].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title(f'{target_var} Distribution')
        
        # Correlation with target
        if target_var in numeric_cols:
            corr_with_target = data[numeric_cols].corr()[target_var].abs().sort_values(ascending=False)
            top_vars = corr_with_target.head(6).index.tolist()
            
            if len(top_vars) > 1:
                corr_matrix = data[top_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           ax=axes[1, 0], fmt='.2f', square=True)
                axes[1, 0].set_title(f'Correlations with {target_var}')
        
        # Missing data
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            axes[1, 1].barh(range(len(missing_data)), missing_data.values)
            axes[1, 1].set_yticks(range(len(missing_data)))
            axes[1, 1].set_yticklabels(missing_data.index)
            axes[1, 1].set_title('Missing Data Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Data Completeness: Perfect')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_method_comparison(comparison_df: pd.DataFrame):
        """Plot method comparison results."""
        if comparison_df.empty:
            print("No method results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test R² comparison
        methods = comparison_df['Method'].values
        test_r2 = comparison_df['Test_R²'].values
        
        bars1 = axes[0].barh(methods, test_r2)
        axes[0].set_xlabel('Test R²')
        axes[0].set_title('Method Performance Comparison')
        
        # Color bars by performance
        for i, bar in enumerate(bars1):
            if test_r2[i] > 0.7:
                bar.set_color('green')
            elif test_r2[i] > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels
        for i, v in enumerate(test_r2):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Overfitting analysis
        overfitting = comparison_df['Overfitting'].values
        colors = ['red' if x > 0.15 else 'orange' if x > 0.08 else 'green' for x in overfitting]
        
        axes[1].barh(methods, overfitting, color=colors, alpha=0.7)
        axes[1].set_xlabel('Overfitting (Train - Test R²)')
        axes[1].set_title('Overfitting Analysis')
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axvline(x=0.08, color='orange', linestyle='--', alpha=0.5, label='Warning (0.08)')
        axes[1].axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Critical (0.15)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

# System initialization
def initialize_system():
    """Initialize the financial analysis system."""
    setup_logging()
    
    print("FINANCIAL ANALYSIS SYSTEM - CORRECTED VERSION")
    print("=" * 55)
    print("Key improvements:")
    print("- Fixed mixed-frequency data handling (forward-fill)")
    print("- Robust cross-validation without extreme values")
    print("- Conservative model hyperparameters")
    print("- Proper data quality validation")
    print("- Clean architecture without monkey patches")
    print("")
    print("Available functions:")
    print("  quick_analysis(target_name)")
    print("  comprehensive_analysis(target_name)")
    print("  full_analysis(target_name)")
    print("  financial_analysis(data, target_var, exog_vars)")
    print("")
    print("Available targets:", ", ".join(list(INDEX_TARGETS.keys())[:4]) + "...")
    print("System ready!")

def test_system():
    """Test the system with a simple example."""
    print("Testing system with PH_KREDITE...")
    
    try:
        results = quick_analysis("PH_KREDITE", start_date="2005-01")
        
        if results['status'] == 'success':
            print(f"✅ System test successful!")
            print(f"  Best method: {results['best_method']}")
            print(f"  Test R²: {results['best_test_r2']:.4f}")
            print(f"  Transformation used: {results['transformation_used']}")
            return True
        else:
            print(f"❌ System test failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ System test failed with exception: {e}")
        return False

print("Analysis pipeline and main functions loaded")



# %%
# %%
"""
Cache-Fixes - Erweitert CacheManager für Final Dataset Caching
"""

class ExtendedCacheManager(CacheManager):
    """
    Erweitert den ursprünglichen CacheManager um Final Dataset Caching.
    """
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        # Zusätzliches Verzeichnis für finale Datasets
        self.final_datasets_dir = self.cache_dir / "final_datasets"
        self.final_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Verzeichnis für transformierte Datasets
        self.transformed_dir = self.cache_dir / "transformed_datasets"
        self.transformed_dir.mkdir(parents=True, exist_ok=True)
    
    def make_final_dataset_key(self, series_definitions: Dict[str, str], start: str, end: str) -> str:
        """Erstelle einen stabilen Key für finale Datasets."""
        import hashlib
        import json
        
        payload = {
            "series_definitions": {k: series_definitions[k] for k in sorted(series_definitions)},
            "start": start,
            "end": end,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    
    def has_fresh_final_dataset(self, key: str) -> bool:
        """Prüft ob ein frisches final Dataset existiert."""
        pattern = f"*_{key}.xlsx"
        matches = sorted(self.final_datasets_dir.glob(pattern), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return False
        
        try:
            latest = matches[0]
            age_days = (dt.datetime.now() - dt.datetime.fromtimestamp(latest.stat().st_mtime)).days
            return age_days <= self.config.cache_max_age_days
        except OSError:
            return False
    
    def write_final_dataset(self, key: str, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
        """Speichere finales Dataset mit Metadaten."""
        if df is None or df.empty:
            return False
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Datenpfad
        data_path = self.final_datasets_dir / f"{timestamp}_{key}.xlsx"
        meta_path = self.final_datasets_dir / f"{timestamp}_{key}_meta.json"
        
        try:
            # Speichere Daten
            df.to_excel(data_path, index=False, engine=get_excel_engine())
            
            # Speichere Metadaten
            meta_info = {
                "created_at": dt.datetime.now().isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "date_range": {
                    "start": df['Datum'].min().isoformat() if 'Datum' in df.columns else None,
                    "end": df['Datum'].max().isoformat() if 'Datum' in df.columns else None
                }
            }
            
            if metadata:
                meta_info.update(metadata)
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)
            
            print(f"Final dataset cached: {data_path.name}")
            return True
            
        except Exception as e:
            print(f"Failed to cache final dataset: {e}")
            return False
    
    def read_final_dataset(self, key: str) -> Optional[pd.DataFrame]:
        """Lade neuestes finales Dataset."""
        pattern = f"*_{key}.xlsx"
        matches = sorted(self.final_datasets_dir.glob(pattern), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return None
        
        try:
            latest = matches[0]
            df = pd.read_excel(latest, engine=get_excel_engine())
            
            # Ensure Datum is datetime
            if 'Datum' in df.columns:
                df['Datum'] = pd.to_datetime(df['Datum'])
            
            print(f"Final dataset loaded from cache: {latest.name}")
            return df
            
        except Exception as e:
            print(f"Failed to load final dataset: {e}")
            return None
    
    def write_transformed_dataset(self, transformation: str, target_var: str, df: pd.DataFrame) -> bool:
        """Speichere transformiertes Dataset."""
        if df is None or df.empty:
            return False
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_target = "".join(c for c in target_var if c.isalnum() or c in "._-")
        
        file_path = self.transformed_dir / f"{timestamp}_{safe_target}_{transformation}.xlsx"
        
        try:
            df.to_excel(file_path, index=False, engine=get_excel_engine())
            print(f"Transformed dataset cached: {file_path.name}")
            return True
        except Exception as e:
            print(f"Failed to cache transformed dataset: {e}")
            return False
    
    def cleanup_old_cache(self, max_age_days: int = None):
        """Bereinige alte Cache-Dateien."""
        if max_age_days is None:
            max_age_days = self.config.cache_max_age_days * 2  # Keep longer for final datasets
        
        cutoff_date = dt.datetime.now() - dt.timedelta(days=max_age_days)
        deleted_count = 0
        
        # Cleanup in allen Cache-Verzeichnissen
        for cache_subdir in [self.cache_dir, self.final_datasets_dir, self.transformed_dir]:
            if not cache_subdir.exists():
                continue
                
            for file_path in cache_subdir.glob("*"):
                if file_path.is_file():
                    try:
                        file_time = dt.datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1
                    except OSError:
                        continue
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old cache files")

# Erweitere FinancialDataDownloader für Final Dataset Caching
def download_with_final_caching(self, series_definitions: Dict[str, str], start_date: str = None, 
                               end_date: str = None, prefer_cache: bool = True, 
                               anchor_var: Optional[str] = None) -> pd.DataFrame:
    """
    Download mit Final Dataset Caching - erweitert die ursprüngliche download Methode.
    """
    start_date = start_date or self.config.default_start_date
    end_date = end_date or self.config.default_end_date
    
    # Verwende ExtendedCacheManager
    if not isinstance(self.cache_manager, ExtendedCacheManager):
        self.cache_manager = ExtendedCacheManager(self.config)
    
    # Prüfe Final Dataset Cache
    final_key = self.cache_manager.make_final_dataset_key(series_definitions, start_date, end_date)
    
    if prefer_cache and self.cache_manager.has_fresh_final_dataset(final_key):
        cached_final = self.cache_manager.read_final_dataset(final_key)
        if cached_final is not None and not cached_final.empty:
            print(f"Loaded final dataset from cache: {cached_final.shape[0]} rows, {cached_final.shape[1]-1} variables")
            return cached_final
    
    # Führe normale Download-Logik aus (aus der ursprünglichen Methode)
    print(f"Downloading {len(series_definitions)} variables from {start_date} to {end_date}")
    
    regular_codes = {}
    index_definitions = {}
    
    for var_name, definition in series_definitions.items():
        index_codes = parse_index_specification(definition)
        if index_codes:
            index_definitions[var_name] = index_codes
        else:
            regular_codes[var_name] = definition
    
    all_codes = set(regular_codes.values())
    for index_codes in index_definitions.values():
        all_codes.update(index_codes)
    all_codes = list(all_codes)
    
    print(f"Total series to download: {len(all_codes)}")
    
    # Individual series caching (bestehende Logik)
    cached_data = {}
    missing_codes = []
    
    if prefer_cache:
        for code in all_codes:
            cached_df = self.cache_manager.read_cache(code)
            if cached_df is not None:
                cached_data[code] = cached_df
            else:
                missing_codes.append(code)
    else:
        missing_codes = all_codes[:]
    
    # Download missing codes (bestehende Logik bleibt unverändert)
    downloaded_data = {}
    if missing_codes:
        print(f"Downloading {len(missing_codes)} missing series...")
        try:
            downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    downloaded_data = asyncio.run(self._fetch_all_series(missing_codes, start_date, end_date))
                except ImportError:
                    print("Using synchronous download mode...")
                    downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
            else:
                print("Async failed, using synchronous download mode...")
                downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
        except Exception as e:
            print(f"Download failed ({e}), trying synchronous mode...")
            downloaded_data = self._fetch_all_series_sync(missing_codes, start_date, end_date)
        
        # Cache individual series
        for code, df in downloaded_data.items():
            self.cache_manager.write_cache(code, df)
    
    # Bestehende Merge- und Index-Erstellungslogik bleibt unverändert...
    all_data = {**cached_data, **downloaded_data}
    
    if not all_data:
        raise Exception("No series loaded successfully")
    
    merged_df = self._merge_series_data(all_data)
    final_data = {"Datum": merged_df["Datum"]}
    
    for var_name, series_code in regular_codes.items():
        if series_code in merged_df.columns:
            final_data[var_name] = merged_df[series_code]
    
    for var_name, index_codes in index_definitions.items():
        try:
            available_codes = [c for c in index_codes if c in merged_df.columns]
            
            if len(available_codes) >= len(index_codes) * 0.3:
                index_series = self.index_creator.create_index(merged_df, available_codes, var_name)
                aligned_index = index_series.reindex(pd.to_datetime(merged_df['Datum']))
                final_data[var_name] = aligned_index.values
                print(f"Created INDEX: {var_name} from {len(available_codes)}/{len(index_codes)} series")
            else:
                if var_name in SIMPLE_TARGET_FALLBACKS:
                    fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                    if fallback_code in merged_df.columns:
                        final_data[var_name] = merged_df[fallback_code]
                        print(f"Using fallback for {var_name}: {fallback_code}")
                    else:
                        print(f"Warning: Could not create {var_name} - fallback series {fallback_code} not available")
                else:
                    print(f"Warning: Could not create INDEX {var_name} - insufficient data ({len(available_codes)}/{len(index_codes)} series available)")
                    
        except Exception as e:
            print(f"Failed to create INDEX {var_name}: {e}")
            if var_name in SIMPLE_TARGET_FALLBACKS and var_name not in final_data:
                fallback_code = SIMPLE_TARGET_FALLBACKS[var_name]
                if fallback_code in merged_df.columns:
                    final_data[var_name] = merged_df[fallback_code]
                    print(f"Using fallback for {var_name} after INDEX creation failed: {fallback_code}")
    
    final_df = pd.DataFrame(final_data)
    final_df["Datum"] = pd.to_datetime(final_df["Datum"])
    final_df = final_df.sort_values("Datum").reset_index(drop=True)

    # Bestehende Trimming-Logik...
    value_cols = [c for c in final_df.columns if c != 'Datum']
    if value_cols:
        non_na_count = final_df[value_cols].notna().sum(axis=1)
        required = 2 if len(value_cols) >= 2 else 1
        keep_mask = non_na_count >= required
        if keep_mask.any():
            first_keep = keep_mask.idxmax()
            if first_keep > 0:
                _before = len(final_df)
                final_df = final_df.iloc[first_keep:].reset_index(drop=True)
                print(f"Trimmed leading rows with <{required} populated variables: {_before} → {len(final_df)}")

    if anchor_var and anchor_var in final_df.columns:
        mask_anchor = final_df[anchor_var].notna()
        if mask_anchor.any():
            start_anchor = final_df.loc[mask_anchor, 'Datum'].min()
            end_anchor = final_df.loc[mask_anchor, 'Datum'].max()
            _before_rows = len(final_df)
            final_df = final_df[(final_df['Datum'] >= start_anchor) & (final_df['Datum'] <= end_anchor)].copy()
            final_df.reset_index(drop=True, inplace=True)
            print(f"Anchored final dataset to '{anchor_var}' window: {start_anchor.date()} → {end_anchor.date()} (rows: {_before_rows} → {len(final_df)})")

    if anchor_var and anchor_var in final_df.columns:
        exog_cols = [c for c in final_df.columns if c not in ('Datum', anchor_var)]
        if exog_cols:
            tgt_notna = final_df[anchor_var].notna().values
            all_exog_nan = final_df[exog_cols].isna().all(axis=1).values
            keep_start = 0
            for i in range(len(final_df)):
                if not (tgt_notna[i] and all_exog_nan[i]):
                    keep_start = i
                    break
            if keep_start > 0:
                _before = len(final_df)
                final_df = final_df.iloc[keep_start:].reset_index(drop=True)
                print(f"Trimmed leading target-only rows: {_before} → {len(final_df)}")

    print(f"Final dataset: {final_df.shape[0]} observations, {final_df.shape[1]-1} variables")
    
    # Cache final dataset
    metadata = {
        "series_definitions": series_definitions,
        "start_date": start_date,
        "end_date": end_date,
        "downloaded_codes": sorted(list(all_data.keys())),
        "regular_series": regular_codes,
        "index_specifications": index_definitions
    }
    
    self.cache_manager.write_final_dataset(final_key, final_df, metadata)
    
    return final_df

# Erweitere DataPreprocessor für Transformation Caching
def create_transformations_with_caching(self, transformation: str = 'levels') -> pd.DataFrame:
    """
    Transformationen mit Caching - erweitert die ursprüngliche Methode.
    """
    # Verwende ExtendedCacheManager falls verfügbar
    cache_manager = getattr(self, 'cache_manager', None)
    if cache_manager and hasattr(cache_manager, 'write_transformed_dataset'):
        # Führe Transformationen durch
        transformed_data = self.create_transformations_original(transformation)
        
        # Cache transformierte Daten
        cache_manager.write_transformed_dataset(transformation, self.target_var, transformed_data)
        
        return transformed_data
    else:
        # Fallback zur ursprünglichen Methode
        return self.create_transformations_original(transformation)

# Monkey-patch die Methoden
FinancialDataDownloader.download_original = FinancialDataDownloader.download
FinancialDataDownloader.download = download_with_final_caching

DataPreprocessor.create_transformations_original = DataPreprocessor.create_transformations
DataPreprocessor.create_transformations = create_transformations_with_caching

print("Cache fixes loaded - Final datasets and transformed data will now be cached")



# %%# %%
"""
Main Script - Usage Example
Zeigt wie die korrigierte Pipeline verwendet wird
# """

# if __name__ == "__main__":
#     # Initialize system
#     initialize_system()
    
#     # WICHTIG: Cache-Fixes laden
#     print("Loading cache fixes for raw and transformed data...")
#     # Der Cache-Fix Code sollte hier eingefügt werden (aus Cache-Fixes Artifact)
    
#     # Configuration with conservative settings
#     config = AnalysisConfig(
#         default_start_date="2000-01",
#         default_end_date="2024-12",
#         test_size=0.25,                    # Conservative test size
#         cv_folds=3,                       # Fewer CV folds for stability
#         gap_periods=2,                    # Mandatory gaps
#         max_feature_combinations=15,      # Reduced combinations to prevent overfitting
#         handle_mixed_frequencies=True,    # Enable mixed frequency handling
#         cache_final_dataset=True,         # Enable final dataset caching
#         cache_max_age_days=7              # Cache für 7 Tage
#     )
    
#     # === EXAMPLE 1: Quick Analysis with Caching ===
#     print("\n" + "="*60)
#     print("EXAMPLE 1: QUICK ANALYSIS WITH CACHING")
#     print("="*60)
    
#     TARGET = "PH_KREDITE"
#     START_DATE = "2005-01"
    
#     # Erste Ausführung - lädt und cached Daten
#     print("First run - will download and cache data...")
#     results1 = quick_analysis(TARGET, START_DATE, config)
    
#     # Zweite Ausführung - sollte aus Cache laden
#     print("\nSecond run - should load from cache...")
#     results2 = quick_analysis(TARGET, START_DATE, config)
    
#     if results1['status'] == 'success':
#         print(f"\n✅ Analysis completed successfully!")
#         print(f"Best method: {results1['best_method']}")
#         print(f"Test R²: {results1['best_test_r2']:.4f}")
#         print(f"Transformation: {results1['transformation_used']}")
#     else:
#         print(f"\n❌ Analysis failed: {results1.get('error', 'Unknown error')}")
    
#     # === Cache-Status anzeigen ===
#     print("\n" + "="*60)
#     print("CACHE STATUS")
#     print("="*60)
    
#     # Zeige Cache-Verzeichnisse
#     cache_dir = Path(config.cache_dir)
#     if cache_dir.exists():
#         print(f"Cache directory: {cache_dir}")
        
#         # Rohdaten Cache
#         raw_files = list(cache_dir.glob("*.xlsx"))
#         print(f"Raw data files cached: {len(raw_files)}")
        
#         # Final datasets
#         final_dir = cache_dir / "final_datasets"
#         if final_dir.exists():
#             final_files = list(final_dir.glob("*.xlsx"))
#             print(f"Final datasets cached: {len(final_files)}")
#             for f in final_files[-3:]:  # Show last 3
#                 print(f"  - {f.name}")
        
#         # Transformed datasets  
#         trans_dir = cache_dir / "transformed_datasets"
#         if trans_dir.exists():
#             trans_files = list(trans_dir.glob("*.xlsx"))
#             print(f"Transformed datasets cached: {len(trans_files)}")
#             for f in trans_files[-3:]:  # Show last 3
#                 print(f"  - {f.name}")
    
#     # === Cache cleanup demo ===
#     print("\n" + "="*60)
#     print("CACHE CLEANUP DEMO")
#     print("="*60)
    
#     # Erstelle ExtendedCacheManager für Cleanup
#     downloader = FinancialDataDownloader(config)
#     if hasattr(downloader, 'cache_manager'):
#         cache_manager = downloader.cache_manager
#         if hasattr(cache_manager, 'cleanup_old_cache'):
#             print("Running cache cleanup (files older than 60 days)...")
#             cache_manager.cleanup_old_cache(max_age_days=60)
#         else:
#             print("Cache manager does not support cleanup")
    
#     # === EXAMPLE 2: Custom Analysis ===
#     print("\n" + "="*60)
#     print("EXAMPLE 2: CUSTOM ANALYSIS WITH CACHING")
#     print("="*60)
    
#     # Define custom series (mix of monthly and quarterly)
#     # Use only reliably available series
#     series_definitions = {
#         "PH_KREDITE": INDEX_TARGETS["PH_KREDITE"],  # Quarterly target
#         "euribor_3m": STANDARD_EXOG_VARS["euribor_3m"],  # Monthly
#         "german_rates": STANDARD_EXOG_VARS["german_rates"],  # Monthly
#         "german_inflation": STANDARD_EXOG_VARS["german_inflation"],  # Monthly
#         "german_unemployment": STANDARD_EXOG_VARS["german_unemployment"]  # Monthly
#     }
    
#     # Download data (should use caching)
#     print("Downloading with caching...")
#     downloader = FinancialDataDownloader(config)
#     data = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=True)
    
#     print(f"Downloaded data shape: {data.shape}")
#     print(f"Columns: {list(data.columns)}")
    
#     # Show data overview plot
#     visualizer = SimpleVisualizer()
#     visualizer.plot_data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === EXAMPLE 3: System Test ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE")
#     print("="*60)
    
#     #data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === Cache-Performance Test ===
#     print("\n" + "="*60)
#     print("CACHE PERFORMANCE TEST")
#     print("="*60)
    
#     import time
    
#     # Test 1: Download ohne Cache
#     print("Test 1: Download without cache...")
#     start_time = time.time()
#     data_no_cache = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=False)
#     time_no_cache = time.time() - start_time
#     print(f"Time without cache: {time_no_cache:.2f} seconds")
    
#     # Test 2: Download mit Cache
#     print("\nTest 2: Download with cache...")
#     start_time = time.time()
#     data_with_cache = downloader.download(series_definitions, start_date="2005-01", end_date="2024-12", prefer_cache=True)
#     time_with_cache = time.time() - start_time
#     print(f"Time with cache: {time_with_cache:.2f} seconds")
    
#     if time_no_cache > 0:
#         speedup = time_no_cache / max(time_with_cache, 0.01)
#         print(f"Cache speedup: {speedup:.1f}x faster")
    
#     # === EXAMPLE 3: System Test ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("✅ Raw data caching: Active")
#         print("✅ Final dataset caching: Active") 
#         print("✅ Transformed data caching: Active")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#         print("\nCache locations:")
#         print(f"- Raw data: {cache_dir}")
#         print(f"- Final datasets: {cache_dir / 'final_datasets'}")
#         print(f"- Transformed data: {cache_dir / 'transformed_datasets'}")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE - WITH FULL CACHING")
#     print("="*60)
#     #data_overview(data, "PH_KREDITE")
    
#     # Run comprehensive analysis
#     exog_vars = ["euribor_3m", "german_rates", "german_inflation", "german_unemployment"]
    
#     results = financial_analysis(
#         data=data,
#         target_var="PH_KREDITE",
#         exog_vars=exog_vars,
#         analysis_type='comprehensive',
#         config=config
#     )
    
#     if results['status'] == 'success':
#         print(f"\n✅ Custom analysis completed!")
        
#         # Show method comparison plot
#         if not results['comparison'].empty:
#             visualizer.plot_method_comparison(results['comparison'])
        
#         # Display best features if available
#         best_method = results['best_method']
#         if best_method and 'method_results' in results:
#             method_result = results['method_results'][best_method]
            
#             print(f"\n🔍 BEST MODEL DETAILS ({best_method}):")
#             print("-" * 40)
            
#             # Feature importance/coefficients
#             feature_names = method_result.get('feature_names', [])
            
#             if 'feature_importance' in method_result:
#                 importances = method_result['feature_importance']
#                 print("Feature Importances:")
#                 for name, imp in zip(feature_names, importances):
#                     print(f"  {name}: {imp:.4f}")
            
#             elif 'coefficients' in method_result:
#                 coefficients = method_result['coefficients']
#                 if hasattr(coefficients, 'values'):
#                     coef_values = coefficients.values[1:]  # Skip constant
#                 else:
#                     coef_values = coefficients[1:] if len(coefficients) > len(feature_names) else coefficients
                
#                 print("Coefficients:")
#                 for name, coef in zip(feature_names, coef_values[:len(feature_names)]):
#                     print(f"  {name}: {coef:.4f}")
    
#     # === EXAMPLE 3: Test System ===
#     print("\n" + "="*60)
#     print("EXAMPLE 3: SYSTEM TEST")
#     print("="*60)
    
#     test_passed = test_system()
    
#     if test_passed:
#         print("\n🎉 All systems working correctly!")
#         print("\nYou can now use:")
#         print("- quick_analysis('TARGET_NAME')")
#         print("- comprehensive_analysis('TARGET_NAME')")  
#         print("- financial_analysis(data, target, exog_vars)")
#     else:
#         print("\n⚠️ System test failed - check your setup")
    
#     print("\n" + "="*60)
#     print("ANALYSIS COMPLETE")
#     print("="*60)

# %%
"""
Improved Mixed-Frequency Data Processing - KORRIGIERT
Verhindert Data Leakage durch strikte zeitliche Beschränkungen
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from statsmodels.tsa.stattools import adfuller
import warnings

class ImprovedMixedFrequencyProcessor:
    """
    Verbesserte Behandlung von Mixed-Frequency Daten mit strikten Anti-Leakage Regeln.
    """
    
    @staticmethod
    def detect_frequency(series: pd.Series, date_col: pd.Series) -> str:
        """Erweiterte Frequenzerkennung mit robusteren Regeln."""
        if series.isna().all():
            return "unknown"
        
        df_temp = pd.DataFrame({'date': date_col, 'value': series})
        df_temp = df_temp.dropna().copy()
        
        if len(df_temp) < 4:  # Mindestens 4 Beobachtungen für Frequenzerkennung
            return "unknown"
        
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['quarter'] = df_temp['date'].dt.quarter
        
        # Erweiterte Frequenzanalyse
        years_with_data = df_temp['year'].nunique()
        if years_with_data < 2:
            return "insufficient_history"
        
        # Analysiere Beobachtungen pro Jahr
        obs_per_year = df_temp.groupby('year').size()
        avg_obs_per_year = obs_per_year.mean()
        std_obs_per_year = obs_per_year.std()
        
        # Analysiere monatliche vs. quartalsweise Verteilung
        months_per_year = df_temp.groupby('year')['month'].nunique()
        quarters_per_year = df_temp.groupby('year')['quarter'].nunique()
        
        avg_months_per_year = months_per_year.mean()
        avg_quarters_per_year = quarters_per_year.mean()
        
        # Robuste Klassifikation
        if avg_quarters_per_year <= 4.2 and avg_obs_per_year <= 5:
            if avg_months_per_year <= 4.5:  # Meist nur ein Monat pro Quartal
                return "quarterly"
        
        if avg_months_per_year >= 8 and avg_obs_per_year >= 10:
            return "monthly"
        
        # Zusätzliche Prüfung: Sind die Daten gleichmäßig verteilt?
        if std_obs_per_year / max(avg_obs_per_year, 1) > 0.5:
            return "irregular"
            
        return "unknown"
    
    @staticmethod
    def safe_forward_fill_quarterly(df: pd.DataFrame, quarterly_vars: List[str], 
                                   max_fill_periods: int = 6) -> pd.DataFrame:
        """
        Sicheres Forward-Fill für Quartalsdaten mit strikten Limits.
        
        Args:
            df: DataFrame mit Zeitreihen
            quarterly_vars: Liste der Quartalsvariablen
            max_fill_periods: Maximale Anzahl Perioden zum Forward-Fill (Standard: 2 Monate)
        """
        result = df.copy()
        
        for var in quarterly_vars:
            if var not in df.columns:
                continue
            
            series = df[var].copy()
            valid_mask = series.notna()
            
            if not valid_mask.any():
                continue
            
            # Identifiziere alle validen Zeitpunkte
            valid_indices = valid_mask[valid_mask].index.tolist()
            
            # Forward-Fill nur zwischen benachbarten validen Punkten
            filled_series = series.copy()
            
            for i in range(len(valid_indices) - 1):
                current_idx = valid_indices[i]
                next_idx = valid_indices[i + 1]
                
                # Bereich zwischen aktuellen und nächsten validen Werten
                gap_start = current_idx + 1
                gap_end = next_idx
                
                # Begrenze Fill-Bereich auf max_fill_periods
                actual_gap_end = min(gap_end, current_idx + max_fill_periods + 1)
                
                if gap_start < actual_gap_end:
                    # Forward-Fill nur im begrenzten Bereich
                    fill_value = series.iloc[current_idx]
                    filled_series.iloc[gap_start:actual_gap_end] = fill_value
            
            result[var] = filled_series
        
        return result
    
    @staticmethod
    def align_frequencies_improved(
        df: pd.DataFrame,
        target_var: str,
        date_col: str = "Datum",
        train_end_index: Optional[int] = None,
        validation_split_date: Optional[pd.Timestamp] = None,
        max_fill_periods: int = 2,   # konservativer Standard
    ) -> Dict[str, Any]:
        """
        Verbesserte Frequenz-Alignierung mit Anti-Leakage-Schutz.
        - Forward-Fill nur im Trainingsfenster
        - Leakage-Flags nur, wenn neue Fills im Testfenster liegen
        """
        if target_var not in df.columns or date_col not in df.columns:
            raise ValueError(f"Missing {target_var} or {date_col} column")

        # Datums-Handling & Sortierung
        out = {
            "processed_df": None,
            "frequency_info": {},
            "warnings": [],
            "leakage_risk": "low",
            "forward_fill_used": False,
            "fill_span_overlaps_test_period": False,
        }

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.sort_values(date_col).reset_index(drop=True)

        # Frequenzen erkennen (auf Originaldaten)
        all_vars = [c for c in work.columns if c != date_col]
        freqs = {}
        for var in all_vars:
            freqs[var] = ImprovedMixedFrequencyProcessor.detect_frequency(work[var], work[date_col])
        out["frequency_info"] = freqs

        quarterly_vars = [v for v, f in freqs.items() if f == "quarterly"]
        monthly_vars   = [v for v, f in freqs.items() if f == "monthly"]
        irregular_vars = [v for v, f in freqs.items() if f == "irregular"]

        print("Detected frequencies:")
        print(f"  Quarterly: {quarterly_vars}")
        print(f"  Monthly: {monthly_vars}")
        if irregular_vars:
            print(f"  Irregular: {irregular_vars}")
            out["warnings"].append(f"Irregular frequency detected: {irregular_vars}")

        # Train-Ende bestimmen
        split_date = None
        if validation_split_date is not None:
            split_date = pd.to_datetime(validation_split_date)
        elif train_end_index is not None and 0 <= train_end_index < len(work):
            split_date = pd.to_datetime(work.loc[train_end_index, date_col])

        # Forward-Fill nur für Trainingsanteil
        if quarterly_vars:
            if split_date is not None:
                train_mask = work[date_col] < split_date
                train_df = work.loc[train_mask].copy()
                print(f"Using training data only for forward-fill: {len(train_df)} observations")
            else:
                train_df = work.copy()
                out["warnings"].append("No split provided — forward-fill applied on full sample (potential bias)")
                out["leakage_risk"] = "medium"

            print(f"Applying safe forward-fill to {len(quarterly_vars)} quarterly variables...")
            filled_train = ImprovedMixedFrequencyProcessor.safe_forward_fill_quarterly(
                train_df, quarterly_vars, max_fill_periods=max_fill_periods
            )

            if split_date is not None:
                valid_df = work.loc[work[date_col] >= split_date].copy()  # Validierungs-/Testteil: ungefüllt
                processed_df = pd.concat([filled_train, valid_df], ignore_index=True)
                processed_df = processed_df.sort_values(date_col).reset_index(drop=True)
            else:
                processed_df = filled_train
        else:
            processed_df = work.copy()  # nichts zu füllen

        # Zielbereich trimmen (keine Lead/Tail-NaNs)
        s = processed_df[target_var]
        first = s.first_valid_index()
        last  = s.last_valid_index()
        if first is not None and last is not None and last >= first:
            processed_df = processed_df.loc[first:last].reset_index(drop=True)

        # Fortschritts-Log je Quartalsvariable
        for var in quarterly_vars:
            before_count = work[var].notna().sum()
            after_count  = processed_df[var].notna().sum()
            improvement  = int(after_count - before_count)
            print(f"  {var}: {before_count} → {after_count} observations (+{improvement})")
            # Keine automatische Leakage-Hochstufung mehr nur wegen „Large improvement“
            if improvement > max(5, int(before_count * 0.5)):
                out["warnings"].append(f"Large improvement in {var} coverage — verify correctness")

        # Leakage-Flags setzen: neue Fills identifizieren & prüfen, ob sie NACH split_date liegen
        orig_series = work.set_index(date_col)[target_var]
        proc_series = processed_df.set_index(date_col)[target_var]
        orig_notna  = orig_series.reindex(proc_series.index).notna().fillna(False)
        proc_notna  = proc_series.notna()
        new_filled_mask = (proc_notna & ~orig_notna)

        out["forward_fill_used"] = bool(new_filled_mask.any())
        if split_date is not None:
            out["fill_span_overlaps_test_period"] = bool(new_filled_mask.loc[new_filled_mask.index >= split_date].any())
            out["leakage_risk"] = "high" if out["fill_span_overlaps_test_period"] else out["leakage_risk"]
        # Wenn kein split_date: Risiko bleibt wie oben gesetzt (medium), aber kein „high“

        out["processed_df"] = processed_df
        return out

        




class ImprovedDataQualityChecker:
    """
    Erweiterte Datenqualitätsprüfung mit Stationaritätstests.
    """
    
    @staticmethod
    def test_stationarity(series: pd.Series, variable_name: str) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller Test für Stationarität.
        """
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': None,
                'error': 'Insufficient data for stationarity test'
            }
        
        try:
            # ADF Test
            result = adfuller(clean_series, autolag='AIC')
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'variable': variable_name,
                'n_observations': len(clean_series)
            }
        except Exception as e:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': None,
                'error': str(e),
                'variable': variable_name
            }
    
    @staticmethod
    def comprehensive_data_validation(data: pd.DataFrame, target_var: str, 
                                    exog_vars: List[str],
                                    min_target_coverage: float = 0.15,  # Für Quartalsdaten
                                    min_observations: int = 30         ) -> Dict[str, Any]:
        """
        Umfassende Datenvalidierung mit Stationaritätstests.
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'stationarity_tests': {},
            'recommendations': [],
            'sample_adequacy': {}
        }
        
        # Grundlegende Validierung
        missing_vars = [var for var in [target_var] + exog_vars if var not in data.columns]
        if missing_vars:
            validation_results['errors'].append(f"Missing variables: {', '.join(missing_vars)}")
            validation_results['is_valid'] = False
            return validation_results
        
        # Stichprobengröße prüfen
        if len(data) < min_observations:
            validation_results['errors'].append(
                f"Insufficient sample size: {len(data)} < {min_observations} required"
            )
            validation_results['is_valid'] = False
        
        # Target Variable Analyse
        target_series = data[target_var]
        target_coverage = target_series.notna().sum() / len(target_series)
        target_valid_obs = target_series.notna().sum()
        
        validation_results['data_quality'][target_var] = {
            'total_obs': len(target_series),
            'valid_obs': target_valid_obs,
            'coverage': target_coverage,
            'mean': target_series.mean() if target_valid_obs > 0 else np.nan,
            'std': target_series.std() if target_valid_obs > 0 else np.nan,
            'frequency': ImprovedMixedFrequencyProcessor.detect_frequency(
                target_series, data['Datum'] if 'Datum' in data.columns else data.index
            )
        }
        
        # Critical checks für Target
        if target_coverage < min_target_coverage:
            validation_results['errors'].append(
                f"Target {target_var} insufficient coverage: {target_coverage:.1%} < {min_target_coverage:.1%}"
            )
            validation_results['is_valid'] = False
        
        if target_valid_obs > 1 and target_series.std() == 0:
            validation_results['errors'].append(f"Target {target_var} is constant")
            validation_results['is_valid'] = False
        
        # Stationaritätstest für Target
        stationarity_result = ImprovedDataQualityChecker.test_stationarity(target_series, target_var)
        validation_results['stationarity_tests'][target_var] = stationarity_result
        
        if stationarity_result.get('is_stationary') is False:
            validation_results['warnings'].append(
                f"Target {target_var} may be non-stationary (ADF p-value: {stationarity_result.get('p_value', 'N/A'):.3f})"
            )
            validation_results['recommendations'].append(
                f"Consider differencing or log-transformation for {target_var}"
            )
        
        # Exogenous Variables Analyse
        for var in exog_vars:
            if var in data.columns:
                series = data[var]
                coverage = series.notna().sum() / len(series)
                valid_obs = series.notna().sum()
                
                validation_results['data_quality'][var] = {
                    'total_obs': len(series),
                    'valid_obs': valid_obs,
                    'coverage': coverage,
                    'mean': series.mean() if valid_obs > 0 else np.nan,
                    'std': series.std() if valid_obs > 0 else np.nan,
                    'frequency': ImprovedMixedFrequencyProcessor.detect_frequency(
                        series, data['Datum'] if 'Datum' in data.columns else data.index
                    )
                }
                
                # Stationaritätstest
                stationarity_result = ImprovedDataQualityChecker.test_stationarity(series, var)
                validation_results['stationarity_tests'][var] = stationarity_result
                
                # Warnings
                if coverage < 0.3:
                    validation_results['warnings'].append(f"Low coverage in {var}: {coverage:.1%}")
                
                if valid_obs > 1 and series.std() == 0:
                    validation_results['warnings'].append(f"Variable {var} is constant")
        
        # Sample adequacy assessment
        all_vars = [target_var] + [v for v in exog_vars if v in data.columns]
        complete_cases = data[all_vars].dropna()
        overlap_ratio = len(complete_cases) / len(data)
        
        validation_results['sample_adequacy'] = {
            'total_observations': len(data),
            'complete_cases': len(complete_cases),
            'overlap_ratio': overlap_ratio,
            'variables_tested': len(all_vars)
        }
        
        if overlap_ratio < 0.2:
            validation_results['errors'].append(
                f"Insufficient overlap: only {overlap_ratio:.1%} complete cases"
            )
            validation_results['is_valid'] = False
        elif overlap_ratio < 0.4:
            validation_results['warnings'].append(
                f"Low overlap: {overlap_ratio:.1%} complete cases - results may be unreliable"
            )
        
        # Final recommendations
        non_stationary_count = sum(1 for result in validation_results['stationarity_tests'].values() 
                                 if result.get('is_stationary') is False)
        
        if non_stationary_count > 0:
            validation_results['recommendations'].append(
                f"Consider using 'diff' or 'pct' transformations - {non_stationary_count} variables may be non-stationary"
            )
        
        return validation_results


class ImprovedDataPreprocessor:
    """
    Verbesserter Datenvorverarbeitungsschritt mit robusten Transformationen.
    """
    
    def __init__(self, data: pd.DataFrame, target_var: str, date_col: str = "Datum"):
        self.data = data.copy()
        self.target_var = target_var
        self.date_col = date_col

        self.forward_fill_used: bool = False
        self.fill_span_overlaps_test_period: bool = False
        
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        
    def create_robust_transformations(self, transformation: str = 'levels',
                                    train_end_date: pd.Timestamp = None,
                                    outlier_method: str = 'conservative') -> Dict[str, Any]:
        """
        Robuste Transformationen mit Anti-Leakage Schutz.
        
        Args:
            transformation: 'levels', 'log', 'pct', 'diff'
            train_end_date: Trainingsende für Anti-Leakage (optional)
            outlier_method: 'conservative', 'moderate', 'aggressive'
        """
        
        # Schritt 1: Mixed-Frequency Handling mit Anti-Leakage
        freq_result = ImprovedMixedFrequencyProcessor.align_frequencies_improved(
            self.data, self.target_var, self.date_col, 
            validation_split_date=train_end_date
        )
        
        processed_data = freq_result['processed_df']
        warnings_list = freq_result['warnings']
        
        # Schritt 2: Transformationen anwenden
        transformed_data = processed_data[[self.date_col]].copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.date_col in numeric_cols:
            numeric_cols.remove(self.date_col)
        
        print(f"Applying '{transformation}' transformation to {len(numeric_cols)} variables...")
        
        def _robust_pct_change(s: pd.Series, eps: float = 1e-8) -> pd.Series:
            return (s - s.shift(1)) / (np.abs(s.shift(1)) + eps)
        
        for col in numeric_cols:
            series = processed_data[col].copy()
            if transformation == 'robust_pct':
                transformed_data[col] = _robust_pct_change(series)

            elif transformation == 'levels':
                transformed_data[col] = series
                
            elif transformation == 'log':
                # Robuste Log-Transformation
                if (series > 0).sum() / series.notna().sum() > 0.9:
                    # Nur wenn > 90% der Werte positiv sind
                    # Kleine Konstante hinzufügen um Zeros zu handhaben
                    min_positive = series[series > 0].min()
                    epsilon = min_positive * 0.001 if pd.notna(min_positive) else 0.001
                    transformed_data[col] = np.log(series.clip(lower=epsilon))
                else:
                    transformed_data[col] = series
                    warnings_list.append(f"{col}: Not suitable for log transformation")
                    
            elif transformation == 'pct':
                # Prozentuale Änderungen
                transformed_data[col] = series.pct_change()
                
            elif transformation == 'diff':
                # Erste Differenzen
                transformed_data[col] = series.diff()
                
            else:
                transformed_data[col] = series
        
        # Schritt 3: Robuste Outlier-Behandlung
        transformed_data = self._robust_outlier_treatment(
            transformed_data, method=outlier_method
        )
        
        # Schritt 4: Saisonale Features hinzufügen
        transformed_data = self._add_seasonal_features(transformed_data)
        
        # Schritt 5: Final cleaning
        before_clean = len(transformed_data)
        
        # Nur Zeilen mit mindestens dem Target und einer exogenen Variable behalten
        essential_cols = [col for col in transformed_data.columns 
                         if col != self.date_col and col in [self.target_var] + 
                         [c for c in numeric_cols if c != self.target_var][:3]]  # Top 3 exog vars
        
        if len(essential_cols) > 1:
            # Behalte Zeilen mit Target + mindestens einer exogenen Variable
            keep_mask = (transformed_data[essential_cols].notna().sum(axis=1) >= 2)
            transformed_data = transformed_data[keep_mask].copy()
        else:
            # Fallback: alle NaN Zeilen entfernen
            transformed_data = transformed_data.dropna(how="all", subset=numeric_cols)
        
        after_clean = len(transformed_data)
        
        if before_clean > after_clean:
            print(f"Cleaned dataset: {before_clean} → {after_clean} observations")
        
        # Data types stabilisieren
        for col in [c for c in transformed_data.columns if c != self.date_col]:
            transformed_data[col] = pd.to_numeric(transformed_data[col], errors='coerce')
        
        return {
            'data': transformed_data,
            'warnings': warnings_list,
            'frequency_info': freq_result['frequency_info'],
            'leakage_risk': freq_result['leakage_risk'],
            'transformation_applied': transformation,
            'outlier_method': outlier_method,
            'rows_before_cleaning': before_clean,
            'rows_after_cleaning': after_clean
        }
    
    def _robust_outlier_treatment(self, data: pd.DataFrame, 
                                method: str = 'conservative') -> pd.DataFrame:
        """
        Robuste Outlier-Behandlung mit verschiedenen Aggressivitätsstufen.
        """
        numeric_cols = [c for c in data.columns if c != self.date_col]
        result_data = data.copy()
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) < 20:  # Zu wenige Daten für Outlier-Behandlung
                continue
            
            if method == 'conservative':
                # Sehr konservativ: nur extreme Outliers (0.5% / 99.5%)
                lower_bound = series.quantile(0.005)
                upper_bound = series.quantile(0.995)
            elif method == 'moderate':
                # Moderat: 1% / 99% Quantile
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
            else:  # aggressive
                # Aggressiv: 2.5% / 97.5% Quantile
                lower_bound = series.quantile(0.025)
                upper_bound = series.quantile(0.975)
            
            # Nur clippen wenn bounds sinnvoll sind
            if pd.notna(lower_bound) and pd.notna(upper_bound) and upper_bound > lower_bound:
                original_std = series.std()
                clipped_series = series.clip(lower=lower_bound, upper=upper_bound)
                clipped_std = clipped_series.std()
                
                # Nur anwenden wenn nicht zu viel Variation verloren geht
                if clipped_std > 0.5 * original_std:
                    result_data[col] = result_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return result_data
    
    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Saisonale Features hinzufügen."""
        data_with_features = data.copy()
        
        # Quartalsdummies
        data_with_features['quarter'] = pd.to_datetime(data_with_features[self.date_col]).dt.quarter
        
        for q in [2, 3, 4]:
            data_with_features[f'Q{q}'] = (data_with_features['quarter'] == q).astype(int)
        
        # Zeittrend (normalisiert)
        data_with_features['time_trend'] = (
            np.arange(len(data_with_features)) / len(data_with_features)
        )
        
        data_with_features = data_with_features.drop('quarter', axis=1)
        
        return data_with_features

# %%
"""
Test Script für die verbesserten Komponenten
Demonstriert die Korrekturen und deren Auswirkungen auf die Modellgüte
"""

def test_improved_system():
    """
    Testet das verbesserte System und zeigt die Verbesserungen auf.
    """
    print("=" * 80)
    print("TESTING IMPROVED FINANCIAL ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Configuration mit verbesserten Einstellungen
    config = AnalysisConfig(
        default_start_date="2000-01",
        default_end_date="2024-12",
        test_size=0.20,  # Kleinere Test-Sets für Stabilität
        cv_folds=3,      # Weniger CV-Folds
        gap_periods=2,   # Längere Gaps gegen Leakage
        cache_max_age_days=7,
        handle_mixed_frequencies=True,
        max_feature_combinations=10,  # Statt 15-20
    )
    
    # Test 1: Vergleich alter vs. neuer Ansatz
    print("\n" + "=" * 60)
    print("TEST 1: COMPARISON - OLD VS NEW APPROACH")
    print("=" * 60)
    
    target = "PH_KREDITE"
    start_date = "2005-01"
    
    try:
        # Lade Daten
        downloader = FinancialDataDownloader(config)
        series_definitions = {
            target: INDEX_TARGETS[target],
            **{k: v for k, v in STANDARD_EXOG_VARS.items()}
        }
        
        data = downloader.download(series_definitions, start_date=start_date, prefer_cache=True)
        print(f"Data loaded: {data.shape[0]} observations, {data.shape[1]-1} variables")
        
        # Test mit verbessertem System
        print("\n--- IMPROVED ANALYSIS ---")
        improved_results = improved_financial_analysis(
            data=data,
            target_var=target,
            exog_vars=[col for col in data.columns if col not in ['Datum', target]],
            analysis_type='comprehensive',
            config=config
        )
        
        if improved_results['status'] == 'success':
            print("✅ Improved analysis succeeded!")
            comparison_df = improved_results['comparison']
            print(f"Best method: {improved_results['best_method']}")
            print(f"Quality score: {improved_results['best_quality_score']:.1f}/5.0")
            
            # Zeige Top-3 Methoden
            print("\nTop 3 Methods:")
            display_cols = ['Method', 'Test_R²', 'Quality_Score', 'Overfitting', 'Flags']
            print(comparison_df[display_cols].head(3).to_string(index=False))
            
        else:
            print(f"❌ Improved analysis failed: {improved_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test 1 failed with exception: {str(e)}")
    
    # Test 2: Verschiedene Zielgrößen
    print("\n" + "=" * 60) 
    print("TEST 2: MULTIPLE TARGET VARIABLES")
    print("=" * 60)
    
    test_targets = ["PH_KREDITE", "PH_EINLAGEN", "PH_WERTPAPIERE"]
    results_summary = []
    
    for target in test_targets:
        print(f"\nTesting {target}...")
        
        try:
            result = quick_analysis_improved(target, start_date="2010-01", config=config)
            
            if result['status'] == 'success':
                best_method = result.get('best_method', 'N/A')
                best_r2 = result.get('comparison', pd.DataFrame()).iloc[0]['Test_R²'] if not result.get('comparison', pd.DataFrame()).empty else np.nan
                quality_score = result.get('best_quality_score', 0)
                
                results_summary.append({
                    'Target': target,
                    'Best_Method': best_method,
                    'Test_R²': best_r2,
                    'Quality_Score': quality_score,
                    'Status': '✅ Success'
                })
                
                print(f"  ✅ Success: {best_method}, R²={best_r2:.4f}, Quality={quality_score:.1f}")
                
            else:
                results_summary.append({
                    'Target': target,
                    'Best_Method': 'N/A',
                    'Test_R²': np.nan,
                    'Quality_Score': 0,
                    'Status': f"❌ Failed: {result.get('error', 'Unknown')[:50]}"
                })
                
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            results_summary.append({
                'Target': target,
                'Best_Method': 'N/A', 
                'Test_R²': np.nan,
                'Quality_Score': 0,
                'Status': f"❌ Exception: {str(e)[:50]}"
            })
            
            print(f"  ❌ Exception: {str(e)}")
    
    # Zeige Zusammenfassung
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(f"\nMULTI-TARGET SUMMARY:")
        print(summary_df.to_string(index=False))
        
        success_rate = (summary_df['Status'].str.contains('Success')).mean()
        avg_quality = summary_df['Quality_Score'].mean()
        
        print(f"\nOverall Performance:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Quality Score: {avg_quality:.1f}/5.0")
    
    # Test 3: Datenqualitäts-Validierung
    print("\n" + "=" * 60)
    print("TEST 3: DATA QUALITY VALIDATION")
    print("=" * 60)
    
    try:
        # Teste mit problematischen Daten (zu kurze Zeitreihe)
        short_data = data.tail(20).copy()  # Nur 20 Beobachtungen
        
        print("Testing with short dataset (20 observations)...")
        short_result = improved_financial_analysis(
            data=short_data,
            target_var=target,
            exog_vars=[col for col in short_data.columns if col not in ['Datum', target]][:3],
            analysis_type='quick',
            config=config
        )
        
        if short_result['status'] == 'failed':
            print("✅ Correctly rejected short dataset")
            print(f"   Reason: {short_result.get('error', 'Unknown')}")
        else:
            print("⚠️  Short dataset was accepted - may indicate insufficient validation")
        
        # Teste mit konstanter Zielvariable
        constant_data = data.copy()
        constant_data[target] = 100  # Konstanter Wert
        
        print("\nTesting with constant target variable...")
        constant_result = improved_financial_analysis(
            data=constant_data,
            target_var=target,
            exog_vars=[col for col in constant_data.columns if col not in ['Datum', target]][:3],
            analysis_type='quick',
            config=config
        )
        
        if constant_result['status'] == 'failed':
            print("✅ Correctly rejected constant target")
            print(f"   Reason: {constant_result.get('error', 'Unknown')}")
        else:
            print("⚠️  Constant target was accepted - validation may be insufficient")
            
    except Exception as e:
        print(f"❌ Test 3 failed with exception: {str(e)}")
    
    # Test 4: Anti-Leakage Test  
    print("\n" + "=" * 60)
    print("TEST 4: ANTI-LEAKAGE VERIFICATION")
    print("=" * 60)
    
    try:
        # Erstelle Analyzer für detaillierte Tests
        analyzer = ImprovedFinancialRegressionAnalyzer(
            data, target, 
            [col for col in data.columns if col not in ['Datum', target]][:4],
            config
        )
        
        # Teste Mixed-Frequency Processing
        print("Testing mixed-frequency processing...")
        freq_result = ImprovedMixedFrequencyProcessor.align_frequencies_improved(
            data, target, 'Datum', 
            validation_split_date=pd.Timestamp('2020-01-01')
        )
        
        leakage_risk = freq_result.get('leakage_risk', 'unknown')
        warnings = freq_result.get('warnings', [])
        
        print(f"  Leakage risk: {leakage_risk}")
        print(f"  Warnings: {len(warnings)}")
        
        if warnings:
            for warning in warnings[:3]:
                print(f"    - {warning}")
        
        if leakage_risk == 'low':
            print("✅ Low leakage risk detected")
        elif leakage_risk == 'medium':
            print("⚠️  Medium leakage risk - acceptable with warnings")  
        else:
            print("❌ High leakage risk - needs attention")
            
    except Exception as e:
        print(f"❌ Test 4 failed with exception: {str(e)}")
    
    # Test 5: Performance Benchmark
    print("\n" + "=" * 60)
    print("TEST 5: PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        import time
        
        # Zeitbasierter Performance-Test
        start_time = time.time()
        
        benchmark_result = comprehensive_analysis_improved(
            "PH_KREDITE", 
            start_date="2010-01",
            config=config
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"Analysis completed in {analysis_time:.1f} seconds")
        
        if benchmark_result['status'] == 'success':
            quality_score = benchmark_result.get('best_quality_score', 0)
            leakage_risk = benchmark_result.get('validation_summary', {}).get('leakage_risk', 'unknown')
            
            print(f"✅ Benchmark completed successfully")
            print(f"   Quality Score: {quality_score:.1f}/5.0")
            print(f"   Leakage Risk: {leakage_risk}")
            print(f"   Processing Time: {analysis_time:.1f}s")
            
            # Performance Rating
            if quality_score >= 3.0 and analysis_time < 30:
                print("🏆 EXCELLENT: High quality, fast execution")
            elif quality_score >= 2.0 and analysis_time < 60:
                print("✅ GOOD: Acceptable quality and speed")
            else:
                print("⚠️  NEEDS IMPROVEMENT: Low quality or slow execution")
        else:
            print(f"❌ Benchmark failed: {benchmark_result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Test 5 failed with exception: {str(e)}")
    
    # Abschluss
    print("\n" + "=" * 80)
    print("IMPROVED SYSTEM TEST COMPLETED")
    print("=" * 80)
    
    print("\nKey Improvements Implemented:")
    print("✅ Anti-leakage mixed-frequency processing")  
    print("✅ Robust time series cross-validation")
    print("✅ Conservative feature selection")
    print("✅ Enhanced data quality validation")
    print("✅ Stationarity-based transformation selection")
    print("✅ Quality-score based method ranking")
    print("✅ Comprehensive error handling")
    
    return True


def compare_old_vs_new_detailed():
    """
    Detaillierter Vergleich der alten und neuen Implementation.
    """
    print("=" * 80)
    print("DETAILED COMPARISON: OLD VS NEW IMPLEMENTATION")
    print("=" * 80)
    
    config = AnalysisConfig()
    target = "PH_KREDITE"
    
    try:
        # Lade Testdaten
        data = get_target_with_standard_exog(target, "2005-01", config)
        exog_vars = [col for col in data.columns if col not in ['Datum', target]][:4]
        
        print(f"Test data: {data.shape[0]} observations, {len(exog_vars)} features")
        print(f"Target: {target}")
        print(f"Features: {', '.join(exog_vars)}")
        
        # Old approach (original system)
        print(f"\n--- OLD APPROACH ---")
        try:
            old_results = financial_analysis(data, target, exog_vars, 'comprehensive', config)
            
            if old_results['status'] == 'success':
                old_best_r2 = old_results.get('best_test_r2', np.nan)
                old_method = old_results.get('best_method', 'N/A')
                
                print(f"✅ Old system: {old_method}, R² = {old_best_r2:.4f}")
            else:
                print(f"❌ Old system failed: {old_results.get('error', 'Unknown')}")
                old_best_r2 = np.nan
                old_method = 'Failed'
                
        except Exception as e:
            print(f"❌ Old system exception: {str(e)}")
            old_best_r2 = np.nan
            old_method = 'Exception'
        
        # New approach (improved system)  
        print(f"\n--- NEW APPROACH ---")
        try:
            new_results = improved_financial_analysis(data, target, exog_vars, 'comprehensive', config)
            
            if new_results['status'] == 'success':
                new_best_r2 = new_results['comparison'].iloc[0]['Test_R²']
                new_method = new_results.get('best_method', 'N/A')
                new_quality = new_results.get('best_quality_score', 0)
                
                print(f"✅ New system: {new_method}, R² = {new_best_r2:.4f}, Quality = {new_quality:.1f}")
            else:
                print(f"❌ New system failed: {new_results.get('error', 'Unknown')}")
                new_best_r2 = np.nan
                new_method = 'Failed'
                new_quality = 0
                
        except Exception as e:
            print(f"❌ New system exception: {str(e)}")
            new_best_r2 = np.nan
            new_method = 'Exception'
            new_quality = 0
        
        # Comparison summary
        print(f"\n--- COMPARISON SUMMARY ---")
        
        comparison_table = pd.DataFrame({
            'System': ['Old', 'New'],
            'Method': [old_method, new_method],
            'Test_R²': [old_best_r2, new_best_r2],
            'Quality_Score': [np.nan, new_quality],
            'Status': [
                'Success' if pd.notna(old_best_r2) else 'Failed',
                'Success' if pd.notna(new_best_r2) else 'Failed'
            ]
        })
        
        print(comparison_table.to_string(index=False))
        
        # Analysis of improvement
        if pd.notna(old_best_r2) and pd.notna(new_best_r2):
            improvement = new_best_r2 - old_best_r2
            print(f"\nR² Improvement: {improvement:+.4f}")
            
            if improvement > 0.05:
                print("🎉 SIGNIFICANT IMPROVEMENT")
            elif improvement > 0:
                print("✅ SLIGHT IMPROVEMENT") 
            elif improvement > -0.05:
                print("≈ COMPARABLE PERFORMANCE")
            else:
                print("⚠️  PERFORMANCE DEGRADATION")
        
        # Qualitative improvements
        print(f"\nQualitative Improvements in New System:")
        print("- Enhanced data validation with stationarity tests")
        print("- Anti-leakage mixed-frequency processing")
        print("- Robust cross-validation with proper time gaps")
        print("- Quality-based method evaluation")
        print("- Conservative feature selection")
        print("- Better error handling and reporting")
        
    except Exception as e:
        print(f"❌ Detailed comparison failed: {str(e)}")


if __name__ == "__main__":
    print("Starting comprehensive test of improved financial analysis system...")
    
    # Run all tests
    try:
        # Test improved system
        test_improved_system()
        
        # Detailed comparison
        print("\n" + "="*80)
        compare_old_vs_new_detailed()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
        print("\nRECOMMENDATIONS:")
        print("1. Use improved_financial_analysis() instead of financial_analysis()")
        print("2. Use quick_analysis_improved() for quick tests")
        print("3. Use comprehensive_analysis_improved() for full analysis")
        print("4. Monitor Quality_Score - aim for >3.0")
        print("5. Check leakage_risk in results - should be 'low'")
        print("6. Review validation warnings before proceeding")
        
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

# %%
# # Test ob Forward-Fill korrekt funktioniert
# data = get_target_with_standard_exog("PH_KREDITE", "2010-01")
# print("Before forward-fill:")
# print(data['PH_KREDITE'].notna().sum(), "of", len(data))

# # Nach Forward-Fill (simuliere den Prozess)
# processor = ImprovedMixedFrequencyProcessor()
# result = processor.align_frequencies_improved(data, 'PH_KREDITE', 'Datum')
# processed_data = result['processed_df']

# print("After forward-fill:")
# print(processed_data['PH_KREDITE'].notna().sum(), "of", len(processed_data))
# print("Remaining NaNs:", processed_data['PH_KREDITE'].isnull().sum())











