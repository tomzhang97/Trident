"""
Calibration module for conformal prediction.
UPDATES:
- Fixed PickleError by replacing lambda with named factory function.
- Preserves defaultdict behavior after loading.
"""
import copy
import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

def _to_plain(obj):
    """Recursively convert defaultdicts/numpy types to plain python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(i) for i in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# FIX: Top-level factory function for pickling support
def _nested_dd_factory():
    return defaultdict(list)

class ReliabilityCalibrator:
    def __init__(self, use_mondrian: bool = True, version: str = "v1.0"):
        self.use_mondrian = use_mondrian
        self.version = version
        self.conformal_calibrator = None
        # Compatibility shim for legacy reliability tables and bin statistics
        self.tables: Dict[str, List[Tuple[float, float]]] = {}
        self.bins: Dict[str, Any] = {}

    def set_conformal_calibrator(self, calibrator):
        self.conformal_calibrator = calibrator

    def fit(self, *args, **kwargs):
        pass

    def _get_canonical_key(self, key: str) -> str:
        """Map specific facet types to canonical buckets."""
        if "BRIDGE_HOP" in key:
            return "BRIDGE_HOP"
        return key

    def to_pvalue(self, score: float, facet_type: Union[str, Any], text_length: int = 0) -> float:
        """
        Convert score to p-value with robust fallback.
        """
        # Safety: Coerce inputs
        if hasattr(facet_type, "value"):
            facet_type = facet_type.value
        facet_type = str(facet_type)
        score = float(score)

        if self.conformal_calibrator:
            try:
                return self.conformal_calibrator.get_p_value(score, facet_type, text_length)
            except KeyError:
                # 1. Canonical Fallback (e.g. BRIDGE_HOP3 -> BRIDGE_HOP)
                canon = self._get_canonical_key(facet_type)
                if canon != facet_type:
                    try:
                        return self.conformal_calibrator.get_p_value(score, canon, text_length)
                    except KeyError:
                        pass

                # 2. Global Fallback (Fail-Closed)
                return 1.0

        # Legacy reliability tables fallback (step function on score)
        canon = self._get_canonical_key(facet_type)
        table = self.tables.get(canon)
        if table:
            sorted_table = sorted(table, key=lambda t: t[0])
            for threshold, pval in sorted_table:
                if score <= threshold:
                    return pval
            return sorted_table[-1][1]

        return 1.0

    def save(self, path: str):
        data = {
            "version": self.version,
            "use_mondrian": self.use_mondrian,
            "conformal": self.conformal_calibrator.to_dict() if self.conformal_calibrator else None,
            "tables": self.tables,
            "bins": self.bins,
        }
        with open(path, "w") as f:
            json.dump(_to_plain(data), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        cal = cls(
            use_mondrian=data.get("use_mondrian", True),
            version=data.get("version", "v1.0")
        )
        if data.get("conformal"):
            cal.conformal_calibrator = SelectionConditionalCalibrator.from_dict(data["conformal"])
        cal.tables = data.get("tables", {}) or {}
        cal.bins = data.get("bins", {}) or {}
        return cal

# --- INNER CALIBRATOR ---

class SelectionConditionalCalibrator:
    def __init__(self, n_min: int = 50, use_mondrian: bool = True):
        self.n_min = n_min
        self.use_mondrian = use_mondrian
        # FIX: Use named factory function instead of lambda for pickle safety
        self.bins: Dict[str, Dict[str, List[float]]] = defaultdict(_nested_dd_factory)
        self.thresholds: Dict[str, float] = {}
        self.temp_data = []

    def add_calibration_sample(self, score: float, is_sufficient: bool, facet_type: str, text_length: int):
        self.temp_data.append({
            "s": float(score), "y": bool(is_sufficient), "ft": str(facet_type), "len": int(text_length)
        })

    def finalize(self):
        by_ft = defaultdict(list)
        for d in self.temp_data:
            by_ft[d["ft"]].append(d)

        for ft, items in by_ft.items():
            if self.use_mondrian:
                lengths = [x["len"] for x in items]
                if not lengths: continue
                med_len = float(np.median(lengths))

                short_neg = [x["s"] for x in items if x["len"] <= med_len and not x["y"]]
                long_neg  = [x["s"] for x in items if x["len"] > med_len and not x["y"]]

                # Fallback to single bucket if sparse
                if len(short_neg) < self.n_min or len(long_neg) < self.n_min:
                    all_neg = [x["s"] for x in items if not x["y"]]
                    all_neg.sort()
                    self.bins[ft]["all"] = all_neg
                else:
                    short_neg.sort()
                    long_neg.sort()
                    self.bins[ft]["short"] = short_neg
                    self.bins[ft]["long"] = long_neg
                    self.thresholds[ft] = med_len
            else:
                neg = [x["s"] for x in items if not x["y"]]
                neg.sort()
                self.bins[ft]["all"] = neg

        self.temp_data = []

    def get_p_value(self, score: float, facet_type: str, text_length: int) -> float:
        ft_bins = self.bins.get(facet_type)
        if not ft_bins:
            raise KeyError(f"No calibration data for type: {facet_type}")

        if "all" in ft_bins:
            neg_scores = ft_bins["all"]
        else:
            thresh = self.thresholds.get(facet_type)
            if thresh is None:
                s = ft_bins.get("short", [])
                l = ft_bins.get("long", [])
                if not s and not l:
                    return 1.0
                neg_scores = sorted(s + l)
            else:
                key = "short" if text_length <= thresh else "long"
                neg_scores = ft_bins.get(key, [])

        if not neg_scores:
            return 1.0

        import bisect
        idx = bisect.bisect_left(neg_scores, score)
        count_ge = len(neg_scores) - idx

        return (1.0 + count_ge) / (len(neg_scores) + 1.0)

    def to_dict(self) -> Dict:
        return {
            "n_min": self.n_min,
            "use_mondrian": self.use_mondrian,
            "bins": _to_plain(self.bins),
            "thresholds": _to_plain(self.thresholds)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SelectionConditionalCalibrator":
        obj = cls(n_min=data["n_min"], use_mondrian=data["use_mondrian"])

        # FIX: Use named factory function here too
        obj.bins = defaultdict(_nested_dd_factory)

        plain_bins = data.get("bins", {})
        for ft, inner in plain_bins.items():
            for k, v in inner.items():
                obj.bins[ft][k] = v

        obj.thresholds = data.get("thresholds", {})
        return obj

# --- RESTORED HELPER FOR IMPORT COMPATIBILITY ---

class CalibrationMonitor:
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(int)

    def update(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def get_metrics(self):
        return {}
