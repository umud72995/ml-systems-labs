import numpy as np
import pandas as pd
from typing import Dict, Tuple

class VectorizedTransformer:
    """Vectorized data engine utilizing Pandas pipe transformations."""
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians_: Dict[str, float] = {}
        self.modes_: Dict[str, str] = {}
        self.percentiles_: Dict[str, Tuple[float, float]] = {}

    def fit_from_train(self, df: pd.DataFrame):
        for c in self.num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            self.medians_[c] = float(s.median())
            self.percentiles_[c] = (float(s.quantile(0.01)), float(s.quantile(0.99)))

        for c in self.cat_cols:
            s = df[c].astype(str).str.strip().replace("?", np.nan)
            m = s.dropna().mode()
            self.modes_[c] = m.iloc[0] if not m.empty else "Missing"
        return self

    def pipe_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(self.medians_[c])
            lo, hi = self.percentiles_[c]
            out[c] = out[c].clip(lower=lo, upper=hi)

        for c in self.cat_cols:
            out[c] = out[c].astype(str).str.strip().replace("?", np.nan).fillna(self.modes_[c])
        return out

    def pipe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["capital_net"] = out["capital-gain"] - out["capital-loss"]
        age_sub = np.maximum(out["age"] - 16.0, 1.0)
        out["hours_per_age"] = out["hours-per-week"] / age_sub
        return out
