"""One prespecified nominal tail extension; no tunable cutoff or tail family."""
import numpy as np
from src.models.tvt_control_chain import EmpiricalTail


class ExponentialTail(EmpiricalTail):
    def fit(self, values):
        super().fit(values)
        if self.values.ndim != 1:
            raise ValueError('one-dimensional nominal reference required')
        self.threshold = float(np.quantile(self.values, .95))
        excess = self.values[self.values > self.threshold] - self.threshold
        self.exceedances = len(excess)
        if self.exceedances < 20:
            raise ValueError('fewer than 20 strictly positive nominal exceedances')
        self.scale = float(excess.mean())
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('nonfinite or nonpositive nominal excess scale')
        # Preserve the legacy +1e-12 smoothing at the joining point exactly.
        # The epsilon is NOT added again after exponentiation: no new ceiling.
        self.anchor = float(super().transform(np.array([self.threshold]))[0])
        return self

    @property
    def cap(self):
        return float('inf')

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        body = super().transform(values)
        above = values > self.threshold
        out = np.array(body, copy=True)
        out[above] = self.anchor + (values[above] - self.threshold) / self.scale
        if not np.isfinite(out).all():
            raise ValueError('tail evidence overflow; no clipping fallback')
        return out

    def parameters(self):
        return dict(n=len(self.values), percentile=.95, threshold=self.threshold,
                    positive_exceedances=self.exceedances, scale=self.scale,
                    anchor_evidence=self.anchor,
                    anchor_smoothing='legacy empirical survival + 1e-12')
