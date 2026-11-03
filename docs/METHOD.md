# Frozen monitoring mechanism

Let `C`, `A`, and `Y` denote recorded control intent, logged actuator output, and vehicle-response channels. The monitor reads a completed trailing window of 100 samples. It compares two recorded relations, `C→A` and `A→ΔY`, with nominal behavior, while a separate neural Base reconstructs `[A,Y]`.

## Nominal context matching

The context comprises the driver's coordinatewise temporal mean, population standard deviation and penultimate value, plus the initial response state. The driver summary includes positions 0 through 98 of a 100-sample window. Angular initial states are encoded as sine and cosine in physical units.

For CA, the response is logged `A` throughout the window. For AY, the response is the average of `h` consecutive `h`-step changes, with shortest signed circular differences for angle channels. `h=20` at 50 Hz and `h=8` at 20 Hz. This averages over the already fixed 0.4 s horizon; it does not use samples beyond the scoring endpoint.

Nominal training windows are nonoverlapping and deduplicated in absolute source-recording time. Context standardization is fitted on this library. Exact squared-Euclidean retrieval selects `k=64` neighbors after excluding every library window from the query's source recording. Insufficient eligible neighbors is an unavailable reference/error, never a healthy score or permission to reduce `k`.

## Location and two-sided dispersion

For a response channel, let `m,v` be the query's temporal mean and population variance. The matches provide per-window means `M_i` and variances `V_i`:

```text
mu  = median_i M_i
nu  = median_i V_i
mad = 1.4826 median_i |M_i − mu|
a²  = max(nu + mad², 1e−8)
L_j = [v + (m − mu)²] / a²
R_j = log[(v + 1e−8)/(nu + 1e−8)]
D_j = |R_j − rho_j|
```

`rho_j` is the nominal centering median on its assigned validation role. CA uses the channel mean, and AY uses the channel median, for both the location and dispersion summaries. Location energy already contains within-window variance; removing explicit dispersion does not remove every second-order statistic.

Exactly constant training channels are zeroed at inference and excluded from the relevant evidence statistics. Test ranges do not redefine these channels or their normalization.

## Calibration roles and integration

Each nominal validation source is divided chronologically into quartiles:

1. Nominal checkpoint selection.
2. Dispersion centering.
3. Base/location/dispersion primitive-tail calibration.
4. Maximum-evidence calibration, further separated into edge and final roles.

A 2 s purge on each side of quartile boundaries prevents shared window observations. The fourth role is split at its source-time median with a 1 s gap and complete-history separation. Common-support membership retains the original 199-sample restriction. These are separate observation roles on shared nominal flights, not independent flight cohorts.

Each tail uses the empirical survival body, including repeated values and the original `1e−12` smoothing. Above the nominal 95% quantile, tail evidence continues linearly with scale equal to the mean of strictly positive excesses. At least 20 positive excesses and a finite positive scale are mandatory; there is no percentile/family search or fallback.

```text
U_CA = max(T_location_CA(L_CA), T_dispersion_CA(D_CA))
U_AY = max(T_location_AY(L_AY), T_dispersion_AY(D_AY))
E_CA = T_edge_CA(U_CA)
E_AY = T_edge_AY(U_AY)
B    = T_base(Base_MSE)
score = T_final(max(B, E_CA, E_AY))
```

The successive maps are fitted on the assigned nominal roles. Their separate scales mean that plotted primitive and final scores cannot be interpreted as a direct maximum without their intermediate maps.

## Implementation map

- `aeris/protocol.py`: source-separated membership, nominal roles, support and library deduplication.
- `aeris/training.py`: audited reconstruction constructor, clean auxiliary targets and nominal-only checkpoint selection.
- `src/models/tvt_horizon_response.py`: fixed averaged response operator.
- `src/models/tvt_matched_chain.py`: context coordinates, source-excluded neighbors and reference moments.
- `src/models/tvt_response_median.py`: AY median aggregation.
- `src/models/tvt_exponential_tail.py`: fixed empirical-body/exponential-tail calibration.
- `src/models/tvt_horizon_monitor.py`: calibrated maximum integration and pooled control.
- `src/models/aeris_horizon_frozen.py`: label-free scoring of complete raw-unit windows.
- `src/models/tvt_online.py`, `tvt_stable_window_inference.py`: per-window Fourier selection and numerical near-tie handling.

The numerical source files retain historical class/module names so locally fitted or separately retained original trusted checkpoints can be loaded. Helpers for inactive historical mechanisms in those files are not part of the final AERIS scoring path.
