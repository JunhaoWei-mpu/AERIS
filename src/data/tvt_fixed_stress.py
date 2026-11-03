"""One registered, label-independent corruption on a continuous flight stream."""
import hashlib
import numpy as np

SEED = 20260923
STRESSES = ('noise', 'channel_dropout', 'control_masking')


def flight_key(flight_id):
    return int.from_bytes(hashlib.sha256(f'{SEED}/{flight_id}'.encode()).digest()[:8], 'little')


def corrupt(x, time, flight_id, mean, std, constant, response_candidates, intent, stress):
    if stress not in STRESSES:
        raise ValueError(stress)
    x = np.asarray(x)
    y = x.copy()
    finite = np.isfinite(x)
    key = flight_key(flight_id)
    rng = np.random.default_rng(key)
    # Draw on the full channel grid before excluding constants. Thus a split's
    # constant-channel policy cannot change the random draws of other channels.
    noise = rng.standard_normal(x.shape)
    phase = float((key >> 16) % 10000) / 1000.
    elapsed = np.asarray(time) - float(time[0])
    interval = np.remainder(elapsed + phase, 10.) < 1.
    mask = np.zeros(x.shape, bool)
    channel = None
    if stress == 'noise':
        mask[:] = finite
        mask[:, list(constant)] = False
        delta = noise * (.05 * np.asarray(std))[None, :]
        y[mask] = (x.astype(np.float64) + delta)[mask]
    elif stress == 'control_masking':
        mask[:, list(intent)] = interval[:, None]
        mask &= finite
        replacement = np.broadcast_to(np.asarray(mean), x.shape)
        y[mask] = replacement[mask]
    else:
        channel = int(response_candidates[key % len(response_candidates)])
        mask[:, channel] = interval & finite[:, channel]
        last = float(mean[channel])
        for i in range(len(x)):
            if mask[i, channel]:
                y[i, channel] = last
            elif finite[i, channel]:
                last = float(x[i, channel])
    # Existing missing data stay missing; corruption must not create new windows.
    np.testing.assert_array_equal(np.isfinite(y), finite)
    info = dict(seed=SEED, flight_key=str(key), phase_seconds=phase,
                channel=channel, stress=stress, changed_entries=int(mask.sum()))
    return y, mask, info
