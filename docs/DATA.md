# Data, semantics and fixed partitions

Download ALFA and the real-flight portion of RflyMAD from the links in the main README. No telemetry, labels, checkpoints or experimental predictions are distributed in this repository.

## CSV input layout

ALFA uses the provider's processed-flight CSV exports, not a concatenation of the raw flights:

```text
data/raw/alfa/
  carbonZ_<date-time>_<sequence-description>/
    ...mavros-rc-in.csv
    ...mavros-rc-out.csv
    ...mavros-nav_info-*.csv
    ...mavros-imu-data.csv
    ...mavros-imu-atm_pressure.csv
    ...mavros-global_position-rel_alt.csv
    ...failure-status*.csv
```

Keep the original filenames. `scripts/rebuild_tvt_data.py` uses the topic names and field names in the published CSV export. The ALFA failure-status topic is parsed by `src/data/tvt_raw.py`; retained native failure modes are 1, 2 and 3. A release without native ground truth is excluded.

RflyMAD uses the provider's real-flight topic exports:

```text
data/raw/rflymad/
  Real-No_Fault/<regime>/<case>/log_*/log_*_<topic>.csv
  Real-Motor/<regime>/<case>/log_*/log_*_<topic>.csv
  Real-Sensors/<regime>/<case>/log_*/log_*_<topic>.csv
```

Required PX4 topics are `actuator_controls_0_0`, `actuator_outputs_0`, `sensor_combined_0`, `sensor_baro_0`, `vehicle_attitude_0`, `vehicle_local_position_0`, and the recorded injection-status topic. If your download contains ULogs rather than topic CSVs, use the provider's conversion tools first. The adapter deliberately does not replace missing fields or infer an injection from an anomaly score.

## Causal preprocessing

- ALFA: 50 Hz, using ROS receipt times. RflyMAD: 20 Hz, using the PX4 clock.
- Each channel is held from the most recently received observation. Maximum age is 1 s. Older samples remain missing; there is no future interpolation.
- Nominal training eligibility ends at the first native injection. Later fault-off periods do not reenter training.
- ALFA additionally marks `[first labeled onset − 0.2 s, first labeled onset)` unknown. `native_label` is retained alongside the guarded evaluation label.
- A window never crosses a recording boundary, clock discontinuity or missing-data span.

The expected retained corpus is 46 ALFA released sequences and 488 real RflyMAD cases for the dataset versions used in the manuscript. These are not counts of independent aircraft. If a provider changes the archive, inspect the preparation exclusions and split coverage rather than renaming or relabeling records to force the expected count.

## Processed format

Each `<flight_id>.npz` contains:

| Key | Shape / meaning |
|---|---|
| `x` | `(T,D)` float32, raw physical units; normalization has not been fitted |
| `t` | `(T,)` float64, seconds from the processed sequence start |
| `label` | `(T,)` int8: nominal 0, fault 1, unknown −1 |
| `nominal_eligible` | `(T,)` boolean; stricter than `label == 0` |
| `channel_age_s` | `(T,D)` most-recent observation ages |
| `sample_rate` | scalar 50 or 20 |
| `channel_names` | fixed ordered channel names |
| `fault_type` | recorded family identifier |

`meta.json` stores absolute sequence starts, source descriptions and per-file checksums. ALFA's guarded version also retains `native_label` and `label_uncertain`.

## Channel order

| Dataset | Indices | Role | Channels |
|---|---|---|---|
| ALFA | 0–3 | C | Pilot roll, pitch, throttle, yaw |
| ALFA | 4–8 | A | Elevator, throttle, rudder, right flaperon, left flaperon PWM |
| ALFA | 9–12 | C | Navigation roll, pitch, yaw, airspeed references |
| ALFA | 13–21 | Y | Airspeed, pressure, gyro x/y/z, roll/pitch/yaw, altitude |
| RflyMAD | 0–3 | C | Recorded roll, pitch, yaw, thrust requests |
| RflyMAD | 4–7 | A | Four logged actuator/motor PWM outputs |
| RflyMAD | 8–15 | Y | Gyro x/y/z, pressure, roll/pitch/yaw, altitude |

ALFA angles are degrees; RflyMAD angles are radians. RflyMAD attitude is derived from the recorded quaternion; altitude uses the negative local vertical coordinate. Logged PWM is not measured surface position, motor speed or thrust.

## Source grouping and settings

The ten files in `splits/` contain the fixed train/validation/test sequence identities and `flight_to_group` mappings. ALFA sibling releases from the same source recording stay in one partition. All fitting checks source-group disjointness before loading training data.

ALFA settings: STRAT, LOFO-ENGINE, LOFO-AILERON, LOFO-ELEV, LOFO-RUDDER. RflyMAD settings: STRAT, LOFO-MOTOR, LOFO-GYRO, LOFO-BARO, LOFO-ACCEL. Because all training is nominal-only, LOFO also changes nominal maneuver/cohort coverage; it is not a pure intervention on whether faults were included in training.

The model itself takes 100 samples. The benchmark's inherited common support requires 199 contiguous samples and is recorded separately from native score availability. It is not a measured warning delay.
