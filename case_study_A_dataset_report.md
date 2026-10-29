# ZeMA Electromechanical Cylinder Dataset — Data Description and Consistency with Dorst et al. 2022

**Dataset:** T. Dorst, M. Gruber, A. P. Vedurmudi, *Sensor data set of one electromechanical cylinder at ZeMA testbed (ZeMA DAQ and Smart-Up Unit)*, Zenodo, 2021, doi:10.5281/zenodo.5185953 — file `axis11_2kHz_ZeMA_PTB_SI.h5` (2.67 GB), CC-BY-4.0.

**Reference paper ("Dorst2022"):** T. Dorst, M. Gruber, B. Seeger, A. P. Vedurmudi, T. Schneider, S. Eichstädt, A. Schütze, *Uncertainty-aware data pipeline of calibrated MEMS sensors used for machine learning*, Measurement: Sensors 22 (2022) 100376, doi:10.1016/j.measen.2022.100376. The publisher's page and the Saarland University open-access copy could not be retrieved automatically; the paper was read in full from its reproduction in T. Dorst, *Measurement uncertainty in machine learning – uncertainty propagation and influence on performance*, PhD thesis, Saarland University, 2023 (Paper 1, thesis pp. 54–66; test-bed description pp. 38–40).

**Reproducibility:** every number in this report is printed by `check_dataset_consistency.py` (output captured in `dataset_consistency.log`), using the project's `zema_env` environment.

---

## 1. Summary

The published file matches the paper on everything that can be checked against it: acquisition systems and sensors, number of signals, cycle selection, window length and sampling rate, operating velocity, total test duration, file organisation and units. Five findings need attention when the dataset is used, especially for uncertainty-aware work:

| # | Finding | Severity |
|---|---|---|
| F1 | The metadata state **4776** cycles; the arrays hold **4766**, which is exactly what the paper's selection rule produces. 4776 is a typo. | Low |
| F2 | The measured axial force (per-cycle mean 5.88–6.00 kN) is below the nominal **7 kN** load stated in the paper, and it is not constant within the window (it rises from 5.35 to 6.53 kN). | Medium |
| F3 | The paper does not describe how the **ZeMA DAQ uncertainties** were obtained. In the file they are either a single constant per channel or an exact 1% / 1.5% of the reading, consistent with datasheet-based Type B values. | High (for metrological use) |
| F4 | The **SUU uncertainty arrays are duplicated across quantities**: the MPU 9250 acceleration, angular-velocity, magnetic-flux-density and temperature uncertainties are identical sample by sample, despite having different units. The three BMA 280 axes are also identical. | High |
| F5 | The SUU uncertainty level (median 0.095 m/s² for the BMA 280, 0.049 m/s² for the MPU 9250 in cycle 51) is **20–40 times larger** than the interpolation-uncertainty term reported in the paper's Appendix A (2.236·10⁻³ m/s²). The two quantities are not necessarily the same: see §6.3. | Medium |

One further point concerns the companion library rather than the dataset: `zema_emc_annotated.ZeMASamples(datapoints_per_cycle=d)` returns the **first d samples** of each 2000-sample window, not d samples spread over it (§7).

---

## 2. Test bed and acquisition (from the paper)

- **Device under test:** Festo ESBF electromechanical cylinder ("axis11"), loaded axially by a pneumatic cylinder simulating **7 kN** (the manufacturer's maximum).
- **Working cycle:** forward stroke, waiting time (150 ms), return stroke; **2.8 s** in total. Both strokes run at approx. **200 mm/s** and **5 m/s²**.
- **Failure criterion:** end-position deviation; the lifetime test lasted approx. **16.5 days (387.83 h)** in April 2021.
- **ZeMA DAQ:** NI PXI system with three modules (PXIe-5170R oscilloscope, 14 bit; PXIe-4492 sound and vibration, 24 bit; PXIe-6341 multifunction I/O, 16 bit). It acquires 11 sensors: 3 motor currents at 1 MHz, a microphone at 100 kHz, 3 accelerometers at 100 kHz, and 4 process signals at 10 kHz (axial force, pneumatic pressure, velocity, active current). Acquisition is event-centric, triggered by the motor controller, and records no absolute timestamps. The raw data total 9.41 TB over **476,560 cycles**.
- **Downsampled ZeMA data set:** **1 s of the return stroke**, every **100th cycle starting from cycle 51**, downsampled to **2 kHz**.
- **SmartUp Unit (SUU):** STM32F767ZI-based DAQ with GNSS-disciplined absolute timestamps. It carries an InvenSense MPU-9250 (9-axis IMU), a Bosch BMA280 (3-axis accelerometer) and a TE MS5837-02BA (pressure/temperature, excluded because it failed after installation). The SUU is mounted on the same holder as the Kistler 8712A5M1 accelerometer at the plain bearing. It records continuously, 71 GB in total; **476,617 cycles** are detected from the recorded ZeMA trigger.
- **SUU preprocessing:** cycle detection and extraction, time-glitch correction, then interpolation to an equidistant time base with PyDynamic (`interp1d_unc`, five schemes). Data uncertainty is set to ½ LSB, and an uncertainty from dynamic calibration (an IIR inverse filter with GUM-S2 uncertainty) is also derived.

---

## 3. File structure and metadata

```
/                       attributes: Experiment, Person, Project, Publication (JSON)
├── ZeMA_DAQ/           7 quantity groups, 11 signals, 2000 samples × 4766 cycles
│   ├── Acceleration/            qudt:value, qudt:standardUncertainty  (3, 2000, 4766)
│   ├── Active_Current/          (2000, 4766)
│   ├── Force/                   (2000, 4766)
│   ├── Motor_Current/           (3, 2000, 4766)
│   ├── Pressure/                (2000, 4766)
│   ├── Sound_Pressure/          (2000, 4766)
│   └── Velocity/                (2000, 4766)
└── PTB_SUU/            13 signals, 1000 samples × 4766 cycles
    ├── BMA_280/Acceleration/              (3, 1000, 4766)
    └── MPU_9250/{Acceleration, Angular_Velocity, Magnetic_Flux_Density}  (3, 1000, 4766)
                 Temperature/              (1000, 4766)
```

- Every quantity group has a `qudt:value` dataset and a `qudt:standardUncertainty` dataset of the same shape. Arrays are indexed as `[axis,] sample, cycle`.
- Each group is annotated with:
  - `si:unit` — D-SI notation;
  - `sosa:madeBySensor` — sensor model;
  - `qudt:hasQuantityKind` — the physical quantity measured;
  - `rdf:type`;
  - `si:label` — one label per axis or phase.
- ZeMA value datasets carry `misc: {"raw_data": false, "comment": "Converted from ADC values based on appropriate conversion."}`. SUU groups carry `misc: {"interpolation_scheme": "cubic"}`.
- Root metadata:
  - **Experiment:** date 2021-03-29/2021-04-15, DUT "Festo ESBF cylinder", identifier "axis11".
  - **Person:** Dorst (ZeMA), Gruber and Vedurmudi (PTB).
  - **Project:** Met4FoF, EMPIR 17IND12.
  - **Publication:** DOI, CC-BY-4.0, description, `dc:SizeOrDuration` = "24 sensors, 4776 cycles and 2000 datapoints each".
- No time vector is stored. The time axis is implicit: sample *i* is at *i*/2000 s (ZeMA) or *i*/1000 s (SUU) from the start of the extracted window.
- There are no NaN or Inf values in any value or uncertainty dataset.

---

## 4. Channel inventory — ZeMA DAQ (2 kHz, 2000 samples per cycle)

Statistics are computed over all 4766 cycles × 2000 samples. *u/s* is the median declared standard uncertainty divided by the median cross-cycle standard deviation at a fixed within-window position: it compares the measurement uncertainty with the cycle-to-cycle variability that a model can learn from (§6.1).

| Signal | Sensor (`sosa:madeBySensor`) | Unit (`si:unit`) | Min | Max | Mean | Std | Standard uncertainty | *u/s* |
|---|---|---|---|---|---|---|---|---|
| Acceleration plain bearing | Kistler 8712A5M1 | m s⁻² | −8.91 | 10.46 | 0.034 | 1.61 | constant 2.832 | 1.89 |
| Acceleration piston rod | Kistler 8704B50M1 | m s⁻² | −31.26 | 32.05 | −0.317 | 5.38 | constant 2.832 | 0.60 |
| Acceleration ball bearing | Kistler 8704B50M1 | m s⁻² | −19.73 | 37.06 | 0.002 | 3.87 | constant 2.832 | 0.91 |
| Active current | Festo CMMP-C10-11A-AS (motor controller) | A | −5.06 | −1.26 | −3.24 | 0.34 | 1.0% of \|value\| | 0.19 |
| Axial force | Lorenz K-25 | **kN** | 3.08 | 7.33 | 5.96 | 0.36 | constant 0.01244 | 0.24 |
| Motor current phase 1 | Sensitec CMS 2005 | A | −6.55 | 6.82 | 0.041 | 3.30 | 1.5% of \|value\| | 0.023 |
| Motor current phase 2 | Sensitec CMS 2005 | A | −6.69 | 7.22 | −0.041 | 3.30 | 1.5% of \|value\| | 0.024 |
| Motor current phase 3 | Sensitec CMS 2005 | A | −6.45 | 6.99 | 0.024 | 3.28 | 1.5% of \|value\| | 0.024 |
| Pneumatic pressure | Festo VPPM | Pa | 4.96·10⁵ | 9.77·10⁵ | 8.64·10⁵ (8.64 bar) | 1.77·10⁴ | constant 1.683·10⁴ | **10.6** |
| Sound pressure | G.R.A.S. 46BE | Pa | −3.36 | 2.98 | −0.100 | 0.64 | constant 2.788·10⁻⁵ | 5·10⁻⁵ |
| Velocity | Festo CMMP-C10-11A-AS (motor controller) | **mm s⁻¹** | −219.4 | −119.1 | −200.4 | 2.04 | 1.0% of \|value\| | 4.6 |

**Unit notes.**
- **Axial force is in kN and velocity in mm/s**, not in base SI units; the unit must be read from `si:unit` before any physical computation.
- The negative velocity is the sign convention of the return stroke.
- The pressure level (≈8.6 bar) is plausible for the pneumatic load cylinder, although the paper does not state it.

**Uncertainty structure.** Two models occur, both fully deterministic:

1. **Constant:** a single value for every sample of every cycle. This applies to the accelerometers, axial force, pneumatic pressure and sound pressure.
2. **Exactly proportional to the reading:** 1% for active current and velocity, 1.5% for the motor currents. A regression of *u* on \|*x*\| gives zero intercept and zero residual.

No Type A component, quantisation term or dynamic component is present.

---

## 5. Channel inventory — PTB SmartUp Unit (1 kHz after cubic interpolation, 1000 samples per cycle)

| Signal | Sensor | Unit | Min | Max | Mean | Std | Standard uncertainty (all cycles) |
|---|---|---|---|---|---|---|---|
| X / Y / Z acceleration | BMA 280 | m s⁻² | −129.3 / −117.2 / −109.4 | 147.1 / 132.5 / 78.9 | 0.022 / 0.852 / **10.34** | 1.48 / 2.08 / 1.36 | time-varying, 0.085–1.56 (median 0.095); **identical for X, Y, Z** |
| X / Y / Z acceleration | MPU 9250 | m s⁻² | −6.76 / −8.53 / −14.55 | 41.4 / 26.6 / 54.7 | 0.19 / 0.30 / **−9.41** | 0.87 / 0.67 / 0.63 | time-varying, 0.043–0.646 (median 0.047) |
| X / Y / Z angular velocity | MPU 9250 | rad s⁻¹ | −2.27 / −0.40 / −28.1 | 0.23 / 0.17 / 87.0 | −0.014 / −0.022 / −0.008 | 0.028 / 0.024 / 0.69 | **identical to MPU acceleration** |
| X / Y / Z magnetic flux density | MPU 9250 | µT | −102.8 / −118.2 / −129.7 | −66.9 / 17.6 / 5.8 | −72.9 / 4.6 / −84.6 | 1.18 / 1.92 / 2.20 | **identical to MPU acceleration** |
| Temperature | MPU 9250 | °C | 30.5 | 58.3 | 52.2 | 2.90 | **identical to MPU acceleration** |

**Physical plausibility.**
- The vertical axes measure gravity, as expected for static accelerometers: the mean magnitude of the acceleration vector is 10.40 m/s² for the BMA 280 and 9.42 m/s² for the MPU 9250, against *g* = 9.81 m/s². The two sensors are mounted with opposite Z orientation.
- The BMA 280 reads 6% high and the MPU 9250 4% low. These offsets are compatible with uncompensated MEMS sensitivity/offset errors and are much larger than the declared uncertainties (≈0.05–0.1 m/s²).
- The magnetic field magnitude (≈110 µT) is about twice the geomagnetic field, plausible next to an electric motor.
- The temperature (≈52 °C) is the IMU die temperature.
- Occasional spikes reach ±130 m/s² on the BMA 280 and +87 rad/s on the MPU Z gyroscope. These spikes are not flagged in the file.

---

## 6. Observations on the data

### 6.1 Measurement uncertainty versus usable signal variability

The *u/s* ratio identifies channels whose declared uncertainty exceeds the variability between cycles:
- **pneumatic pressure: *u/s* ≈ 10.6** — the declared uncertainty is about ten times the cross-cycle spread;
- **velocity: ≈ 4.6**;
- **plain-bearing acceleration: ≈ 1.9**.

In an uncertainty-propagation study these channels dominate the propagated input uncertainty, whatever scaling is applied: z-score normalisation divides value and uncertainty by the same factor and leaves *u/s* unchanged. The motor currents (*u/s* ≈ 0.02) and the sound pressure (≈ 5·10⁻⁵) are, by contrast, almost uncertainty-free relative to their variability.

### 6.2 Behaviour within the 1 s window

Mean over all cycles, in 20 consecutive 50 ms blocks:

| Signal | Block 1 | Block 10 | Block 11 | Block 12 | Block 20 |
|---|---|---|---|---|---|
| Velocity (mm/s) | −199.9 | −200.4 | −200.4 | −200.5 | −200.6 |
| Axial force (kN) | 5.35 | 5.90 | 6.11 | 6.35 | 6.53 |
| Pneumatic pressure (bar) | 8.33 | 8.61 | 8.66 | 8.76 | 8.88 |
| Active current (A) | −2.89 | −3.21 | −3.34 | −3.44 | −3.57 |

Velocity is constant over the window, as the paper states. Load-related signals (force, pressure, active current) are **not** constant: they rise by about 20% across the second, with a step between 0.50 and 0.60 s.

### 6.3 SUU uncertainties

- **Identical arrays across quantities.** The same uncertainty array is stored for all ten MPU 9250 signals, which have four different units (m s⁻², rad s⁻¹, µT, °C). A standard uncertainty of 0.047 °C, 0.047 µT and 0.047 rad/s cannot simultaneously be the result of a per-quantity evaluation; this is most likely a copy error in the dataset-generation script. The three BMA 280 axes also share one array.
- **Magnitude.** The paper reports, for cycle 51, a median interpolation-uncertainty contribution from the data values of 2.236·10⁻³ m/s² (Appendix A, eq. A.2). The file holds medians of 0.095 m/s² (BMA 280) and 0.049 m/s² (MPU 9250) for the same cycle. If the stored values also include the dynamic-calibration term (paper §4.5, Fig. 10), a larger value is expected. The paper does not say which uncertainty components are stored in the published file, so the difference cannot be resolved from the paper alone.

---

## 7. Access through `zema_emc_annotated` (v0.7.1)

The companion library reads only the `ZeMA_DAQ` group: 11 signals, values and uncertainties. `ZeMASamples(SampleSize(n_cycles, datapoints_per_cycle=d))` returns arrays of shape (n_cycles, 11·d) built from `values[:d, cycles]`. **These are the first d samples (first d/2 ms) of each 1 s window, not a downsampling.** It was verified that `ZeMASamples(datapoints_per_cycle=100)` equals the first 100 raw samples of the file.

With `normalize=True`, the library also centres and scales each cycle by its own mean and standard deviation over the full 2000-sample window. The uncertainties are divided by the same per-cycle factor.

The SUU group is not accessible through the library.

---

## 8. Consistency checks: dataset versus Dorst2022

Legend: ✅ consistent · ⚠️ partially consistent or not verifiable · ❌ inconsistent.

| # | Item | Paper (Dorst2022) | Dataset | Verdict |
|---|---|---|---|---|
| C1 | Measurement systems | ZeMA DAQ + SmartUp Unit | groups `ZeMA_DAQ`, `PTB_SUU` | ✅ |
| C2 | ZeMA signals | 11: 3 accelerometers, 1 microphone, 3 motor currents, 4 process signals (force, pressure, velocity, active current) | the same 11 signals, with sensor models in metadata | ✅ |
| C3 | SUU signals | MPU-9250 (9-axis), BMA280 (3-axis); MS5837 excluded as defective | 13 signals: BMA 3 + MPU 3+3+3 + MPU temperature; no MS5837 | ✅ (the temperature comes from the MPU 9250 die sensor, not from the excluded MS5837) |
| C4 | Plain-bearing co-location | SUU sensors on the same holder as the Kistler 8712A5M1 | Kistler 8712A5M1 is labelled "Acceleration plain bearing"; per-cycle RMS correlates with BMA 280 (r = 0.40) and MPU 9250 (r = 0.44) | ✅ (moderate correlation, expected given different bandwidth and resolution) |
| C5 | Cycle selection | every 100th cycle from cycle 51 out of 476,560 | 4766 cycles = len(51…476,560 step 100) | ✅ |
| C6 | Cycle count in metadata | — | `dc:description` and `dc:SizeOrDuration`: "4776 cycles" | ❌ typo (F1) |
| C7 | Window | 1 s of the return stroke | 2000 samples at 2 kHz (ZeMA), 1000 at 1 kHz (SUU) | ✅ |
| C8 | ZeMA sampling | 10 kHz–1 MHz raw, downsampled to 2 kHz | 2000 samples / 1 s; file name `2kHz` | ✅ (the downsampling filter is described neither in the paper nor in the metadata) |
| C9 | SUU sampling / interpolation | interpolated to an equidistant base; ML at 1 kHz; cubic among the schemes | 1000 samples / 1 s; `interpolation_scheme: cubic` | ✅ |
| C10 | MPU magnetometer rate | 100 Hz (thesis p. 49), interpolated | magnetometer series much smoother than accelerometer (2nd-difference/std 0.37 vs 0.91) | ✅ consistent with upsampled 100 Hz data |
| C11 | MPU duplicated-sample bug | MPU accelerations recorded twice per sample; only every second sample used (thesis p. 50) | no equal consecutive samples in MPU acceleration | ✅ (also expected after cubic interpolation) |
| C12 | Test duration | ≈16.5 days, 387.83 h | metadata date range 2021-03-29/2021-04-15; 387.83 h / 476,560 cycles = 2.93 s per cycle = 2.8 s cycle + ≈0.1 s gap stated in the paper | ✅ |
| C13 | Return-stroke velocity | ≈200 mm/s, constant in the window | median −200.4 mm/s, block means −199.9…−200.6 | ✅ |
| C14 | Axial load | 7 kN, constant pulling | per-cycle mean 5.88–6.00 kN (median 5.96); within-window rise 5.35 → 6.53 kN; maximum sample 7.32 kN | ❌ / ⚠️ (F2): lower than nominal and not constant |
| C15 | Units | SI values; D-SI unit strings (SUU protocol) | D-SI `si:unit` on every group; values converted from ADC (`raw_data: false`) | ✅ (kN, mm/s and µT are SI-prefixed, not base units) |
| C16 | Timestamps | ZeMA: none, assumed equidistant; SUU: GNSS absolute | no time datasets stored; implicit equidistant axis | ✅ (SUU absolute timestamps and trigger not published) |
| C17 | ZeMA uncertainties | **not described** | constant or 1% / 1.5% relative values | ⚠️ not verifiable (F3) |
| C18 | SUU uncertainty method | ½ LSB → interpolation (PyDynamic) → dynamic calibration (IIR, GUM-S2) | time-varying values, consistent with propagated uncertainty | ⚠️ method plausible, content not verifiable |
| C19 | SUU uncertainty per quantity | evaluated per sensor/axis | identical arrays across 10 MPU signals with 4 units; identical across BMA axes | ❌ (F4) |
| C20 | SUU uncertainty magnitude | App. A: median interpolation term 2.236·10⁻³ m/s² in cycle 51 | cycle 51 median 0.095 (BMA) / 0.049 (MPU) m/s² | ⚠️ (F5): 20–40× larger; stored components not specified |
| C21 | Downsampled ZeMA size | "less than 2 GB" | 2.67 GB file, but including the SUU group and all uncertainty arrays | ✅ |
| C22 | Data quality | — | no NaN/Inf; no missing cycles | ✅ |

---

## 9. Implications for using the dataset

1. **Report the cycle count as 4766** and cite the cycle-selection rule (every 100th cycle from cycle 51); the metadata value 4776 is a typo.
2. **Read units from `si:unit`.** Force is in kN, velocity in mm/s, magnetic flux density in µT.
3. **Treat the ZeMA DAQ uncertainties as declared Type B values of undocumented provenance.** State this as a limitation in any metrological analysis. Their correlation structure (between samples, cycles and channels) is not specified: the constant values behave like systematic errors, which matters as soon as samples are averaged or differenced.
4. **Do not use the SUU uncertainties for the MPU 9250 angular velocity, magnetic flux density or temperature** until the duplication is clarified with the authors. Treat the per-axis SUU uncertainties with caution as well.
5. **Pneumatic pressure (*u/s* ≈ 10) and velocity (*u/s* ≈ 5) dominate any propagated input uncertainty.** This is a property of the declared uncertainties, not of normalisation.
6. **Within the 1 s window, load-related signals are not stationary** (+20% rise, step at ≈0.5 s). Features computed on part of the window, such as the first 50 ms returned by `ZeMASamples(datapoints_per_cycle=100)`, are not representative of the whole window.
7. **Only one cylinder (axis11) is included.** Generalisation across devices cannot be assessed with this file.
8. **Where *u* > *s*, most of the declared uncertainty must be systematic.** For a random component, *s*² = var(true) + *u*_rand², so *u*_rand ≤ *s*. For Pressure, Velocity and the plain-bearing accelerometer this leaves at least 98.7%, 93.9% and 74% of *u*² common to all cycles (first 100 samples, training cycles). A systematic error common to all cycles cancels in a z-score normalization fitted on the same instrument, so it should not be propagated as independent noise for same-instrument use; it matters only for transfer to another instrument. The dataset gives no repeatability or noise specification, so the random part can only be bounded from the data. See `case_study_A_report_budget.md`.
9. **Consecutive cycles are nearly identical, and the label is the cycle index.** A random train/test split leaves a neighbouring cycle of almost every test cycle in training. Use a blocked split (whole blocks of consecutive cycles) as a check; for this dataset it raises the RMSE of the case-study MLP by only 5–16% (`case_study_A_report_budget.md` §3.2).

---

## 10. Reproducing this report

```bash
source zema_env/bin/activate
python check_dataset_consistency.py | tee dataset_consistency.log
```

The script downloads or reuses the cached HDF5 (hash-checked against `zema_emc_annotated.ZEMA_DATASET_HASH`) and prints:
- the root metadata;
- the full channel inventory (§4–5);
- the cycle-count and duration checks;
- the operating point;
- the within-window profiles (§6.2);
- the SUU checks (§6.3, C4, C10, C11);
- the `zema_emc_annotated` extraction check (§7).

Runtime is a few minutes with the file cached (the hash check of the 2.67 GB file and full-array reads dominate).
