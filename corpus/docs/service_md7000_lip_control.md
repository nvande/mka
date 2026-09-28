# Service Procedure: MD-7000 Lip Control Troubleshooting

**Equipment:** Meridian MD-7000 Hydraulic Dock Leveler
**Classification:** Technician only
**Revision:** 2024-08

## Symptom: Lip will not extend

1. Check lip cylinder wiring harness at junction **J3** for corrosion or loose connections.
2. Verify lip limit switch **LS2** — should actuate at 87° lip angle (±2°).
3. Measure 24VDC at solenoid coil terminals during activation.
4. If voltage is present but cylinder does not actuate, inspect cylinder for internal leakage — replace seal kit (part MD7-LCS-12) or full cylinder if scored.

## Symptom: Lip will not retract

1. Inspect the lip return spring tension. Specification: 40 lb-in torque at rest.
2. Check cylinder rod seals for visible oil weepage.
3. Verify that the gravity-return mode is not disabled in control panel menu **3-2-6**.

## Symptom: Lip falls short of trailer (incomplete extension)

1. Verify dock sensor plate alignment — plate must be parallel to leveler front edge within 1/4".
2. Recalibrate working range via control panel menu path: **3 → 2 → 4 → Run Auto-Calibrate**.
3. Factory mode access code: **2580** (technician only — do not share with operators).
4. If auto-calibration fails, manually set end-of-travel position using menu 3-2-5 with lip fully extended into test trailer.

## Symptom: Lip extends then immediately retracts

Most commonly caused by false trigger on the lip limit switch. Inspect LS2 mounting bracket for looseness or vibration-induced drift. Re-secure with thread locker.

## Related parts
- Lip cylinder seal kit: MD7-LCS-12
- Lip limit switch: MD7-LS2-R
- Lip cylinder complete: MD7-LC-ASY
