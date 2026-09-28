# Service Procedure: DockGuard Vehicle Restraint Fault Diagnostics

**Equipment:** DockGuard Vehicle Restraint (all revisions)
**Classification:** Technician only
**Revision:** 2024-10

## Fault code reference

### E01 — No trailer detected
Control panel does not register a trailer at the dock.

**Check:**
1. Sensor array under the dock plate — clear of debris, snow, or ice
2. 24VDC supply at sensor harness connector J-7
3. Sensor cable for rodent damage (common in exterior installations)
4. If sensors test good, replace sensor array (part **DG-SA-01**)

### E02 — Hook will not extend
Restraint hook fails to move toward engagement position.

**Check:**
1. Solenoid valve **SV1** — listen for click on activation
2. Air supply — verify 80–100 psi at restraint inlet
3. Manual override lever position (should be in AUTO)
4. Hook track for mechanical obstruction

### E03 — Hook will not retract
Hook stays engaged after release command.

**Check:**
1. Solenoid valve **SV2** — listen for click
2. Hook track for debris, ice, or deformation
3. Air supply pressure under load (minimum 80 psi during motion)
4. If hook binds, clean and lubricate track with dry PTFE spray — **do not use oil-based lubricants** (attracts contamination)

### E04 — Light tree communication fault
Inside control panel cannot communicate with outside red/green light tree.

**Check:**
1. RS-485 wiring between panel and light tree
2. 120Ω termination resistor at end of data line
3. Light tree transformer output (24VAC)
4. Replace light tree controller (part **DG-LT-CTRL**) if voltages test good but communication fails

### E05 — Emergency stop engaged
E-stop button has been pressed at the dock, office station, or light tree.

**Reset procedure:**
1. Verify dock area is clear of personnel
2. Rotate E-stop button clockwise to release
3. Press the green RESET button on the main control panel
4. If fault persists, check E-stop circuit continuity at terminals **TB3-1 through TB3-4**
