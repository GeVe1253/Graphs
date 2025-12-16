import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- 1. Define Constants and Parameters ---
TC = 1000.0  # Time scaling factor used for consistent time basis
Ts = 50.0    # Sample Time (s)
Ts_r = Ts / TC # Rescaled Sample Time

# Process Parameters (Two-Tau Model) - SHARED
tau1 = 309.2
tau2 = 1168.8
tau1_r = tau1 / TC
tau2_r = tau2 / TC
K_process = 1.0

# PID Parameters (Two Separate Controllers)
# PID_A (FAST Tuning) - PID 1
PID_A_PARAMS = {'Kp': 2.2, 'Ti': 1168.85, 'Td': 157.37}
Kp_A = PID_A_PARAMS['Kp']
Ti_A = PID_A_PARAMS['Ti']
Td_A = PID_A_PARAMS['Td']
Ki_A = Kp_A / Ti_A # Integral Gain
Kd_A = Kp_A * Td_A # Derivative Gain

# PID_B (SLOW Tuning) - PID 2
PID_B_PARAMS = {'Kp': 2.2, 'Ti': 1478.03, 'Td': 0.00}
Kp_B = PID_B_PARAMS['Kp']
Ti_B = PID_B_PARAMS['Ti']
Td_B = PID_B_PARAMS['Td']
Ki_B = Kp_B / Ti_B # Integral Gain
Kd_B = Kp_B * Td_B # Derivative Gain

# Valve Limits (Saturation) - Applied INDIVIDUALLY to each MV
MV_MIN = 0.0
MV_MAX = 0.625 # HALF the original max (1.25)

# Simulation Time
time_sim = 8000.0
time_vec = np.arange(0, time_sim, Ts)
num_steps = len(time_vec)

# --- 2. Discrete Process Model (Controllable Canonical Form) ---
# Defines the system G(s) = K / (tau1*tau2*s^2 + (tau1 + tau2)*s + 1)
a = tau1_r * tau2_r
b = tau1_r + tau2_r

# Continuous State-Space Matrices
A_c = np.array([[0, 1], [-1/a, -b/a]]) # 2x2
B_c = np.array([[0], [K_process / a]]) # 2x1 (B_c now handles the *sum* of the MVs)
C_c = np.array([[1, 0]]) # 1x2
D_c = np.array([[0.0]]) # 1x1

# Convert Continuous State-Space (SS) to Discrete SS
G_d = signal.cont2discrete((A_c, B_c, C_c, D_c), dt=Ts_r)

A_d = G_d[0]
B_d = G_d[1]
C_d = G_d[2]
D_d = G_d[3]

# --- 3. Discrete Simulation Loop ---

# Initialization
PV = np.zeros(num_steps)
SP = np.ones(num_steps) # Unit Step Input (Setpoint = 1.0)
State = np.zeros((2, 1)) # State vector for the process model

# Controller A (PID 1) Initialization
MV_A = np.zeros(num_steps)
integral_term_A = 0.0
PV_prev_A = 0.0

# Controller B (PID 2) Initialization
MV_B = np.zeros(num_steps)
integral_term_B = 0.0
PV_prev_B = 0.0

for k in range(num_steps):
    # Process Variable (PV) reading - SHARED
    PV[k] = (C_d @ State)[0, 0]

    # Error Calculation - SHARED
    Error = SP[k] - PV[k]

    # --- PID 1 (FAST) Calculation ---
    # P Term A
    P_term_A = Kp_A * Error

    # D Term A (Derivative on PV)
    if k > 0:
        D_term_A = - Kd_A * (PV[k] - PV_prev_A) / Ts
    else:
        D_term_A = 0.0

    # Calculate Raw Output A
    raw_mv_A = P_term_A + D_term_A + integral_term_A

    # Apply Saturation A (MV Clipping)
    mv_out_A = np.clip(raw_mv_A, MV_MIN, MV_MAX)
    MV_A[k] = mv_out_A

    # Anti-Windup A: Conditional Integration
    if raw_mv_A == mv_out_A:
        integral_term_A += Ki_A * Error * Ts

    # Store previous PV for next D-term calculation
    PV_prev_A = PV[k]

    # --- PID 2 (SLOW) Calculation ---
    # P Term B
    P_term_B = Kp_B * Error

    # D Term B (Derivative on PV)
    if k > 0:
        D_term_B = - Kd_B * (PV[k] - PV_prev_B) / Ts
    else:
        D_term_B = 0.0

    # Calculate Raw Output B
    raw_mv_B = P_term_B + D_term_B + integral_term_B

    # Apply Saturation B (MV Clipping)
    mv_out_B = np.clip(raw_mv_B, MV_MIN, MV_MAX)
    MV_B[k] = mv_out_B

    # Anti-Windup B: Conditional Integration
    if raw_mv_B == mv_out_B:
        integral_term_B += Ki_B * Error * Ts

    # Store previous PV for next D-term calculation
    PV_prev_B = PV[k]

    # --- Process Update ---
    # Total input to the process is the sum of the two saturated control variables
    mv_total = mv_out_A + mv_out_B

    # Update Process State (New Input is the Saturated MV_total)
    State = A_d @ State + B_d * mv_total

# --- 4. Visualization ---

FIGSIZE = (14, 10)
PV_LW = 2.5
CV_LW = 1.5
MV2_COLOR = 'maroon' # New color for PID 2

# Title string for PV plot (Graph 1)
PV_TITLE = r'PV Response: Two-Tau Model ($\mathit{\tau_1=' + str(tau1) + r', \tau_2=' + str(tau2) + r'}$) with Dual PID Control'

# Title string for PID 1 (Graph 2)
PID1_TITLE = f'PID 1 (FAST) Tuning: Kp={Kp_A:.3f}, Ti={Ti_A:.3f}, Td={Td_A:.3f} (with Saturation/Anti-Windup)'

# Title string for PID 2 (Graph 3)
PID2_TITLE = f'PID 2 (SLOW) Tuning: Kp={Kp_B:.3f}, Ti={Ti_B:.3f}, Td={Td_B:.3f} (with Saturation/Anti-Windup)'


fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
time_out_s = time_vec

# 🌡️ Plot Process Variable (PV)
ax1 = axes[0]
ax1.plot(time_out_s, PV, 'b-', linewidth=PV_LW, label='Process Variable (PV)')
ax1.axhline(y=1.0, color='g', linestyle='--', linewidth=1.0, label='Setpoint (SP)')
ax1.set_ylabel('Process Output Value', color='b')
ax1.grid(True)
ax1.set_title(PV_TITLE)
ax1.legend(loc='lower right')

# 📈 Plot Control Variable A (MV_A) - PID 1
ax2 = axes[1]
ax2.plot(time_out_s, MV_A, 'r-', linewidth=CV_LW, label='MV_A (PID 1 - FAST) - Clipped')
ax2.axhline(y=MV_MAX, color='red', linestyle=':', linewidth=0.5, label=f'MV Max ({MV_MAX})')
ax2.axhline(y=MV_MIN, color='red', linestyle=':', linewidth=0.5, label=f'MV Min ({MV_MIN})')
ax2.set_ylabel(f'MV_A Value [{MV_MIN}-{MV_MAX}]', color='r')
ax2.set_title(PID1_TITLE)
ax2.grid(True)
ax2.legend(loc='lower right')

# 📉 Plot Control Variable B (MV_B) - PID 2
ax3 = axes[2]
ax3.plot(time_out_s, MV_B, color=MV2_COLOR, linestyle='-', linewidth=CV_LW, label='MV_B (PID 2 - SLOW) - Clipped')
ax3.axhline(y=MV_MAX, color=MV2_COLOR, linestyle=':', linewidth=0.5, label=f'MV Max ({MV_MAX})')
ax3.axhline(y=MV_MIN, color=MV2_COLOR, linestyle=':', linewidth=0.5, label=f'MV Min ({MV_MIN})')
ax3.set_ylabel(f'MV_B Value [{MV_MIN}-{MV_MAX}]', color=MV2_COLOR)
ax3.set_xlabel('Time (seconds)')
ax3.set_title(PID2_TITLE)
ax3.grid(True)
ax3.legend(loc='lower right')

plt.tight_layout()
plt.show()
