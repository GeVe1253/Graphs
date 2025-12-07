import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess Data ---
path = "Rise1MergedSheet1.csv"
try:
    df = pd.read_csv(path, header=None)
except FileNotFoundError:
    print(f"Error: File '{path}' not found. Please ensure the file is in the working directory.")
    exit()

df.columns = ["DateTime", "Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]
df['ts'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
first_date = df['ts'].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")
df['t'] = (df['ts'] - open_time).dt.total_seconds()
df = df[df['t'] >= 0].copy().reset_index(drop=True)

# Define PV Combinations
df['PV1'] = (df['Temp_In'] + df['Temp_Center']) / 2
df['PV2'] = (df['Temp_Center'] + df['Temp_Out']) / 2

# --- 2. PID Calculation Function (TZN4M-14C Simulation) ---
def calculate_pid_response_filtered(time, pv, sp, kp, ti, td, filter_tau=3.0, bias=0, limits=(0, 100)):
    """
    Calculates the PID output series simulating a controller with a First-Order Lag (FOL) 
    filter on the Process Variable (PV) for derivative calculation.
    """
    dt = np.diff(time)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 1.0)
    dt = np.maximum(dt, 1e-6)  # Avoid division by zero
    
    error = sp - pv
    output_unclipped = np.zeros_like(pv)
    output = np.zeros_like(pv)
    current_integral = 0
    pv_filtered = np.zeros_like(pv)

    # Calculate filter coefficients
    # The coefficients are based on the continuous filter equation discretized.
    alpha = dt / (filter_tau + dt) 
    
    for k in range(len(pv)):
        # 1. Filter the PV signal (only if filter_tau > 0)
        if k == 0 or filter_tau <= 0:
            pv_filtered[k] = pv[k]
        else:
            # First-Order Lag Filter (TZN4M-14C Digital Filter simulation)
            pv_filtered[k] = (1 - alpha[k]) * pv_filtered[k-1] + alpha[k] * pv[k]
        
        # 2. Proportional Term (P)
        P = kp * error[k]
        
        # 3. Integral Term (I)
        if ti > 0:
            current_integral += (kp / ti) * error[k] * dt[k]
        
        # Anti-windup 
        current_integral = np.clip(current_integral, -100, 100) 
        I = current_integral
        
        # 4. Derivative Term (D) - CALCULATED using Filtered PV
        if k > 0 and dt[k] > 0:
            # D uses the rate of change of the FILTERED PV
            d_pv_filtered = (pv_filtered[k] - pv_filtered[k-1]) / dt[k]
            D = -1.0 * kp * td * d_pv_filtered
        else:
            D = 0
            
        # Calculate Total Output (Raw and Clipped)
        raw_out = P + I + D + bias
        out = np.clip(raw_out, 0, 100)
        
        output_unclipped[k] = raw_out
        output[k] = out
        
    return output, output_unclipped

# --- 3. Define Setpoints and Run Calculation ---

# Setpoint Assumption (mean of the last 50 data points)
SP1 = df['PV1'].tail(50).mean()
SP2 = df['PV2'].tail(50).mean()

print(f"Calculated Setpoint 1: {SP1:.2f}")
print(f"Calculated Setpoint 2: {SP2:.2f}")

# IMPORTANT: Setting filter_tau=3.0 seconds (a common default value for industrial filters)
FILTER_TAU = 3.0 

# Valve 1 Parameters: Kp=4.6, Ti=506, Td=126
cmd1, raw1 = calculate_pid_response_filtered(
    df['t'].values, df['PV1'].values, 
    sp=SP1, kp=4.6, ti=506, td=126, 
    filter_tau=FILTER_TAU, bias=0
)

# Valve 2 Parameters: Kp=2.4, Ti=2019, Td=504
cmd2, raw2 = calculate_pid_response_filtered(
    df['t'].values, df['PV2'].values, 
    sp=SP2, kp=2.4, ti=2019, td=504, 
    filter_tau=FILTER_TAU, bias=0
)

df['Valve1_CMD_F'] = cmd1
df['Valve1_RAW_F'] = raw1
df['Valve2_CMD_F'] = cmd2
df['Valve2_RAW_F'] = raw2

# --- 4. Visualization with Synchronized Scales ---

# Define plot parameters
FIGSIZE = (14, 12) 
PV_LW = 1.0; CMD_LW = 2.0; RAW_LW = 1.0 

# Scale Synchronization
temp_min = min(df['PV1'].min(), df['PV2'].min())
temp_max = max(df['PV1'].max(), df['PV2'].max())
TEMP_YLIM_MIN = temp_min * 0.95
TEMP_YLIM_MAX = temp_max * 1.05
CMD_YLIM_MIN = 0
CMD_YLIM_MAX = 110 

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

## 🌡️ Plot Valve 1 PID Response
ax1 = axes[0]
ax1.plot(df['t'], df['PV1'], 'b-', linewidth=PV_LW, label='PV1 (In/Center Avg)')
ax1.axhline(y=SP1, color='g', linestyle='--', linewidth=PV_LW, label=f'Setpoint 1 ({SP1:.1f})')
ax1.set_ylabel('Temperature (°C)', color='b')
ax1.grid(True)
ax1.set_ylim(TEMP_YLIM_MIN, TEMP_YLIM_MAX)

ax1_twin = ax1.twinx()
ax1_twin.plot(df['t'], df['Valve1_CMD_F'], 'r-', linewidth=CMD_LW, label='Valve 1 CMD (Filtered)')
ax1_twin.plot(df['t'], df['Valve1_RAW_F'], 'r--', alpha=0.4, linewidth=RAW_LW, label='Theoretical CMD (Unclipped)')
ax1_twin.set_ylabel('Valve Output (%)', color='r')
ax1_twin.set_ylim(CMD_YLIM_MIN, CMD_YLIM_MAX) 
ax1_twin.legend(loc='lower right')
ax1.set_title(f'Valve 1: TZN4M-14C Simulation (Kp=4.6, Ti=506, Td=126, Filter={FILTER_TAU}s)')

## 🌡️ Plot Valve 2 PID Response
ax2 = axes[1]
ax2.plot(df['t'], df['PV2'], 'b-', linewidth=PV_LW, label='PV2 (Center/Out Avg)')
ax2.axhline(y=SP2, color='g', linestyle='--', linewidth=PV_LW, label=f'Setpoint 2 ({SP2:.1f})')
ax2.set_ylabel('Temperature (°C)', color='b')
ax2.grid(True)
ax2.set_ylim(TEMP_YLIM_MIN, TEMP_YLIM_MAX) 

ax2_twin = ax2.twinx()
ax2_twin.plot(df['t'], df['Valve2_CMD_F'], 'r-', linewidth=CMD_LW, label='Valve 2 CMD (Filtered)')
ax2_twin.plot(df['t'], df['Valve2_RAW_F'], 'r--', alpha=0.4, linewidth=RAW_LW, label='Theoretical CMD (Unclipped)')
ax2_twin.set_ylabel('Valve Output (%)', color='r')
ax2_twin.set_ylim(CMD_YLIM_MIN, CMD_YLIM_MAX) 
ax2_twin.legend(loc='lower right')
ax2.set_title(f'Valve 2: TZN4M-14C Simulation (Kp=2.4, Ti=2019, Td=504, Filter={FILTER_TAU}s)')

plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.show()
