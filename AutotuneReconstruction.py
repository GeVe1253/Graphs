import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess Data ---
path = "Rise1MergedSheet1.csv"

try:
    # Assuming your CSV has no header row
    df = pd.read_csv(path, header=None)
except FileNotFoundError:
    print(f"Error: File '{path}' not found. Please ensure the file is in the working directory.")
    exit()

# Assign columns based on your provided structure
df.columns = ["DateTime", "Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]

# Parse the DateTime string
df['ts'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')

# Select the date of the first timestamp and set open_time at 08:04:00
first_date = df['ts'].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")

# Compute t in seconds relative to open_time
df['t'] = (df['ts'] - open_time).dt.total_seconds()

# Filter out rows before the start time
df = df[df['t'] >= 0].copy().reset_index(drop=True)

# --- 2. Define PV Combinations ---
# Valve 1 PV: Average of In and Center
df['PV1'] = (df['Temp_In'] + df['Temp_Center']) / 2

# Valve 2 PV: Average of Center and Out
df['PV2'] = (df['Temp_Center'] + df['Temp_Out']) / 2

# --- 3. PID Calculation Function (with Anti-Windup) ---
def calculate_pid_response(time, pv, sp, kp, ti, td, bias=0, limits=(0, 100)):
    """
    Calculates the PID output series based on historical PV data.
    Assumes Standard PID Form with Derivative-on-PV and integral anti-windup.
    """
    # Calculate time steps (dt)
    dt = np.diff(time)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 1.0)
    dt = np.maximum(dt, 1e-6)  # Avoid division by zero
    
    error = sp - pv
    output_unclipped = np.zeros_like(pv)
    output = np.zeros_like(pv)
    current_integral = 0
    
    for k in range(len(pv)):
        # 1. Proportional Term (P)
        P = kp * error[k]
        
        # 2. Integral Term (I)
        if ti > 0:
            current_integral += (kp / ti) * error[k] * dt[k]
        
        # Anti-windup (Clamping the integral term to a reasonable range)
        current_integral = np.clip(current_integral, -100, 100) 
        I = current_integral
        
        # 3. Derivative Term (D) - Derivative on PV
        if k > 0 and dt[k] > 0:
            d_pv = (pv[k] - pv[k-1]) / dt[k]
            D = -1.0 * kp * td * d_pv
        else:
            D = 0
            
        # Calculate Total Output (Raw)
        raw_out = P + I + D + bias
        out = np.clip(raw_out, limits[0], limits[1]) # Clipped (physical) output
        
        output_unclipped[k] = raw_out
        output[k] = out
        
    return output, output_unclipped

# --- 4. Define Setpoints and Run Calculation ---

# Setpoint Assumption: Use the mean of the last 50 data points (assuming steady-state target)
SP1 = df['PV1'].tail(50).mean()
SP2 = df['PV2'].tail(50).mean()

print(f"Calculated Setpoint 1: {SP1:.2f}")
print(f"Calculated Setpoint 2: {SP2:.2f}")

# Valve 1 Parameters: Kp=4.6, Ti=506, Td=126
cmd1, raw1 = calculate_pid_response(
    df['t'].values, df['PV1'].values, 
    sp=SP1, kp=4.6, ti=506, td=126, 
    bias=0
)

# Valve 2 Parameters: Kp=2.4, Ti=2019, Td=504
cmd2, raw2 = calculate_pid_response(
    df['t'].values, df['PV2'].values, 
    sp=SP2, kp=2.4, ti=2019, td=504, 
    bias=0
)

df['Valve1_CMD'] = cmd1
df['Valve1_RAW'] = raw1
df['Valve2_CMD'] = cmd2
df['Valve2_RAW'] = raw2

# --- 5. Visualization with Synchronized Scales ---

# Define plot parameters
FIGSIZE = (14, 12) 

# --- Temperature (PV) Scale Synchronization ---
# Find the global minimum and maximum across both PV columns
temp_min = min(df['PV1'].min(), df['PV2'].min())
temp_max = max(df['PV1'].max(), df['PV2'].max())
# Set a buffer (e.g., 5%) for better viewing
TEMP_YLIM_MIN = temp_min * 0.95
TEMP_YLIM_MAX = temp_max * 1.05

# --- Command (MV) Scale Synchronization ---
CMD_YLIM_MIN = 0
CMD_YLIM_MAX = 110 

# Define line thicknesses
PV_LW = 1.0       
CMD_LW = 2.0      
RAW_LW = 1.0      

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

## 🌡️ Plot Valve 1 PID Response
ax1 = axes[0]
ax1.plot(df['t'], df['PV1'], 'b-', linewidth=PV_LW, label='PV1 (In/Center Avg)')
ax1.axhline(y=SP1, color='g', linestyle='--', linewidth=PV_LW, label=f'Setpoint 1 ({SP1:.1f})')
ax1.set_ylabel('Temperature (°C)', color='b')
ax1.grid(True)
# Apply synchronized temperature limits
ax1.set_ylim(TEMP_YLIM_MIN, TEMP_YLIM_MAX)

ax1_twin = ax1.twinx()
ax1_twin.plot(df['t'], df['Valve1_CMD'], 'r-', linewidth=CMD_LW, label='Valve 1 CMD (Clipped)')
ax1_twin.plot(df['t'], df['Valve1_RAW'], 'r--', alpha=0.4, linewidth=RAW_LW, label='Theoretical CMD (Unclipped)')
ax1_twin.set_ylabel('Valve Output (%)', color='r')
# Apply synchronized command limits
ax1_twin.set_ylim(CMD_YLIM_MIN, CMD_YLIM_MAX) 
ax1_twin.legend(loc='lower right')
ax1.set_title('Valve 1: PID Reconstruction (Kp=4.6, Ti=506, Td=126)')

## 🌡️ Plot Valve 2 PID Response
ax2 = axes[1]
ax2.plot(df['t'], df['PV2'], 'b-', linewidth=PV_LW, label='PV2 (Center/Out Avg)')
ax2.axhline(y=SP2, color='g', linestyle='--', linewidth=PV_LW, label=f'Setpoint 2 ({SP2:.1f})')
ax2.set_ylabel('Temperature (°C)', color='b')
ax2.grid(True)
# Apply synchronized temperature limits
ax2.set_ylim(TEMP_YLIM_MIN, TEMP_YLIM_MAX) 

ax2_twin = ax2.twinx()
ax2_twin.plot(df['t'], df['Valve2_CMD'], 'r-', linewidth=CMD_LW, label='Valve 2 CMD (Clipped)')
ax2_twin.plot(df['t'], df['Valve2_RAW'], 'r--', alpha=0.4, linewidth=RAW_LW, label='Theoretical CMD (Unclipped)')
ax2_twin.set_ylabel('Valve Output (%)', color='r')
# Apply synchronized command limits
ax2_twin.set_ylim(CMD_YLIM_MIN, CMD_YLIM_MAX) 
ax2_twin.legend(loc='lower right')
ax2.set_title('Valve 2: PID Reconstruction (Kp=2.4, Ti=2019, Td=504)')

plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.show()
