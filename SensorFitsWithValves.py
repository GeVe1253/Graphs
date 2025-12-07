import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fixed parameter for the valve stroke time
TAU_VALVE = 73.333  # 73.333 seconds (approx 22/18 minutes)

# --- Data Loading and Preparation (UNCHANGED) ---

# Load CSV (no header)
path = "Rise1MergedSheet1.csv"
df = pd.read_csv(path, header=None)

# assign columns
df.columns = ["DateTime", "Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]

# parse the DateTime string (dayfirst format like "25/04/2025 08:04:18")
df['ts'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')

# select the date of the first timestamp and set open_time at 08:04:00 that day (as requested)
first_date = df['ts'].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")

# compute t in seconds relative to open_time
df['t'] = (df['ts'] - open_time).dt.total_seconds()

# keep only t >= 0 (data after valve open); if none, fall back to relative to first timestamp
if (df['t'] >= 0).sum() == 0:
    df['t'] = (df['ts'] - df['ts'].iloc[0]).dt.total_seconds()

df_pos = df[df['t'].notna()].copy()
df_pos = df_pos.reset_index(drop=True)

# --- 3-Tau Model Definition (NEW) ---

# Model for an overdamped three-tau step response with a fixed third tau (tv)
def three_tau_fixed_valve(t, K, tau1, tau2, T0):
    tv = TAU_VALVE
    
    # Enforce positive taus inside evaluation
    t1_abs = np.abs(tau1) + 1e-9
    t2_abs = np.abs(tau2) + 1e-9
    
    # Sort the three time constants to assign coefficients A, B, C consistently
    # And slightly adjust if any are too close to prevent singularity
    taus = sorted([t1_abs, t2_abs, tv])
    t_A, t_B, t_C = taus[0], taus[1], taus[2]
    
    # Simple check/adjustment for near-equal roots (highly unlikely but safeguards the math)
    if np.abs(t_A - t_B) < 1e-4: t_B += 0.1
    if np.abs(t_B - t_C) < 1e-4: t_C += 0.1
    if np.abs(t_A - t_C) < 1e-4: t_C += 0.1
    
    # Coefficients based on Partial Fraction Expansion (standard formula for 3 distinct poles)
    A = (t_A**2) / ((t_A - t_B) * (t_A - t_C))
    B = (t_B**2) / ((t_B - t_A) * (t_B - t_C))
    C = (t_C**2) / ((t_C - t_A) * (t_C - t_B))
    
    # Final step response equation
    return T0 + K * (1 - (A * np.exp(-t/t_A) + B * np.exp(-t/t_B) + C * np.exp(-t/t_C)))

# --- Fitting and Plotting ---

sensors = ["Temp_In", "Temp_Center", "Temp_Out"]
colors = ["lightgreen", "gold", "plum"]
results = []

tdata = df_pos['t'].values
plt.figure(figsize=(10,6))

for col, color in zip(sensors, colors):
    y = pd.to_numeric(df_pos[col], errors='coerce').interpolate().values
    
    # initial guesses: Adjusting tau guesses based on expected reduction
    T0_guess = np.median(y[:max(3, min(20, len(y)))])
    K0 = max(0.1, np.nanmax(y) - np.nanmin(y))
    # Using new expected tau values: ~230s and ~1170s
    p0 = [K0, 230.0, 1170.0, T0_guess]
    bounds = ([0.0, 1.0, 1.0, -100.0], [500.0, 1e6, 1e6, 200.0])
    
    try:
        # Use the NEW three-tau function
        popt, pcov = curve_fit(three_tau_fixed_valve, tdata, y, p0=p0, bounds=bounds, maxfev=200000)
    except Exception as e:
        # Fallback (may not be necessary but good practice)
        p0b = [K0, 50.0, 1200.0, T0_guess]
        popt, pcov = curve_fit(three_tau_fixed_valve, tdata, y, p0=p0b, bounds=bounds, maxfev=200000)
        
    y_fit = three_tau_fixed_valve(tdata, *popt)
    residuals = y - y_fit
    variance = np.var(residuals)
    
    # The fitted taus are tau1 and tau2 from popt
    K_fit, tau1_fit, tau2_fit, T0_fit = popt
    
    # Store results including the FIXED tau_valve as tau3
    results.append({
        "Sensor": col,
        "K": float(K_fit),
        "Tau1": float(tau1_fit),  # Fitted Oil/Convection Tau
        "Tau2": float(tau2_fit),  # Fitted Oil/Convection Tau
        "Tau3 (Valve)": TAU_VALVE, # Fixed Valve Tau
        "T0": float(T0_fit),
        "Variance": float(variance)
    })
    
    plt.plot(tdata/60.0, y, label=f"{col} data", color=color)
    plt.plot(tdata/60.0, y_fit, linestyle="--", color=color, alpha=0.7, label=f"{col} fit (3-tau)")

# --- Weighted Average Calculation (MODIFIED) ---

# Compute inverse-variance weights
df_results = pd.DataFrame(results)
df_results['Variance'] = df_results['Variance'].replace(0, np.nan)
weights = 1.0 / df_results['Variance']
weights = weights.fillna(weights.mean())

K_avg = np.sum(weights * df_results['K']) / np.sum(weights)
tau1_avg = np.sum(weights * df_results['Tau1']) / np.sum(weights)
tau2_avg = np.sum(weights * df_results['Tau2']) / np.sum(weights)
T0_avg = np.sum(weights * df_results['T0']) / np.sum(weights)

# The weighted average still requires all three taus (tau_valve is constant)
tau_valve_avg = TAU_VALVE
tau_final = sorted([tau1_avg, tau2_avg, tau_valve_avg]) # Sort for reporting consistency

# plot weighted-average model (must use the three_tau function)
y_avg = three_tau_fixed_valve(tdata, K_avg, tau1_avg, tau2_avg, T0_avg)

plt.plot(tdata/60.0, y_avg, 'k--', linewidth=2, 
         label=f"Weighted avg (τ1={tau_final[0]:.1f}s, τ2={tau_final[1]:.1f}s, τ3={tau_final[2]:.1f}s)")

plt.xlabel("Time since valve open (min)")
plt.ylabel("Temperature (°C)")
plt.title("Individual sensor fits (Three-Tau Model with fixed Valve $\\tau$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Show numeric results (including variances)
df_results_with_avg = df_results.copy()
df_results_with_avg.loc[len(df_results_with_avg)] = [
    "Weighted avg", K_avg, tau1_avg, tau2_avg, tau_valve_avg, T0_avg, np.nan
]

# --- MANUAL TABLE DISPLAY (Replaces tools.display_dataframe_to_user) ---
print("\n--- Per-sensor fits + weighted average (Three-Tau Model) ---")

# Define the format string for the numeric values
NUM_FORMAT = "{:<10.4f}" # Left-aligned, 10 characters wide, 4 decimal places
STR_FORMAT = "{:<15}"   # Left-aligned, 15 characters wide

# Print Header
header = (STR_FORMAT + STR_FORMAT + STR_FORMAT + STR_FORMAT + STR_FORMAT + STR_FORMAT + STR_FORMAT).format(
    "Sensor", "K", "Tau1 (s)", "Tau2 (s)", "Tau3 (Valve)", "T0", "Variance"
)
print("-" * len(header))
print(header)
print("-" * len(header))

# Print data rows
for index, row in df_results_with_avg.iterrows():
    # Handle NaN for the Weighted Avg variance
    variance_str = NUM_FORMAT.format(row['Variance']) if not np.isnan(row['Variance']) else "        N/A"
    
    row_output = (STR_FORMAT + NUM_FORMAT + NUM_FORMAT + NUM_FORMAT + NUM_FORMAT + NUM_FORMAT + STR_FORMAT).format(
        row['Sensor'],
        row['K'],
        row['Tau1'],
        row['Tau2'],
        row['Tau3 (Valve)'],
        row['T0'],
        variance_str
    )
    print(row_output)
print("-" * len(header))

# No need for tools.display_dataframe_to_user() anymore!
# The code ends after the print statements.
