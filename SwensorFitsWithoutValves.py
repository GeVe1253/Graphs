import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Load CSV (no header)
path = "Rise1MergedSheet1.csv"
# Try/Except block added just in case the file isn't found locally during testing, 
# but this assumes your file exists as per your code.
try:
    df = pd.read_csv(path, header=None)
except FileNotFoundError:
    print(f"Error: File '{path}' not found. Please ensure the file is in the working directory.")
    exit()

# 2. Assign columns
df.columns = ["DateTime", "Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]

# 3. Parse the DateTime string
df['ts'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')

# 4. Select the date of the first timestamp and set open_time at 08:04:00 that day
first_date = df['ts'].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")

# 5. Compute t in seconds relative to open_time
df['t'] = (df['ts'] - open_time).dt.total_seconds()

# 6. Keep only t >= 0 (data after valve open); fallback if empty
if (df['t'] >= 0).sum() == 0:
    df['t'] = (df['ts'] - df['ts'].iloc[0]).dt.total_seconds()

df_pos = df[df['t'].notna()].copy()
df_pos = df_pos.reset_index(drop=True)

# 7. Define Model (overdamped two-tau step response)
def two_tau(t, K, tau1, tau2, T0):
    # enforce positive taus inside evaluation
    tau1 = np.abs(tau1) + 1e-9
    tau2 = np.abs(tau2) + 1e-9
    if np.abs(tau1 - tau2) < 1e-8:
        tau = 0.5*(tau1 + tau2)
        return T0 + K * (1 - (1 + t/tau) * np.exp(-t/tau))
    return T0 + K * (1 - (tau1*np.exp(-t/tau1) - tau2*np.exp(-t/tau2)) / (tau1 - tau2))

sensors = ["Temp_In", "Temp_Center", "Temp_Out"]
colors = ["lightgreen", "gold", "plum"]
results = []

tdata = df_pos['t'].values
plt.figure(figsize=(10,6))

# 8. Fit data for each sensor
for col, color in zip(sensors, colors):
    y = pd.to_numeric(df_pos[col], errors='coerce').interpolate().values
    
    # initial guesses
    T0_guess = np.median(y[:max(3, min(20, len(y)))])
    K0 = max(0.1, np.nanmax(y) - np.nanmin(y))
    p0 = [K0, 200.0, 1000.0, T0_guess]
    bounds = ([0.0, 1.0, 1.0, -100.0], [500.0, 1e6, 1e6, 200.0])
    
    try:
        popt, pcov = curve_fit(two_tau, tdata, y, p0=p0, bounds=bounds, maxfev=200000)
    except Exception:
        # fallback guess
        p0b = [K0, 50.0, 1200.0, T0_guess]
        try:
            popt, pcov = curve_fit(two_tau, tdata, y, p0=p0b, bounds=bounds, maxfev=200000)
        except Exception:
            popt = p0 # Fallback if fit fails completely
            print(f"Fit failed for {col}, using initial guess.")

    y_fit = two_tau(tdata, *popt)
    residuals = y - y_fit
    variance = np.var(residuals)
    
    results.append({
        "Sensor": col,
        "K": float(popt[0]),
        "Tau1": float(popt[1]),
        "Tau2": float(popt[2]),
        "T0": float(popt[3]),
        "Variance": float(variance)
    })
    
    plt.plot(tdata/60.0, y, label=f"{col} data", color=color)
    plt.plot(tdata/60.0, y_fit, linestyle="--", color=color, alpha=0.7, label=f"{col} fit")

# 9. Compute inverse-variance weights
df_results = pd.DataFrame(results)
df_results['Variance'] = df_results['Variance'].replace(0, np.nan) # avoid div by zero
weights = 1.0 / df_results['Variance']
weights = weights.fillna(weights.mean())

K_avg = np.sum(weights * df_results['K']) / np.sum(weights)
tau1_avg = np.sum(weights * df_results['Tau1']) / np.sum(weights)
tau2_avg = np.sum(weights * df_results['Tau2']) / np.sum(weights)
T0_avg = np.sum(weights * df_results['T0']) / np.sum(weights)
tau1_avg, tau2_avg = sorted([tau1_avg, tau2_avg])

# 10. Plot weighted-average model
y_avg = two_tau(tdata, K_avg, tau1_avg, tau2_avg, T0_avg)
plt.plot(tdata/60.0, y_avg, 'k--', linewidth=2, label=f"Weighted avg (τ1={tau1_avg:.1f}s, τ2={tau2_avg:.1f}s)")

plt.xlabel("Time since valve open (min)")
plt.ylabel("Temperature (°C)")
plt.title("Individual sensor fits (using DateTime string as time base)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 11. FIXED TABLE DISPLAY
df_results_with_avg = df_results.copy()
df_results_with_avg.loc[len(df_results_with_avg)] = ["Weighted avg", K_avg, tau1_avg, tau2_avg, T0_avg, np.nan]

print("\n--- Per-sensor fits + weighted average (Original Two-Tau Model) ---")

# Define widths for alignment
w_name = 15
w_num = 12

# Print Header (Uses string formatting for everything)
header_str = f"{'Sensor':<{w_name}}{'K':<{w_num}}{'Tau1 (s)':<{w_num}}{'Tau2 (s)':<{w_num}}{'T0':<{w_num}}{'Variance':<{w_name}}"
print("-" * len(header_str))
print(header_str)
print("-" * len(header_str))

# Print Rows
for index, row in df_results_with_avg.iterrows():
    # Format variance: if it's NaN (for the average row), print N/A, otherwise print float
    if pd.isna(row['Variance']):
        var_str = "N/A"
    else:
        var_str = f"{row['Variance']:.4f}"

    # Print data row using f-strings for safety and clarity
    print(f"{row['Sensor']:<{w_name}}{row['K']:<{w_num}.4f}{row['Tau1']:<{w_num}.4f}{row['Tau2']:<{w_num}.4f}{row['T0']:<{w_num}.4f}{var_str:<{w_name}}")

print("-" * len(header_str))
