import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preprocessing ---

# Load file
path = "Rise1MergedSheet1.csv" 
df = pd.read_csv(path, header=None)
df.columns = ["DateTime", "Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]
# Convert DateTime string to datetime objects
df["ts"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")

# Set reference time = 08:04:00 on same date as first record
first_date = df["ts"].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")
# Calculate time relative to the step input (in seconds)
df["t"] = (df["ts"] - open_time).dt.total_seconds()
# Filter out data before the step and drop any rows where time conversion failed
df = df[df["t"] >= 0].dropna(subset=["t"]).reset_index(drop=True)

# Average temperature
df["Temp_Avg"] = df[["Temp_In", "Temp_Center", "Temp_Out"]].mean(axis=1)

# Basic preprocessing
t = df["t"].values
y = df["Temp_Avg"].values

# --- 2. Parameter Estimation ---

# Initial and final steady states
T0 = np.median(y[:10])
Tf = np.median(y[-10:])
K = Tf - T0

# Normalized response
yn = (y - T0) / K

# Find 63.2% and 86.5% times using linear interpolation
def find_time_for_fraction(frac):
    """Finds the time 't' when the normalized response 'yn' reaches a given fraction 'frac'."""
    idx = np.where(yn >= frac)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return t[0]
    # Linear interpolation between the point before (i-1) and the point at (i)
    x0, x1 = yn[i-1], yn[i]
    t0, t1 = t[i-1], t[i]
    return t0 + (frac - x0) * (t1 - t0) / (x1 - x0)

t63 = find_time_for_fraction(0.632)
t86 = find_time_for_fraction(0.865)

# Solve for tau1, tau2 using the empirical relations:
# t63 = τ₁ + τ₂
# t86 = 0.5τ₁ + 2τ₂  <-- Setting τ₂ (column 1) as the dominant time constant (coeff 2)
A = np.array([[1, 1], [0.5, 2]])  # Matrix for [tau1, tau2]
b = np.array([t63, t86])
tau1, tau2 = np.linalg.solve(A, b)

# --- 3. Model Generation ---

def two_tau_model(t_vals, K, T0, tau1, tau2):
    """
    Calculates the overdamped second-order step response.
    y(t) = K * [1 - (tau1*e^(-t/tau1) - tau2*e^(-t/tau2))/(tau1 - tau2)] + T0
    """
    # Handle edge case where tau1 == tau2 (critically damped)
    if abs(tau1 - tau2) < 1e-6:
        tau = (tau1 + tau2) / 2
        # Critically damped formula: 1 - e^(-t/tau)(1 + t/tau)
        response = 1 - np.exp(-t_vals/tau) * (1 + t_vals/tau)
    else:
        # Overdamped formula (standard second-order response)
        term1 = tau1 * np.exp(-t_vals / tau1)
        term2 = tau2 * np.exp(-t_vals / tau2)
        response = 1 - (term1 - term2) / (tau1 - tau2)
    
    return T0 + K * response

# Calculate model values across the full time range
y_model = two_tau_model(t, K, T0, tau1, tau2)

# --- 4. Plotting ---

plt.figure(figsize=(10, 6))

# Plot raw data
plt.plot(t/60, y, label="Average temp", color="black", linewidth=1.5)

# Plot the new Characteristic-time model
plt.plot(t/60, y_model, color=(0.6, 0.2, 0.8), linestyle="--", linewidth=2, 
         label="Characteristic-time model") # Purple color

# Plot markers/reference lines
plt.axhline(Tf, color="gray", linestyle=":", alpha=0.6, label="Final steady state")
plt.axhline(T0, color="gray", linestyle=":", alpha=0.6, label="Initial")
plt.axvline(t63/60, color="red", linestyle="--", alpha=0.7, label=f"t63 ({t63:.1f}s)")
plt.axvline(t86/60, color="blue", linestyle="--", alpha=0.7, label=f"t86 ({t86:.1f}s)")

plt.xlabel("Time since valve open (min)")
plt.ylabel("Temperature (°C)")
plt.title(f"Step response characteristic times and the Two-Tau model (τ₁={tau1:.1f}s, τ₂={tau2:.1f}s)", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 5. Manual Table Display (Replaces caas_jupyter_tools) ---

# Report results (DataFrame for easy structure/access)
results = pd.DataFrame({
    "Parameter": ["K", "T0", "t63 (s)", "t86 (s)", "τ₁ (sec)", "τ₂ (dom)"], 
    "Value": [K, T0, t63, t86, tau1, tau2]
})

print("\n--- Traditional Two-Tau Parameter Estimates ---")

# Define the format string for the numeric values
NUM_FORMAT = "{:<12.3f}" # Left-aligned, 12 characters wide, 3 decimal places
STR_FORMAT = "{:<15}"

# Print Header
header = (STR_FORMAT + STR_FORMAT).format(
    "Parameter", "Value"
)
print("-" * len(header))
print(header)
print("-" * len(header))

# Print data rows
for index, row in results.iterrows():
    row_output = (STR_FORMAT + NUM_FORMAT).format(
        row['Parameter'],
        row['Value'],
    )
    print(row_output)
print("-" * len(header))

