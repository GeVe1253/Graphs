import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preprocessing ---

# Load file
path = "Rise1MergedSheet1.csv" 
# NOTE: This line requires you to have a file named "Rise1MergedSheet1.csv" 
# in the same directory as this script.
try:
    df = pd.read_csv(path, header=None)
except FileNotFoundError:
    print(f"Error: The file '{path}' was not found. Please ensure it is in the correct directory.")
    exit()

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

# Check if data remains after filtering
if df.empty:
    print("Error: No valid data remains after filtering based on time.")
    exit()

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

# Solve for tau1, tau2 using the empirical relations (still calculated for the table output)
# t63 = τ₁ + τ₂
# t86 = 0.5τ₁ + 2τ₂  <-- Setting τ₂ (column 1) as the dominant time constant (coeff 2)
A = np.array([[1, 1], [0.5, 2]])  # Matrix for [tau1, tau2]
b = np.array([t63, t86])
# Check if a solution exists (non-singular matrix)
if np.linalg.det(A) != 0:
    tau1, tau2 = np.linalg.solve(A, b)
else:
    tau1, tau2 = np.nan, np.nan


# --- 3. Model Generation (Skipping model calculation as it's not plotted) ---

# --- 4. Plotting (Updated) ---

plt.figure(figsize=(10, 6))

# Plot raw data
plt.plot(t/60, y, label="Average temp", color="black", linewidth=1.5)

# Removed: The Characteristic-time model (y_model) line

# Plot markers/reference lines
plt.axhline(Tf, color="gray", linestyle=":", alpha=0.6, label="Final steady state")
plt.axhline(T0, color="gray", linestyle=":", alpha=0.6, label="Initial")
plt.axvline(t63/60, color="red", linestyle="--", alpha=0.7, label=f"t63 ({t63:.1f}s)")
plt.axvline(t86/60, color="blue", linestyle="--", alpha=0.7, label=f"t86 ({t86:.1f}s)")

plt.xlabel("Time since valve open (min)")
plt.ylabel("Temperature (°C)")
# Updated Title
plt.title("Step response characteristic times", fontsize=14)
# Updated Legend location
plt.legend(loc="lower right") 
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 5. Manual Table Display (Updated) ---

# Report results (DataFrame now excludes tau1 and tau2)
results = pd.DataFrame({
    "Parameter": ["K", "T0", "t63 (s)", "t86 (s)"], 
    "Value": [K, T0, t63, t86]
})

print("\n--- Characteristic Time Estimates ---")

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
