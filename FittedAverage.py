import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the merged CSV with three temperature columns
path = "Rise1MergedSheet1.csv"
df = pd.read_csv(path)

print("Columns detected:", list(df.columns))
print(df.head(5))

# --- Identify columns ---
time_col = None
temp_cols = []

for c in df.columns:
    cl = c.lower()
    if 'time' in cl and time_col is None:
        time_col = c
    if any(k in cl for k in ['temp', 'in', 'center', 'out', 'var']):
        temp_cols.append(c)

# Fallback if needed
if time_col is None:
    time_col = df.columns[0]
if len(temp_cols) < 3:
    # assume next three columns
    temp_cols = df.columns[1:4].tolist()

print(f"\nUsing time column '{time_col}' and temperature columns {temp_cols}")

# Parse time column
df['ts'] = pd.to_datetime(df[time_col], errors='coerce')
df = df.dropna(subset=['ts']).reset_index(drop=True)

# Convert temperature columns to numeric
for c in temp_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Compute average temperature (for fitting)
df['T_avg'] = df[temp_cols].mean(axis=1)

# Determine open time = 08:04 on first date in data
first_date = df['ts'].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")

# Compute relative seconds from open time
df['t'] = (df['ts'] - open_time).dt.total_seconds()

# Keep t>=0
df_pos = df[df['t'] >= 0].copy()

# Baseline T0 = median of first 60s window
early = df[(df['t'] >= -300) & (df['t'] <= 60)]
T0 = early['T_avg'].median() if len(early) > 3 else df_pos['T_avg'].iloc[0]

# Step response model function
def step_response_model(t, delta, tau1, tau2, T0):
    tau1 = np.maximum(tau1, 1e-6)
    tau2 = np.maximum(tau2, 1e-6)
    if abs(tau1 - tau2) < 1e-8:
        tau = (tau1 + tau2) / 2.0
        resp = delta * (1 - (1 + t / tau) * np.exp(-t / tau))
    else:
        resp = delta * (1 - (tau1 * np.exp(-t / tau1) - tau2 * np.exp(-t / tau2)) / (tau1 - tau2))
    return resp + T0

# Data for fitting
tdata = df_pos['t'].values
ydata = df_pos['T_avg'].values

# Initial guesses
SP = 82.0
delta_guess = SP - T0
tau1_guess = tdata.max() * 0.2
tau2_guess = tdata.max() * 0.6
p0 = [delta_guess, tau1_guess, tau2_guess]

# Fit the model
bounds = ([-200.0, 1e-3, 1e-3], [500.0, 1e6, 1e6])
popt, pcov = curve_fit(lambda t, d, t1, t2: step_response_model(t, d, t1, t2, T0),
                       tdata, ydata, p0=p0, bounds=bounds, maxfev=200000)

delta_fit, tau1_fit, tau2_fit = popt
t_fit = np.linspace(tdata.min(), tdata.max(), 800)
y_fit = step_response_model(t_fit, delta_fit, tau1_fit, tau2_fit, T0)

# User custom model
tau1_user, tau2_user = 827.45, 269.94
y_user = step_response_model(t_fit, delta_fit, tau1_user, tau2_user, T0)

# Plot all
plt.figure(figsize=(10,6))
plt.plot(df_pos['t'], df_pos[temp_cols[0]], label='In', color='lightgreen')
plt.plot(df_pos['t'], df_pos[temp_cols[1]], label='Center', color='gold')
plt.plot(df_pos['t'], df_pos[temp_cols[2]], label='Out', color='plum')
plt.plot(t_fit, y_fit, '--', label=f'Fitted avg (τ₁={tau1_fit:.1f}s, τ₂={tau2_fit:.1f}s)')
plt.plot(t_fit, y_user, 'r--', label='User model (τ₁=827.5s, τ₂=269.9s)')
plt.xlabel('Time since valve open (s)')
plt.ylabel('Temperature (°C)')
plt.title('Step response — In/Center/Out temperatures with average model fit')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Fitted parameters:\nΔ = {delta_fit:.3f} °C, τ₁ = {tau1_fit:.2f} s, τ₂ = {tau2_fit:.2f} s")
print(f"Baseline T₀ = {T0:.2f} °C, final predicted T = {T0 + delta_fit:.2f} °C")
