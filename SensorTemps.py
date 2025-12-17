import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load CSV (no header)
path = "Rise1MergedSheet1.csv"
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

# 7. Setup Plot
sensors = ["Temp_In", "Temp_Center", "Temp_Out"]
colors = ["lightgreen", "gold", "plum"]
tdata = df_pos['t'].values

plt.figure(figsize=(10, 6))

# 8. Plot each sensor
for col, color in zip(sensors, colors):
    y = pd.to_numeric(df_pos[col], errors='coerce').interpolate().values
    plt.plot(tdata / 60.0, y, label=f"{col} data", color=color)

# 9. Formatting
plt.xlabel("Time since valve open (min)")
plt.ylabel("Temperature (°C)")
plt.title("Individual sensor temperature")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save or show the plot
plt.savefig("sensor_temperature_plot.png")
plt.show()
