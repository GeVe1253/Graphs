import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

# --- Configuration ---
TAU_VALVE = 73.333  # Fixed valve tau (for the 3-tau fit)
IMC_LAMBDA_FACTOR = 1.0 # Tune this: Lambda = Factor * (Tau1 + Tau2) (Conservative starting point)

# --- 1. Data Loading (UNCHANGED) ---
path = "Rise1MergedSheet1.csv"
try:
    df = pd.read_csv(path, header=None)
except FileNotFoundError:
    # Creating dummy data for demonstration if file doesn't exist
    print("CSV not found. Generating dummy step response data...")
    t_dummy = np.linspace(0, 3000, 300)
    # Simulation of a 3-tau system: 50s, 230s, 1170s
    y_dummy = 150 * (1 - (0.1*np.exp(-t_dummy/50) + 0.2*np.exp(-t_dummy/230) + 0.7*np.exp(-t_dummy/1170))) + 25
    df = pd.DataFrame({'DateTime': pd.date_range("2025-04-25 08:04:00", periods=300, freq="10s"),
                        'Temp_Center': y_dummy})
    df['Temp_In'] = y_dummy * 0.98
    df['Temp_Out'] = y_dummy * 1.02
    # Add dummy columns to match shape
    for c in ["Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]:
        if c not in df.columns: df[c] = 0

df.columns = ["DateTime", "Temp_In", "Temp_Center", "Temp_Out", "Flag", "Timestamp"]
df['ts'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')

first_date = df['ts'].dt.date.iloc[0]
open_time = pd.to_datetime(f"{first_date} 08:04:00")
df['t'] = (df['ts'] - open_time).dt.total_seconds()

if (df['t'] >= 0).sum() == 0:
    df['t'] = (df['ts'] - df['ts'].iloc[0]).dt.total_seconds()

df_pos = df[df['t'].notna()].copy()
df_pos = df_pos.reset_index(drop=True)

# --- 2. Model Definitions ---

# A. The 3-Tau Model (with Fixed Valve Tau)
def three_tau_fixed_valve(t, K, tau1, tau2, T0):
    tv = TAU_VALVE
    t1_abs = np.abs(tau1) + 1e-9
    t2_abs = np.abs(tau2) + 1e-9
    
    # Sort taus to ensure math stability
    taus = sorted([t1_abs, t2_abs, tv])
    t_A, t_B, t_C = taus
    
    # Avoid singularities
    if np.abs(t_A - t_B) < 1e-4: t_B += 0.1
    if np.abs(t_B - t_C) < 1e-4: t_C += 0.1
    if np.abs(t_A - t_C) < 1e-4: t_C += 0.1
    
    # Partial Fraction Expansion Coeffs
    A = (t_A**2) / ((t_A - t_B) * (t_A - t_C))
    B = (t_B**2) / ((t_B - t_A) * (t_B - t_C))
    C = (t_C**2) / ((t_C - t_A) * (t_C - t_B))
    
    return T0 + K * (1 - (A * np.exp(-t/t_A) + B * np.exp(-t/t_B) + C * np.exp(-t/t_C)))

# B. The 2-Tau Model Equation (for the solver)
def second_order_step(t, tau1, tau2):
    # Normalized step response (0 to 1) for K=1, T0=0
    # y(t) = 1 - (t1*exp(-t/t1) - t2*exp(-t/t2))/(t1-t2)
    if abs(tau1 - tau2) < 1e-5:
        # Critical damping case (unlikely but safe to handle)
        return 1 - (1 + t/tau1) * np.exp(-t/tau1)
    
    num = tau1 * np.exp(-t/tau1) - tau2 * np.exp(-t/tau2)
    den = tau1 - tau2
    return 1 - (num / den)

# --- 3. The Characteristic Times Method (t63/t86) ---

def calculate_taus_from_t63_t86(t_data, y_data):
    """
    Calculates tau1 and tau2 using the t63 (63.2%) and t86 (86.5%) points.
    Returns: tau1, tau2, K, T0, t63_val, t86_val
    """
    # 1. Estimate Steady State (K) and Baseline (T0)
    T0 = np.min(y_data[:20]) # Assume start is min
    K_abs = np.max(y_data[-20:]) - T0 # Total span
    
    # Thresholds
    y_63 = T0 + 0.632 * K_abs
    y_86 = T0 + 0.865 * K_abs
    
    # 2. Find time t63 and t86 using interpolation
    idx_63 = np.where(y_data >= y_63)[0]
    idx_86 = np.where(y_data >= y_86)[0]
    
    if len(idx_63) == 0 or len(idx_86) == 0:
        return np.nan, np.nan, K_abs, T0, np.nan, np.nan

    t63 = t_data[idx_63[0]]
    t86 = t_data[idx_86[0]]

    # 3. Define solver function
    def residuals(p):
        t1, t2 = p
        if t1 <= 0 or t2 <= 0: return [1e5, 1e5] # Penalty for negative taus
        val_63 = second_order_step(t63, t1, t2) - 0.632
        val_86 = second_order_step(t86, t1, t2) - 0.865
        return [val_63, val_86]

    p0 = [t63/2.0, t63/2.0]
    
    try:
        tau_sol = fsolve(residuals, p0)
        tau1, tau2 = sorted(tau_sol)
    except:
        tau1, tau2 = np.nan, np.nan
        
    return tau1, tau2, K_abs, T0, t63, t86

# --- 4. Main Processing Loop ---

sensors = ["Temp_In", "Temp_Center", "Temp_Out"]
colors = ["lightgreen", "gold", "plum"]
results_3tau = []
results_2tau_im = [] 

tdata = df_pos['t'].values
plt.figure(figsize=(12, 7))

for col, color in zip(sensors, colors):
    # Pre-process signal
    y = pd.to_numeric(df_pos[col], errors='coerce').interpolate().values
    
    # --- A. Fit the 3-Tau Model (Original Method) ---
    T0_guess = np.median(y[:10])
    K0 = np.max(y) - np.min(y)
    p0 = [K0, 230.0, 1170.0, T0_guess]
    bounds = ([0.0, 1.0, 1.0, -100.0], [500.0, 5000.0, 5000.0, 200.0])
    
    try:
        popt, _ = curve_fit(three_tau_fixed_valve, tdata, y, p0=p0, bounds=bounds, maxfev=100000)
    except:
        popt = p0 # Fallback
        
    y_fit = three_tau_fixed_valve(tdata, *popt)
    variance = np.var(y - y_fit)
    
    results_3tau.append({
        "Sensor": col,
        "K": popt[0], "Tau1": popt[1], "Tau2": popt[2], "Tau3": TAU_VALVE, "T0": popt[3], "Var": variance
    })
    
    # Use consistent labels: 'Temp_In data' and 'Temp_In fit (3-tau)'
    plt.plot(tdata/60.0, y, label=f"{col} data", color=color, alpha=0.6)
    plt.plot(tdata/60.0, y_fit, linestyle="--", color=color, label=f"{col} fit (3-tau)")

    # --- B. Calculate 2-Tau Characteristic Times (New Request) ---
    t1_char, t2_char, K_char, T0_char, t63, t86 = calculate_taus_from_t63_t86(tdata, y)
    
    if not np.isnan(t1_char):
        tau_c = (t1_char + t2_char) * IMC_LAMBDA_FACTOR 
        
        Kc_imc = (t1_char + t2_char) / (K_char * tau_c)
        Ti_imc = t1_char + t2_char
        Td_imc = (t1_char * t2_char) / (t1_char + t2_char)
        
        results_2tau_im.append({
            "Sensor": col,
            "t63 (s)": t63,
            "t86 (s)": t86,
            "Tau1 (calc)": t1_char,
            "Tau2 (calc)": t2_char,
            "IMC Kc": Kc_imc,
            "IMC Ti": Ti_imc,
            "IMC Td": Td_imc
        })
        
# --- 5. Weighted Average for 3-Tau (LEGEND FIX HERE) ---

df_r = pd.DataFrame(results_3tau)
weights = 1.0 / df_r['Var'].replace(0, 1e-6)
avg_K = np.average(df_r['K'], weights=weights)
avg_tau1 = np.average(df_r['Tau1'], weights=weights)
avg_tau2 = np.average(df_r['Tau2'], weights=weights)
avg_T0 = np.average(df_r['T0'], weights=weights)

# Sort the three time constants for the legend
all_avg_taus = sorted([avg_tau1, avg_tau2, TAU_VALVE])
t_small, t_medium, t_large = all_avg_taus[0], all_avg_taus[1], all_avg_taus[2]

# Plot Average 3-Tau curve
y_avg_3tau = three_tau_fixed_valve(tdata, avg_K, avg_tau1, avg_tau2, avg_T0)

# FIX: Construct the legend label exactly as requested, using sorted taus
legend_label = (
    f"Weighted avg ($\\tau_1$={t_small:.1f}s, "
    f"$\\tau_2$={t_medium:.1f}s, "
    f"$\\tau_3$={t_large:.1f}s)"
)
# Note: Changing 'k-' (solid black) to 'k--' (dashed black) for consistency with the reference image.
plt.plot(tdata/60.0, y_avg_3tau, "k--", linewidth=2, label=legend_label)

# --- 6. Characteristic Method on the AVERAGE Curve ---

# Apply the t63/t86 method to the Weighted Average Curve itself
t1_avg_c, t2_avg_c, K_avg_c, T0_avg_c, t63_avg, t86_avg = calculate_taus_from_t63_t86(tdata, y_avg_3tau)

if not np.isnan(t1_avg_c):
    # Plot the characteristic points
    plt.scatter([t63_avg/60, t86_avg/60], [T0_avg_c + 0.632*K_avg_c, T0_avg_c + 0.865*K_avg_c], 
                color="red", zorder=10, marker="x", s=100, label="t63/t86 Points")
    
    # Calculate Final IMC for the Average Model
    tau_c_avg = (t1_avg_c + t2_avg_c) * 1.0 
    
    Kc_final = (t1_avg_c + t2_avg_c) / (K_avg_c * tau_c_avg)
    Ti_final = t1_avg_c + t2_avg_c
    Td_final = (t1_avg_c * t2_avg_c) / (t1_avg_c + t2_avg_c)

    print("\n" + "="*50)
    print(f"TWO-TAU CHARACTERISTIC METHOD (t63/t86) RESULTS")
    print("="*50)
    print(f"Based on weighted average step response:")
    print(f"Measured t63: {t63_avg:.2f} s")
    print(f"Measured t86: {t86_avg:.2f} s")
    print(f"-"*30)
    print(f"Calculated Tau1: {t1_avg_c:.2f} s")
    print(f"Calculated Tau2: {t2_avg_c:.2f} s  (Dominant)")
    print(f"-"*30)
    print(f"IMC PID SETTINGS (Chien/Fruehauf Parallel Form):")
    print(f"Assumed Lambda (τc): {tau_c_avg:.1f} s")
    print(f"Kc (Gain): {Kc_final:.5f}")
    print(f"τI (Integral): {Ti_final:.2f} s")
    print(f"τD (Derivative): {Td_final:.2f} s")
    print("="*50 + "\n")

plt.title("Individual sensor sits (Three-Tau) and resulting characteristic points (t63/t86)")
plt.xlabel("Time since valve open (min)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# Display DataFrame for individual sensors
df_2tau = pd.DataFrame(results_2tau_im)
import tools 
try:
    tools.display_dataframe_to_user("Characteristic Method & IMC Tuning", df_2tau)
except:
    print(df_2tau)
