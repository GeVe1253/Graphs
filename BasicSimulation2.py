import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- TIME RESCALING FACTOR ---
TC = 1000.0 
N_filter = 10.0 # Standard filter coefficient for the derivative term

# --- 1. Define Model Parameters and Conversions ---

# Two-Tau Process Model
K = 1.0
tau1 = 309.2 
tau2 = 1168.8 
tau1_r = tau1 / TC 
tau2_r = tau2 / TC

# PID Controller Values (Ti/Td form)
Kp = 1.33 
Ti = 1097.29
Td = 203.54
Ti_r = Ti / TC 
Td_r = Td / TC 

# Conversion to standard PID form (Kp, Ki, Kd) using RESCALED time
Ki = Kp / Ti_r
Kd = Kp * Td_r

# --- 2. Define Numerator/Denominator Coefficients (Proper PID) ---

# To avoid improper transfer functions (which cause the ValueError), 
# we implement a standard derivative filter (N=10).

# a) Process Coefficients G(s)
G_num = np.array([K])
G_den = np.array([tau1_r * tau2_r, tau1_r + tau2_r, 1])

# b) Proper PID Controller Denominator C(s)
# C(s) = (Kd*s^2 + Kp*s + Ki) / (s * (Td/N * s + 1))
C_num = np.array([Kd, Kp, Ki]) # Numerator remains the same
C_den = np.convolve(np.array([1, 0]), np.array([Td_r/N_filter, 1])) 
# C_den now is: [Td_r/N, 1, 0]

# --- 3. Calculate Open-Loop Components and T_cv Numerator ---
L_num = np.convolve(C_num, G_num)
L_den = np.convolve(C_den, G_den) # L_den length is now 5
T_cv_num_raw = np.convolve(C_num, G_den) # T_cv_num_raw length is now 5

# --- 4. Align Polynomial Orders for T_den Calculation ---
max_initial_len = max(len(L_den), len(L_num), len(T_cv_num_raw))

L_num_padded = np.pad(L_num, (max_initial_len - len(L_num), 0), 'constant')
L_den_padded = np.pad(L_den, (max_initial_len - len(L_den), 0), 'constant')
T_den_raw = L_den_padded + L_num_padded

# --- 5. Trimming Trailing Zeros (s=0 root cancellation) ---
T_den_trimmed = np.trim_zeros(T_den_raw, 'b') # Denominator now has correct final length (5)
final_den_len = len(T_den_trimmed)

# --- 6. Define Process Output (PV) Transfer Function T_pv(s) ---
T_pv_num_raw = np.trim_zeros(L_num_padded, 'b')
T_pv_num_final = np.pad(T_pv_num_raw, (final_den_len - len(T_pv_num_raw), 0), 'constant')
T_pv = signal.TransferFunction(T_pv_num_final, T_den_trimmed)

# --- 7. Define Control Variable (CV) Transfer Function T_cv(s) ---
T_cv_num_raw = np.trim_zeros(T_cv_num_raw, 'b')
T_cv_num_final = np.pad(T_cv_num_raw, (final_den_len - len(T_cv_num_raw), 0), 'constant')
T_cv = signal.TransferFunction(T_cv_num_final, T_den_trimmed)

# --- 8. Simulate Step Responses ---
time_sim_r = 8000.0 / TC 
time_vec_r = np.linspace(0, time_sim_r, 500)

time_out_r, pv_response = signal.step(T_pv, T=time_vec_r)
time_out_r, cv_response = signal.step(T_cv, T=time_vec_r)

# Convert Rescaled Time back to Seconds for plotting
time_out_s = time_out_r * TC

# --- 9. Visualization ---
FIGSIZE = (14, 8)
PV_LW = 2.0
CV_LW = 1.5

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

# 🌡️ Plot Process Variable (PV)
ax1 = axes[0]
ax1.plot(time_out_s, pv_response, 'b-', linewidth=PV_LW, label='Process Variable (PV)')
ax1.axhline(y=1.0, color='g', linestyle='--', linewidth=1.0, label='Setpoint (SP)')
ax1.set_ylabel('Process Output Value', color='b')
ax1.grid(True)
ax1.set_title(r'PV Response: Two-Tau Model ($\mathit{\tau_1=309.2, \tau_2=1168.8}$) with Proper PID Control')
ax1.legend(loc='lower right')

# 📈 Plot Control Variable (CV)
ax2 = axes[1]
ax2.plot(time_out_s, cv_response, 'r-', linewidth=CV_LW, label='Control Variable (CV)')
ax2.axhline(y=0.0, color='k', linestyle=':', linewidth=0.5, label='Bias/Zero Line')
ax2.set_ylabel('Control Signal Value (MV)', color='r')
ax2.set_xlabel('Time (seconds)')
ax2.grid(True)
ax2.set_title(f'CV Response: Controller Output (Kp={Kp}, Ti={Ti}, Td={Td}) - Filtered')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('pid_new_tuning_simulation.png')
print("Simulation complete. The step response plot is saved as 'pid_new_tuning_simulation.png'.")
