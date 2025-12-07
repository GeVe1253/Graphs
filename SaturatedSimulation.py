import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- 1. Define Constants and Parameters ---
TC = 1000.0  # Time scaling factor used for consistent time basis
Ts = 50.0    # Sample Time (s)
Ts_r = Ts / TC # Rescaled Sample Time

# Process Parameters (Two-Tau Model)
tau1 = 309.2
tau2 = 1168.8
tau1_r = tau1 / TC 
tau2_r = tau2 / TC 
K_process = 1.0

# PID Parameters (Aggressive Tuning)
Kp = 2.2
Ti = 1097.29
Td = 203.54
Ki = Kp / Ti # Integral Gain
Kd = Kp * Td # Derivative Gain

# Valve Limits (Saturation)
MV_MIN = 0.0
MV_MAX = 1.25

# Simulation Time
time_sim = 8000.0
time_vec = np.arange(0, time_sim, Ts)
num_steps = len(time_vec)

# --- 2. Discrete Process Model (Controllable Canonical Form) ---
# Defines the system G(s) = K / (tau1*tau2*s^2 + (tau1 + tau2)*s + 1)
a = tau1_r * tau2_r
b = tau1_r + tau2_r

# Continuous State-Space Matrices 
A_c = np.array([[0, 1], [-1/a, -b/a]]) # 2x2
B_c = np.array([[0], [K_process / a]]) # 2x1
C_c = np.array([[1, 0]]) # 1x2
D_c = np.array([[0.0]]) # 1x1 FIX: Explicit 2D array definition

# Convert Continuous State-Space (SS) to Discrete SS
G_d = signal.cont2discrete((A_c, B_c, C_c, D_c), dt=Ts_r)

# --- GUARANTEED FIX: Indexing the output tuple ---
A_d = G_d[0] 
B_d = G_d[1]
C_d = G_d[2]
D_d = G_d[3]

# --- 3. Discrete Simulation Loop ---

# Initialization
PV = np.zeros(num_steps)
MV = np.zeros(num_steps)
SP = np.ones(num_steps) # Unit Step Input (Setpoint = 1.0)
State = np.zeros((2, 1)) # State vector for the process model
integral_term = 0.0
PV_prev = 0.0 # For Derivative action

for k in range(num_steps):
    # Process Variable (PV) reading
    PV[k] = (C_d @ State)[0, 0]
    
    # Error Calculation
    Error = SP[k] - PV[k]
    
    # --- PID Calculation (Discrete Position Algorithm) ---
    
    # 1. Proportional Term
    P_term = Kp * Error
    
    # 2. Derivative Term (Derivative on PV for smoother control)
    if k > 0:
        D_term = - Kd * (PV[k] - PV_prev) / Ts 
    else:
        D_term = 0.0
        
    # Calculate Raw Output (P + I + D)
    raw_mv = P_term + D_term + integral_term 
    
    # 3. Apply Saturation (MV Clipping)
    mv_out = np.clip(raw_mv, MV_MIN, MV_MAX)
    MV[k] = mv_out
    
    # 4. Anti-Windup: Conditional Integration
    if raw_mv == mv_out:
        integral_term += Ki * Error * Ts
    
    # Store previous PV for next D-term calculation
    PV_prev = PV[k]
    
    # Update Process State (New Input is the Saturated MV)
    State = A_d @ State + B_d * mv_out

# --- 4. Visualization ---

FIGSIZE = (14, 8)
PV_LW = 2.0
CV_LW = 1.5

# Title string including the PID parameters
tuning_title = f'PID Tuning: Kp={Kp:.3f}, Ti={Ti:.3f}, Td={Td:.3f} (with Saturation/Anti-Windup)'

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
time_out_s = time_vec 

# 🌡️ Plot Process Variable (PV)
ax1 = axes[0]
ax1.plot(time_out_s, PV, 'b-', linewidth=PV_LW, label='Process Variable (PV)')
ax1.axhline(y=1.0, color='g', linestyle='--', linewidth=1.0, label='Setpoint (SP)')
ax1.set_ylabel('Process Output Value', color='b')
ax1.grid(True)
ax1.set_title(r'PV Response: Discrete Two-Tau Model' + '\n' + tuning_title)
ax1.legend(loc='lower right')

# 📈 Plot Control Variable (CV)
ax2 = axes[1]
ax2.plot(time_out_s, MV, 'r-', linewidth=CV_LW, label='Control Variable (MV) - Clipped')
ax2.axhline(y=MV_MAX, color='orange', linestyle=':', linewidth=0.5, label=f'MV Max ({MV_MAX})')
ax2.axhline(y=MV_MIN, color='orange', linestyle=':', linewidth=0.5, label=f'MV Min ({MV_MIN})')
ax2.set_ylabel(f'Control Signal Value (MV) [{MV_MIN}-{MV_MAX}]', color='r')
ax2.set_xlabel('Time (seconds)')
ax2.grid(True)
ax2.set_title(tuning_title)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()
