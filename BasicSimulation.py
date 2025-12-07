import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- 1. Define Model Parameters (as specified by user) ---

# Two-Tau Process Model
K = 1.0     # Process Gain
tau1 = 309.2 # Time constant 1
tau2 = 1168.8 # Time constant 2

# PID Controller Values (Kp, Ti, Td form)
Kp = 1.33
Ti = 1097.29
Td = 203.54

# --- 2. Convert to Kp, Ki, Kd Form ---

# Ki = Kp / Ti
Ki = Kp / Ti
# Kd = Kp * Td
Kd = Kp * Td

# --- 3. Define Numerator/Denominator Coefficients ---

# a) Process Coefficients G(s)
# Denominator: tau1*tau2*s^2 + (tau1 + tau2)*s + 1
G_num = np.array([K])
G_den = np.array([tau1 * tau2, tau1 + tau2, 1])

# b) PID Controller Coefficients C(s)
# C(s) = (Kd*s^2 + Kp*s + Ki) / s
C_num = np.array([Kd, Kp, Ki])
C_den = np.array([1, 0])

# --- 4. Calculate Open-Loop Transfer Function L(s) = C(s) * G(s) ---

# Multiply transfer functions by convolving their numerators and denominators
L_num = np.convolve(C_num, G_num)
L_den = np.convolve(C_den, G_den)

# --- 5. Calculate Closed-Loop Transfer Function T(s) = L(s) / (1 + L(s)) ---

# T_num = L_num
T_num = L_num

# T_den = L_den + L_num (requires polynomial addition/padding)
max_len = max(len(L_num), len(L_den))
L_num_padded = np.pad(L_num, (max_len - len(L_num), 0), 'constant')
L_den_padded = np.pad(L_den, (max_len - len(L_den), 0), 'constant')
T_den = L_den_padded + L_num_padded
T_num_final = np.pad(T_num, (len(T_den) - len(T_num), 0), 'constant')

# Create the Closed-Loop Transfer Function object
T_closed_loop = signal.TransferFunction(T_num_final, T_den)

# --- 6. Simulate Step Response ---

# Define time vector (extended due to large time constants)
time_vec = np.linspace(0, 8000, 500)

# Calculate the step response (assuming a unit step input)
time_out, response_out = signal.step(T_closed_loop, T=time_vec)

# --- 7. Plot the Results ---

plt.figure(figsize=(10, 6))
plt.plot(time_out, response_out, label='Closed-Loop Step Response')
plt.plot(time_out, np.ones_like(time_out), 'r--', label='Set Point (Target = 1.0)')
plt.title(f'Step Response (tau1={tau1}, tau2={tau2}) with PID (Kp={Kp}, Ti={Ti}, Td={Td})')
plt.xlabel('Time (s)')
plt.ylabel('Output Value')
plt.ylim(0, 1.3)
plt.grid(True)
plt.legend()
plt.savefig('pid_large_tau_simulation.png')
print("Simulation complete. The step response plot is saved as 'pid_large_tau_simulation.png'.")
