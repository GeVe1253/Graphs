import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================================
# --- USER EDITABLE VARIABLES ---
# ==========================================================

# PROCESS PARAMETERS (G(s) = K_PROC / [(tau1*s + 1)*(tau2*s + 1)])
TAU_1 = 309.2       # Radiator Lag (τ₁)
TAU_2 = 1168.8      # Convection/Room Lag (τ₂)
K_PROC = 51.88        # Process Gain (K_p - used in G(s))

# CONTROLLER PARAMETERS (C(s) = Kp * [1 + 1/(Ti*s) + Td*s])
KP_CONTROLLER = 2.2  # Proportional Gain (Kp)
TI_INTEGRAL = 1097.29 # Integral Time Constant (Ti).
TD_DERIVATIVE = 203.54  # Derivative Time Constant (Td).

# ==========================================================

def closed_loop_bode_performance_visualizer(tau1, tau2, k_proc, Kp, Ti, Td):
    """
    Generates a Bode Plot for the Closed-Loop Transfer Function T(s) = L(s) / (1 + L(s)).
    Focuses on performance metrics: Bandwidth and Resonance Peak (M_p).
    """
    
    # --- 1. Define Open-Loop L(s) Coefficients ---
    
    # Plant G(s)
    num_G = [k_proc]
    den_G = [tau1 * tau2, tau1 + tau2, 1]

    # Controller C(s)
    if Ti == 0 and Td == 0:
        num_C = [Kp]
        den_C = [1]
    elif Ti == 0:
        num_C = [Kp * Td, Kp]
        den_C = [1]
    elif Td == 0:
        num_C = [Kp * Ti, Kp]
        den_C = [Ti, 0]
    else:
        num_C = [Kp * Ti * Td, Kp * Ti, Kp]
        den_C = [Ti, 0]
        
    # Open-Loop L(s) = C(s) * G(s)
    num_L = signal.convolve(num_C, num_G)
    den_L = signal.convolve(den_C, den_G)
    
    # --- 2. Calculate Closed-Loop Transfer Function T(s) ---
    
    len_n = len(num_L)
    len_d = len(den_L)
    
    # Pad the smaller array with leading zeros
    if len_n > len_d:
        den_L_padded = np.pad(den_L, (len_n - len_d, 0), 'constant')
        num_L_padded = num_L
    elif len_d > len_n:
        num_L_padded = np.pad(num_L, (len_d - len_n, 0), 'constant')
        den_L_padded = den_L
    else:
        num_L_padded = num_L
        den_L_padded = den_L
        
    # Closed-Loop Denominator (Characteristic Equation: D_L + N_L)
    den_T = den_L_padded + num_L_padded
    
    # Closed-Loop Numerator (N_L)
    num_T = num_L_padded  
    
    T_closed_loop = signal.TransferFunction(num_T, den_T)
    
    # --- 3. Calculate Frequency Response ---
    w = np.logspace(-6, 0, 1000)  
    w, mag, phase = signal.bode(T_closed_loop, w) # mag is in dB
    
    # --- 4. Performance Metrics ---
    
    # 4.1. Resonance Peak (M_p) and Resonant Frequency (w_r)
    M_p_dB = np.max(mag)
    idx_res = np.argmax(mag)
    w_res = w[idx_res]

    # 4.2. Bandwidth Frequency (w_bw): where Magnitude drops to -3 dB
    def find_nearest_idx(array, value):
        # Find the index of the value closest to the target value
        return (np.abs(array - value)).argmin()
        
    idx_bw = find_nearest_idx(mag, -3.0)
    w_bw = w[idx_bw]
    
    # --- 5. Visualization ---
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.1)  

    # --- MAGNITUDE PLOT (T(jw)) ---
    ax1.semilogx(w, mag, color='g', linewidth=2.5, label='$|T(j\omega)|$')
    ax1.axhline(0, color='r', linestyle='-', linewidth=1, alpha=0.7)
    ax1.axhline(-3, color='darkorange', linestyle='--', linewidth=1.5, alpha=0.8, label='-3 dB Bandwidth')
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title(f'Closed-Loop Bode Plot $T(j\omega)$ - Performance Analysis', fontsize=16)
    ax1.grid(True, which="both", linestyle=':', alpha=0.6)
    
    # Set Y-limits based on data but constrained (e.g., M_p + 5dB, and -20dB)
    y_max = np.ceil(M_p_dB + 5) if M_p_dB > 5 else 5
    y_min = -20
    ax1.set_ylim(y_min, y_max)
    
    # Plot/Mark Bandwidth point (darkorange circle)
    ax1.axvline(w_bw, color='darkorange', linestyle='--', alpha=0.8, linewidth=1.5)
    ax1.plot(w_bw, -3, marker='o', color='darkorange', markersize=8)
    
    # Plot/Mark Resonance Peak (M_p) (red cross)
    ax1.plot(w_res, M_p_dB, marker='x', color='r', markersize=10, markeredgewidth=2)
    
    # --- Performance Metrics Legend (Bottom Left) - MODIFIED LOCATION ---
    # \circ (circle) for Bandwidth, \times (cross) for Resonance Peak
    legend_text = (
        f"Controller: PID ($K_p={Kp:.1f}, T_i={Ti:.0f}, T_d={Td:.1f}$)\n"
        f"\n"
        f"$\\circ$ Bandwidth ($\omega_{{BW}}$): {w_bw:.4f} rad/s\n"
        f"$\\times$ Resonance Peak ($M_p$): {M_p_dB:.2f} dB\n"
        f"$\\times$ Resonant Freq. ($\omega_{{r}}$): {w_res:.4f} rad/s"
    )
    
    # MODIFIED: Coordinates changed from (0.98, 0.90) to (0.02, 0.05) 
    # Alignment changed from 'top'/'right' to 'bottom'/'left'
    ax1.text(0.02, 0.05, legend_text, 
             transform=ax1.transAxes, 
             fontsize=11, 
             verticalalignment='bottom', # Anchor point is the bottom of the text box
             horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkgray', boxstyle='round,pad=1.0'))


    # --- PHASE PLOT ---
    ax2.semilogx(w, phase, color='g', linewidth=2.5, label='Phase of $T(j\omega)$')
    ax2.set_ylabel('Phase (degrees)', fontsize=12)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=14)
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    
    # --- Console Summary (Kept for reference) ---
    print("--- Closed-Loop Performance Analysis ---")
    print(f"Bandwidth Frequency (w_bw): {w_bw:.4f} rad/s")
    print(f"Resonance Peak (M_p): {M_p_dB:.2f} dB")
    print(f"Resonant Frequency (w_r): {w_res:.4f} rad/s")
    
    plt.tight_layout()
    plt.savefig('closed_loop_bode_plot_bottom_left.png')
    # plt.show() # Disabled for sandbox

# Run Analysis
closed_loop_bode_performance_visualizer(TAU_1, TAU_2, K_PROC, KP_CONTROLLER, TI_INTEGRAL, TD_DERIVATIVE)
