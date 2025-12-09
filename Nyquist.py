import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

# ==========================================================
# --- USER EDITABLE VARIABLES ---
# ==========================================================
# Plant Parameters
TAU_1 = 309.2       # Radiator Lag (τ₁)
TAU_2 = 1168.8      # Convection/Room Lag (τ₂)
K_PROC = 51.88      # Process Gain (K_p)

# Controller Parameters (PID)
KP_CONTROLLER = 2.2      # Proportional Gain (Kp)
TI_INTEGRAL = 1097.29    # Integral Time Constant (Ti)
TD_DERIVATIVE = 203.54   # Derivative Time Constant (Td)
# ==========================================================

def nyquist_stability_analysis(Kp, Ti, Td, K_p, tau1, tau2):
    """
    Performs Nyquist analysis for a PID controlled 2nd order system.
    Plots the Open Loop Transfer Function L(jω) in the complex plane.
    Includes correct visualization of Gain and Phase Margins.
    """
    
    print(f"\n--- Nyquist Stability Analysis ---")
    
    # --- 1. Define Frequency Range (Logarithmic scale) ---
    # Avoid 0 to prevent division by zero in the integral term
    omega = np.logspace(-4, 1, 2000) 
    s = 1j * omega

    # --- 2. Calculate Transfer Functions ---
    # Plant Gp(s)
    Gp = K_p / ((tau1 * s + 1) * (tau2 * s + 1))
    
    # Controller Gc(s) = Kp * (1 + 1/(Ti*s) + Td*s)
    Gc = Kp * (1 + (1 / (Ti * s)) + (Td * s))
    
    # Open Loop Transfer Function L(s) = Gc(s) * Gp(s)
    L = Gc * Gp
    
    # --- 3. Extract Real and Imaginary parts ---
    real_L = np.real(L)
    imag_L = np.imag(L)

    # --- 4. Calculate Margins ---
    mag_L = np.abs(L)
    phase_L = np.angle(L, deg=True)

    # --- Phase Margin (PM) calculation ---
    # Find Gain Crossover Frequency (where magnitude is closest to 1)
    idx_gc = np.argmin(np.abs(mag_L - 1.0)) 
    pm_loc = L[idx_gc]
    
    # Phase at crossover
    phase_at_gc = np.angle(pm_loc, deg=True)
    phase_margin = 180 + phase_at_gc 

    # --- Gain Margin (GM) calculation ---
    # Find Phase Crossover Frequency (where phase is -180 deg or Imag part is 0)
    
    # Filter for points where the curve is close to the negative real axis
    neg_real_indices = np.where((real_L < -0.1) & (real_L > -2.5))[0] 
    if len(neg_real_indices) > 0:
        # Of these, find where Imag is closest to 0
        idx_pc_subset = np.argmin(np.abs(imag_L[neg_real_indices]))
        idx_pc = neg_real_indices[idx_pc_subset]
        gm_loc = np.real(L[idx_pc])
        
        # Calculate GM (Distance from -1 to gm_loc)
        gain_margin_lin = 1.0 / np.abs(gm_loc)
        gain_margin_db = 20 * np.log10(gain_margin_lin)
    else:
        gm_loc = None
        gain_margin_db = float('inf')


    # --- 5. Visualization ---
    plt.figure(figsize=(12, 9))
    
    # Plot Limits - focus on the critical point (-1, 0)
    plt.xlim(-2.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # Axes
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # Critical Point (-1 + 0j)
    plt.plot(-1, 0, 'rx', markersize=14, markeredgewidth=3, zorder=5, label='Critical Point (-1+0j)')

    # Unit Circle (Magnitude = 1) - Visual reference for Phase Margin
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.4, label='Unit Circle (|L|=1)')
    plt.gca().add_artist(unit_circle)

    # Plot Nyquist Curve (Positive Frequencies)
    plt.plot(real_L, imag_L, color='b', linewidth=2, label='L(j$\omega$) $\omega$>0')
    
    # Plot Mirror (Negative Frequencies) - Dashed
    plt.plot(real_L, -imag_L, color='b', linewidth=1.5, linestyle=':', alpha=0.5, label='L(j$\omega$) $\omega$<0')

    # Arrow indicating direction of increasing frequency
    arrow_idx = len(real_L) // 4
    plt.arrow(real_L[arrow_idx], imag_L[arrow_idx], 
              real_L[arrow_idx+1]-real_L[arrow_idx], imag_L[arrow_idx+1]-imag_L[arrow_idx],
              shape='full', lw=0, length_includes_head=True, head_width=0.08, color='b', zorder=4)

    # --- Draw Margins ---
    
    # 1. Gain Margin Visualization (Gold)
    if gm_loc is not None and -2.5 < gm_loc < 0:
        # Line from -1 to phase crossover point
        plt.plot([gm_loc, -1], [0, 0], color='gold', linewidth=3, solid_capstyle='butt', alpha=0.7)
        plt.text((gm_loc - 1)/2, 0.1, f'GM = {gain_margin_db:.1f} dB', color='gold', 
                 fontweight='bold', ha='center', fontsize=12)

    # 2. Phase Margin Visualization (Orange)
    if phase_margin > 0 and np.abs(phase_at_gc) > 0.01:
        # Line from origin to gain crossover
        plt.plot([0, np.real(pm_loc)], [0, np.imag(pm_loc)], color='darkorange', linewidth=2, linestyle='--')
        
        # Arc radius for visualization
        arc_radius = 0.5 

        # Draw the arc from the negative real axis (180 deg) to the crossover point (phase_at_gc)
        # We use a rotation angle of 0 and define the start/end angles relative to the positive x-axis
        
        # We need the arc to go from -180 degrees to phase_at_gc.
        # Since phase_at_gc is typically negative (e.g., -167.9), we use the smaller angle first.
        theta1 = -180 # Start at the negative real axis
        theta2 = phase_at_gc # End at the crossover point angle
        
        # Matplotlib Arc needs theta1 < theta2 usually, so we swap them if needed
        # Or more simply, let's use the actual PM angle relative to the negative real axis.
        arc = Arc((0, 0), arc_radius * 2, arc_radius * 2, angle=0, 
                  theta1=theta2, theta2=theta1, # Draw from crossover angle (-167.9) to -180
                  color='darkorange', linewidth=3, alpha=0.7)
        plt.gca().add_patch(arc)
        
        # Text placement (Adjusted to be near the arc)
        text_angle = (phase_at_gc + (-180)) / 2 # Midpoint of the PM arc
        text_x = arc_radius * 1.1 * np.cos(np.deg2rad(text_angle))
        text_y = arc_radius * 1.1 * np.sin(np.deg2rad(text_angle))
        plt.text(text_x, text_y, f'PM = {phase_margin:.1f}°', color='darkorange', 
                 fontweight='bold', fontsize=12, ha='center', va='center')


    # --- Labels & Legends ---
    plt.title(f'Nyquist Plot Stability Analysis\n($K_p={Kp}, T_i={Ti}, T_d={Td}$)', fontsize=16)
    plt.xlabel('Real Axis', fontsize=14)
    plt.ylabel('Imaginary Axis', fontsize=14)

    # Custom Legend
    custom_lines = [
        Line2D([0], [0], color='b', lw=2),
        Line2D([0], [0], color='r', marker='x', lw=0, ms=14, markeredgewidth=3),
        Line2D([0], [0], color='gray', lw=1.5, ls='--'),
        Line2D([0], [0], color='gold', lw=3, alpha=0.7),
        Line2D([0], [0], color='darkorange', lw=3, alpha=0.7),
    ]
    plt.legend(custom_lines, 
               ['Nyquist Curve L(j$\omega$)', 'Critical Point (-1, 0)', 'Unit Circle', 'Gain Margin (GM)', 'Phase Margin (PM)'], 
               loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print("-" * 40)
    print(f"Gain Margin (GM): {gain_margin_db:.2f} dB")
    print(f"Phase Margin (PM): {phase_margin:.2f} degrees")
    print("-" * 40)
    
    if gain_margin_db > 0 and phase_margin > 0:
        print("RESULT: System is STABLE (Positive margins)")
    else:
        print("RESULT: System is UNSTABLE or MARGINALLY STABLE")

# Run Analysis
nyquist_stability_analysis(KP_CONTROLLER, TI_INTEGRAL, TD_DERIVATIVE, K_PROC, TAU_1, TAU_2)
