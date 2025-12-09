import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Needed for custom legend

# ==========================================================
# --- USER EDITABLE VARIABLES ---
# ==========================================================
TAU_OIL = 309.2    # Radiator Lag (τ₁)
TAU_AIR = 1168.8  # Convection/Room Lag (τ₂) - The dominant one
K_GAIN = 2.0      # Controller Gain (K)
# ==========================================================

def hurwitz_pole_visualization_relative_stability(K_controller, tau_oil, tau_air):
    """
    Analyzes stability by calculating closed-loop poles and plotting them on the s-plane,
    including lines to visualize relative stability (decay rate and damping).
    
    Modified for better presentation aspect ratio (4:3 Landscape) and readability.
    """
    
    # --- 1. Calculate Characteristic Equation Coefficients ---
    a2 = tau_oil * tau_air
    a1 = tau_oil + tau_air
    a0 = 1 + K_controller
    
    print(f"\n--- Relative Stability Analysis (K={K_controller:.1f}) ---")
    
    # --- 2. Calculate Closed-Loop Poles (Roots) ---
    system_poles = np.roots([a2, a1, a0])
    
    # --- 3. Visualization on the S-Plane ---
    
    # MODIFICATION 1: Change figsize for a horizontal aspect ratio (e.g., 4:3 landscape)
    plt.figure(figsize=(12, 9)) 
    
    # Draw the axes (Imaginary and Real) with different colors
    plt.axvline(0, color='r', linestyle='-', linewidth=1.5, label='Imaginary Axis (Stability Boundary)')
    plt.axhline(0, color='g', linestyle='-', linewidth=1.5, label='Real Axis')
    
    # Calculate plot limits
    max_pole_mag = np.max(np.abs(system_poles)) * 1.1 # Use magnitude for scale
    
    # Set limits: focus on the LHP
    plt.xlim(-max_pole_mag, max_pole_mag / 4) 
    plt.ylim(-max_pole_mag, max_pole_mag)
    
    # Plot the poles
    pole_real = np.real(system_poles)
    pole_imag = np.imag(system_poles)
    
    # Check final stability via poles
    if np.any(pole_real >= 0):
        plot_color = 'r'
        stability_result = "UNSTABLE (Pole(s) in RHP)"
    else:
        plot_color = 'b'
        stability_result = "STABLE (All Poles in LHP)"

    plt.plot(pole_real, pole_imag, marker='x', linestyle='', 
             markersize=14, markeredgewidth=3, color=plot_color, zorder=3, label='Closed-Loop Poles')
    
    # --- Relative Stability Visualization ---
    for r, i in zip(pole_real, pole_imag):
        # 1. Decay Rate (sigma): Dashed yellow line from pole to Imaginary Axis
        plt.plot([r, 0], [i, i], color='gold', linestyle='--', linewidth=1.5, alpha=0.8) 
        
        # 2. Damping (Angle phi): Dashed orange line from Origin (0,0) to pole
        plt.plot([0, r], [0, i], color='darkorange', linestyle='--', linewidth=1.5, alpha=0.8) 
        
        # Annotate the pole with its value
        plt.annotate(f'p = {r:.4f} + j{i:.4f}', (r, i), textcoords="offset points", 
                     xytext=(10, 5), ha='left', fontsize=12)
        
        # Calculate and annotate Damping Ratio (zeta) for one pole (the conjugate pair has the same zeta)
        if i >= 0 and r < 0:
            sigma = -r  # Decay rate is the positive magnitude of the real part
            omega_n = np.sqrt(r**2 + i**2) # Natural frequency
            
            # Damping ratio calculation: zeta = sigma / omega_n
            # Handle division by zero for origin pole if it occurs, though unlikely for stable system
            zeta = sigma / omega_n if omega_n != 0 else 1.0 
            
            # Annotate Decay Rate (sigma)
            plt.text(r * 0.9, i * 0.7, # Adjusted y-position for Decay text
                     f'Decay $\sigma = {sigma:.4f}$', 
                     fontsize=12, color='gold', ha='right')
            
            # Annotate Damping Ratio (zeta) - Placed under Decay text
            plt.text(r * 0.9, i * 0.5, # Adjusted y-position for Damping text
                     f'Damping $\zeta = {zeta:.2f}$', 
                     fontsize=12, color='darkorange', ha='right')


    # MODIFICATION 2 & 3: Increase font size for title/labels for projection
    plt.title(f'$s$-Plane Pole Plot & Relative Stability ($K={K_controller}$)', fontsize=16)
    plt.xlabel('Real Axis ($\sigma$) - Decay Rate', fontsize=14)
    plt.ylabel('Imaginary Axis ($j\omega$) - Oscillation Frequency', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Custom legend entries for the relative stability lines
    custom_lines = [
        Line2D([0], [0], color='r', lw=1.5),
        Line2D([0], [0], color='g', lw=1.5),
        Line2D([0], [0], color='gold', lw=1.5, ls='--'),
        Line2D([0], [0], color='darkorange', lw=1.5, ls='--'),
        Line2D([0], [0], color='b', marker='x', lw=0, ms=14) # Increased marker size
    ]
    
    # MODIFICATION 4: Change legend location to 'best' or 'upper left' for better visibility
    plt.legend(custom_lines, 
               ['Imaginary Axis', 'Real Axis', 'Decay Rate ($\sigma$)', 'Damping ($\zeta$ Angle)', 'Closed-Loop Poles'], 
               loc='best', fontsize=11) # 'best' will try to find a spot that doesn't overlap data

    # REMOVED: plt.gca().set_aspect('equal', adjustable='box') - Allows horizontal stretching
    
    plt.tight_layout() # Adjust plot to prevent clipping of labels/titles
    plt.show()
    
# Run Analysis
hurwitz_pole_visualization_relative_stability(K_GAIN, TAU_OIL, TAU_AIR)
