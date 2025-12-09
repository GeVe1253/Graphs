import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# --- USER EDITABLE VARIABLES ---
# ==========================================================
TAU_OIL = 309.2   # Radiator Lag (τ₁)
TAU_AIR = 1168.8  # Convection/Room Lag (τ₂) - The dominant one
K_GAIN = 2.0      # Controller Gain (K)
# ==========================================================

def hurwitz_pole_visualization(K_controller, tau_oil, tau_air):
    """
    Analyzes stability by calculating closed-loop poles and plotting them on the s-plane.
    
    Characteristic Equation: a2*s^2 + a1*s + a0 = 0
    """
    
    # --- 1. Calculate Characteristic Equation Coefficients ---
    a2 = tau_oil * tau_air
    a1 = tau_oil + tau_air
    a0 = 1 + K_controller
    
    print(f"\n--- Hurwitz Visualization Analysis (K={K_controller:.1f}) ---")
    print(f"Polynomial: {a2:.1f}s² + {a1:.1f}s + {a0:.1f} = 0")
    
    # --- 2. Calculate Closed-Loop Poles (Roots) ---
    system_poles = np.roots([a2, a1, a0])
    
    # --- 3. Visualization on the S-Plane ---
    
    plt.figure(figsize=(10, 10)) # Retaining the larger size for clarity
    
    # Draw the axes (Imaginary and Real) with different colors
    # Imaginary Axis (jω, vertical) - Often highlighted as the stability boundary
    plt.axvline(0, color='r', linestyle='-', linewidth=1.5, label='Imaginary Axis (Stability Boundary)')
    # Real Axis (σ, horizontal)
    plt.axhline(0, color='g', linestyle='-', linewidth=1.5, label='Real Axis')
    
    # Define the range for the plot
    max_pole_mag = np.max(np.abs(np.real(system_poles))) * 1.5
    plt.xlim(-max_pole_mag, max_pole_mag / 4) # Focus on the LHP
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
             markersize=12, markeredgewidth=3, color=plot_color, label='Closed-Loop Poles')
    
    # Annotate the poles (using offsets for better spacing)
    for r, i in zip(pole_real, pole_imag):
        plt.annotate(f'p = {r:.4f} + j{i:.4f}', (r, i), textcoords="offset points", 
                     xytext=(10, 5), ha='left', fontsize=10)

    plt.title(f'$s$-Plane Pole Plot for Radiator System ($K={K_controller}$)', fontsize=14)
    plt.xlabel('Real Axis ($\sigma$)', fontsize=12)
    plt.ylabel('Imaginary Axis ($j\omega$)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    print("\n--- Pole Locations ---")
    for i, p in enumerate(system_poles):
        print(f"Pole p{i+1}: {p}")

# Run Analysis
hurwitz_pole_visualization(K_GAIN, TAU_OIL, TAU_AIR)
