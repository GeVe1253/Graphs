import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================================
# --- USER EDITABLE VARIABLES ---
# ==========================================================
TAU_OIL = 309.2   # Radiator Lag (τ₁)
TAU_AIR = 1168.8  # Convection/Room Lag (τ₂)
K_GAIN = 2.0      # Controller Gain (K)
# ==========================================================

def bode_stability_visualizer(K_controller, tau_oil, tau_air):
    """
    Generates a Bode Plot for the Open-Loop Transfer Function L(s) = K * G(s).
    Visualizes Gain Margin (GM) and Phase Margin (PM) for stability analysis.
    Designed for presentation slides (Horizontal Aspect Ratio).
    """
    
    # --- 1. Define Open-Loop System L(s) ---
    # L(s) = K / [ (tau_oil*s + 1) * (tau_air*s + 1) ]
    # Denominator = (tau_oil * tau_air)s^2 + (tau_oil + tau_air)s + 1
    
    num = [K_controller]
    den = [tau_oil * tau_air, tau_oil + tau_air, 1]
    
    # Create the system
    sys = signal.TransferFunction(num, den)
    
    # --- 2. Calculate Frequency Response ---
    # Generate frequencies (log scale): from 10^-5 to 10^-1 rad/s
    w = np.logspace(-5, -1, 1000) 
    w, mag, phase = signal.bode(sys, w)
    
    # --- 3. Calculate Stability Margins ---
    # Find Gain Crossover Frequency (w_gc): where Magnitude is approx 0 dB
    # Find Phase Crossover Frequency (w_pc): where Phase is approx -180 deg
    
    # Helper to find nearest index
    def find_nearest_idx(array, value):
        return (np.abs(array - value)).argmin()

    # Gain Crossover (0 dB)
    idx_gc = find_nearest_idx(mag, 0)
    w_gc = w[idx_gc]
    pm_at_gc = 180 + phase[idx_gc] # Phase Margin
    
    # Phase Crossover (-180 deg)
    idx_pc = find_nearest_idx(phase, -180)
    w_pc = w[idx_pc]
    gm_at_pc = -mag[idx_pc]        # Gain Margin
    
    # --- 4. Visualization (Presentation Style) ---
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.1) # Reduce space between plots

    # --- MAGNITUDE PLOT ---
    ax1.semilogx(w, mag, color='b', linewidth=2.5, label=f'Magnitude (K={K_controller})')
    ax1.axhline(0, color='r', linestyle='-', linewidth=1, alpha=0.7) # 0 dB line
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title(f'Bode Plot Stability Analysis ($K={K_controller}$)', fontsize=16)
    ax1.grid(True, which="both", linestyle=':', alpha=0.6)
    
    # Annotate Phase Margin on Magnitude Plot (Vertical line at w_gc)
    # We draw a line to indicate where we are looking for the PM
    ax1.axvline(w_gc, color='darkorange', linestyle='--', alpha=0.8, linewidth=1.5)
    ax1.text(w_gc, 5, '$\omega_{gc}$', color='darkorange', ha='center', va='bottom', fontsize=12)

    # Annotate Gain Margin (if valid)
    # For a simple 2nd order system, phase might never cross -180, so we check
    if phase.min() > -180:
        gm_text = "Infinite GM (Phase > -180°)"
    else:
        # Draw the GM measure
        ax1.axvline(w_pc, color='gold', linestyle='--', alpha=0.8, linewidth=1.5)
        ax1.plot([w_pc, w_pc], [0, -gm_at_pc], color='k', linewidth=2, solid_capstyle='butt') # The visual bar
        gm_text = f"Gain Margin: {gm_at_pc:.2f} dB"
        ax1.text(w_pc, -gm_at_pc - 5, gm_text, color='gold', fontsize=12, ha='center')

    # --- PHASE PLOT ---
    ax2.semilogx(w, phase, color='b', linewidth=2.5, label='Phase')
    ax2.axhline(-180, color='r', linestyle='-', linewidth=1, alpha=0.7) # -180 deg line
    ax2.set_ylabel('Phase (degrees)', fontsize=12)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=14)
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    
    # Annotate Phase Margin
    ax2.axvline(w_gc, color='darkorange', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Draw arrow/line for PM
    current_phase = phase[idx_gc]
    ax2.plot([w_gc, w_gc], [-180, current_phase], color='darkorange', linewidth=2, solid_capstyle='butt')
    
    # Text for PM
    ax2.annotate(f'Phase Margin: {pm_at_gc:.2f}°', 
                 xy=(w_gc, (current_phase + -180)/2), 
                 xytext=(w_gc * 1.5, -140),
                 arrowprops=dict(facecolor='darkorange', arrowstyle='->', color='darkorange'),
                 fontsize=12, color='darkorange', fontweight='bold')

    # Add text box with summary
    summary_text = (
        f"Analysis Results:\n"
        f"----------------\n"
        f"$\omega_{{gc}}$: {w_gc:.4f} rad/s\n"
        f"PM: {pm_at_gc:.2f}°\n"
        f"{gm_text}"
    )
    
    # Place text box in a nice spot (usually lower left of magnitude or lower left of phase)
    # Using figure coordinates to ensure it doesn't hide data
    fig.text(0.15, 0.55, summary_text, fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))

    print(f"--- Bode Analysis (K={K_controller}) ---")
    print(f"Gain Crossover Freq: {w_gc:.5f} rad/s")
    print(f"Phase Margin: {pm_at_gc:.2f} degrees")
    if phase.min() <= -180:
        print(f"Gain Margin: {gm_at_pc:.2f} dB")
    else:
        print("Gain Margin: Infinite")

    plt.tight_layout()
    plt.show()

# Run Analysis
bode_stability_visualizer(K_GAIN, TAU_OIL, TAU_AIR)
