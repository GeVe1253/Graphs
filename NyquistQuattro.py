import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

# ==========================================================
# --- USER EDITABLE VARIABLES ---
# ==========================================================
# Nominal Plant Parameters
TAU_1_NOM = 309.2       
TAU_2_NOM = 1168.8      
K_PROC = 51.88      

# Controller Parameters (PID)
KP_CONTROLLER = 2.2  
TI_INTEGRAL = 1097.29 
TD_DERIVATIVE = 203.54 

# Robustness Tolerance
TOL = 0.20 
# ==========================================================

def get_nyquist_data(tau1, tau2, Kp, Ti, Td, K_p, omega):
    s = 1j * omega
    Gp = K_p / ((tau1 * s + 1) * (tau2 * s + 1))
    Gc = Kp * (1 + (1 / (Ti * s)) + (Td * s))
    L = Gc * Gp
    return L

def calculate_margins(L):
    mag_L = np.abs(L)
    real_L = np.real(L)
    imag_L = np.imag(L)
    
    idx_gc = np.argmin(np.abs(mag_L - 1.0))
    pm_loc = L[idx_gc]
    phase_at_gc = np.angle(pm_loc, deg=True)
    pm = 180 + phase_at_gc
    
    neg_real_idx = np.where((real_L < -0.1) & (real_L > -5.0))[0]
    if len(neg_real_idx) > 0:
        idx_pc_subset = np.argmin(np.abs(imag_L[neg_real_idx]))
        idx_pc = neg_real_idx[idx_pc_subset]
        gm_loc = np.real(L[idx_pc])
        gm_db = 20 * np.log10(1.0 / np.abs(gm_loc))
    else:
        gm_loc = None
        gm_db = float('inf')
        
    return pm, pm_loc, phase_at_gc, gm_db, gm_loc

def plot_nyquist_subplot(ax, title, L_var, L_nom, pm_data):
    pm, pm_loc, phase_at_gc, gm_db, gm_loc = pm_data
    
    # Square aspect ratio for the data area
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=3, zorder=5)
    
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(unit_circle)

    ax.plot(np.real(L_nom), np.imag(L_nom), color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.plot(np.real(L_var), np.imag(L_var), color='b', linewidth=2)
    
    if gm_loc is not None and -3.0 < gm_loc < 0:
        ax.plot([gm_loc, -1], [0, 0], color='gold', linewidth=3, solid_capstyle='butt', alpha=0.8)
        ax.text((gm_loc - 1)/2, 0.12, f'GM={gm_db:.1f}dB', color='gold', ha='center', fontsize=10)

    if pm > 0:
        ax.plot([0, np.real(pm_loc)], [0, np.imag(pm_loc)], color='darkorange', linestyle='--', linewidth=1.5)
        arc = Arc((0, 0), 1.0, 1.0, angle=0, theta1=phase_at_gc, theta2=-180, color='darkorange', linewidth=2.5)
        ax.add_patch(arc)
        
        mid_angle = np.deg2rad((phase_at_gc + -180) / 2)
        ax.text(0.7 * np.cos(mid_angle), 0.7 * np.sin(mid_angle), f'PM={pm:.1f}°', color='darkorange', ha='center', fontsize=10)

    ax.set_title(title, fontsize=13)

def generate_four_panel_nyquist():
    omega = np.logspace(-4, 1, 2000)
    L_nom = get_nyquist_data(TAU_1_NOM, TAU_2_NOM, KP_CONTROLLER, TI_INTEGRAL, TD_DERIVATIVE, K_PROC, omega)

    scenarios = [
        (f"Max Lag (+{TOL*100:.0f}% t1, +{TOL*100:.0f}% t2)", 1.0+TOL, 1.0+TOL),
        (f"Min Lag (-{TOL*100:.0f}% t1, -{TOL*100:.0f}% t2)", 1.0-TOL, 1.0-TOL),
        (f"Mismatch A (+{TOL*100:.0f}% t1, -{TOL*100:.0f}% t2)", 1.0+TOL, 1.0-TOL),
        (f"Mismatch B (-{TOL*100:.0f}% t1, +{TOL*100:.0f}% t2)", 1.0-TOL, 1.0+TOL)
    ]

    # Adjusted to 20x14 for a sleek landscape orientation
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for i, (title, m1, m2) in enumerate(scenarios):
        t1_var, t2_var = TAU_1_NOM * m1, TAU_2_NOM * m2
        L_var = get_nyquist_data(t1_var, t2_var, KP_CONTROLLER, TI_INTEGRAL, TD_DERIVATIVE, K_PROC, omega)
        pm_data = calculate_margins(L_var)
        plot_nyquist_subplot(axes[i], title, L_var, L_nom, pm_data)

    fig.suptitle(f"Nyquist Robustness Analysis (4-Corner Test)\n$K_p={KP_CONTROLLER}, T_i={TI_INTEGRAL}, T_d={TD_DERIVATIVE}$", 
                 fontsize=20, y=0.96) # Positioned specifically to hug the top

    custom_legend = [
        Line2D([0], [0], color='b', lw=2, label='Scenario Curve'),
        Line2D([0], [0], color='gray', lw=1.5, ls=':', label='Nominal Reference'),
        Line2D([0], [0], color='r', marker='x', lw=0, ms=12, markeredgewidth=3, label='Critical Point'),
        Line2D([0], [0], color='gold', lw=3, label='Gain Margin'),
        Line2D([0], [0], color='darkorange', lw=2, label='Phase Margin')
    ]
    
    fig.legend(handles=custom_legend, loc='lower center', ncol=5, fontsize=12, frameon=False)
    
    # top=0.88 reduces the gap between suptitle and the first row of axes
    plt.subplots_adjust(top=0.88, bottom=0.10, hspace=0.2, wspace=0.1)
    plt.show()

generate_four_panel_nyquist()
