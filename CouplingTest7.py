import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. System Configuration
# ==========================================
# --- Process Models (Standard Oven/Zone Physics) ---
K1, tau11, tau12 = 1.0, 309.2, 1168.8
K2, tau21, tau22 = 1.0, 341.75, 1125.71

# --- Cross-Coupling Dynamics (Advanced - Technically Superior) ---
TAU_CROSS_SLOW = 120.0 
TAU_CROSS_FAST = 10.0  

# --- Simulation Settings ---
dt = 2.0
t_end = 20000
baseline_temp = 80.0
sp1, sp2 = 90.0, 70.0

# --- Controller Definitions (Restoring Code 1's Comparative Logic) ---
#PID_FAST = {'Kp': 4.6, 'Ti': 506, 'Td': 126}
#PID_SLOW = {'Kp': 2.4, 'Ti': 2019, 'Td': 504}
PID_FAST = {'Kp': 2.2, 'Ti': 1168.85, 'Td': 157.37}
PID_SLOW = {'Kp': 2.2, 'Ti': 1478.03, 'Td': 0.00}

# ==========================================
# 2. Robust PID & Process Classes
# ==========================================
class PID_Robust:
    def __init__(self, Kp, Ti, Td, dt, baseline=0.0):
        self.Kp, self.Ti, self.Td, self.dt = Kp, Ti, Td, dt
        self.integral = (baseline * Ti) / Kp 
        self.prev_err = 0
        
    def compute(self, sp, pv):
        err = sp - pv
        P = self.Kp * err
        D = self.Kp * self.Td * (err - self.prev_err) / self.dt
        
        tentative_integral = self.integral + err * self.dt
        I_tentative = (self.Kp / self.Ti) * tentative_integral
        v = P + I_tentative + D
        
        # Anti-windup clamping
        if v >= 100.0:
            out = 100.0
            if err < 0: self.integral = tentative_integral
        elif v <= 0.0:
            out = 0.0
            if err > 0: self.integral = tentative_integral
        else:
            out = v
            self.integral = tentative_integral
            
        self.prev_err = err
        return out

class SOPDT:
    def __init__(self, K, t1, t2, init_val=0.0):
        self.K, self.t1, self.t2 = K, t1, t2
        self.x1 = init_val
        self.x2 = init_val
        
    def step(self, u):
        dx1 = (-self.x1 + self.K * u) / self.t1
        self.x1 += dx1 * dt
        dx2 = (-self.x2 + self.x1) / self.t2
        self.x2 += dx2 * dt
        return self.x2

# ==========================================
# 3. Simulation Logic
# ==========================================
def run_simulation(alpha, params1, params2):
    # params1 -> Valve 1, params2 -> Valve 2
    valve1 = PID_Robust(**params1, dt=dt, baseline=baseline_temp)
    valve2 = PID_Robust(**params2, dt=dt, baseline=baseline_temp)
    
    g11 = SOPDT(K1, tau11, tau12, baseline_temp)
    g22 = SOPDT(K2, tau21, tau22, baseline_temp)
    # Cross coupling with fast/slow dynamics
    g12 = SOPDT(alpha, TAU_CROSS_SLOW, TAU_CROSS_FAST, 0.0) 
    g21 = SOPDT(alpha, TAU_CROSS_SLOW, TAU_CROSS_FAST, 0.0)
    
    pv1, pv2 = baseline_temp, baseline_temp
    
    # History storage
    h_temp = []
    h_valve = []
    
    steps = int(t_end / dt)
    for _ in range(steps):
        q1 = valve1.compute(sp1, pv1)
        q2 = valve2.compute(sp2, pv2)
        
        # Process Physics
        pv1 = g11.step(q1) + g21.step(q2 - baseline_temp)
        pv2 = g22.step(q2) + g12.step(q1 - baseline_temp)
        
        # Store Data
        h_temp.append([pv1, pv2])
        h_valve.append([q1, q2])
        
    return np.array(h_temp), np.array(h_valve)

# ==========================================
# 4. Detailed Stability Analysis (CRITICAL VALUES CONFIRMED)
# ==========================================
def check_stability_detailed(h_temp, h_valve, tail=1000):
    
    results = [] 
    # CONFIRMED: MV required to hold 80.0 deg C steady-state
    MV_EXPECTED_MIDLINE = 80.0 
    # CONFIRMED: Strain threshold requested by user
    LC_STRAIN_THRESHOLD = 19.9 
    
    for i in range(2): # Zone 0 (i=0) and Zone 1 (i=1)
        pv_hist = h_temp[-tail:, i]
        mv_hist = h_valve[-tail:, i]
        
        temp_var = np.std(pv_hist)
        valve_var = np.std(mv_hist)
        avg_mv = np.mean(mv_hist)
        
        status = "OK"
        
        # 1. SATURATED (Hard Limit)
        if avg_mv > 99.0 or avg_mv < 1.0:
            status = "SATURATED"
        
        # 2. FIGHTING (High Variance/Oscillation)
        elif valve_var > 1.0:
            status = "FIGHTING"
            
        # 3. OSCILLATING (Temperature Instability)
        elif temp_var > 0.1:
            status = "OSCILLATING"
            
        # 4. STRAINED (Stable but Mean MV is outside [60.1%, 99.9%])
        elif abs(avg_mv - MV_EXPECTED_MIDLINE) > LC_STRAIN_THRESHOLD:
            status = "STRAINED"

        results.append({
            'status': status,
            'v_var': valve_var,
            't_var': temp_var,
            'avg_mv': avg_mv 
        })
        
    return results[0], results[1]

# ==========================================
# 5. The Sweep (Comparing Config A vs B)
# ==========================================
results_data_A = [] 
results_data_B = [] 
capture_data = None 

for alpha in np.arange(0.1, 1.0, 0.1):
    rga = 1 / (1 - alpha**2)
    
    # Configuration A: Fast PID on Zone 1, Slow PID on Zone 2
    t_A, v_A = run_simulation(alpha, PID_FAST, PID_SLOW)
    P1_A, P2_A = check_stability_detailed(t_A, v_A)
    
    # Configuration B: Swapped (Slow PID on Zone 1, Fast PID on Zone 2)
    t_B, v_B = run_simulation(alpha, PID_SLOW, PID_FAST)
    P1_B, P2_B = check_stability_detailed(t_B, v_B)
    
    # Log Data for the Tables (using PID 1 and PID 2)
    results_data_A.append([
        f"{alpha:.1f}", f"{rga:.2f}", P1_A['status'], f"{P1_A['v_var']:.2f}", f"{P1_A['t_var']:.2f}", 
        P2_A['status'], f"{P2_A['v_var']:.2f}", f"{P2_A['t_var']:.2f}"
    ])
    results_data_B.append([
        f"{alpha:.1f}", f"{rga:.2f}", P1_B['status'], f"{P1_B['v_var']:.2f}", f"{P1_B['t_var']:.2f}", 
        P2_B['status'], f"{P2_B['v_var']:.2f}", f"{P2_B['t_var']:.2f}"
    ])
    
    # Logic to capture the FIRST overall failure (Config A preferred if both fail)
    if capture_data is None:
        if P1_A['status'] != "OK" or P2_A['status'] != "OK":
            # REMOVED fast/slow nomenclature
            capture_data = (t_A, v_A, alpha, f"Config A (PID 1: {PID_FAST['Kp']:.1f}Kp, PID 2: {PID_SLOW['Kp']:.1f}Kp)") 
        elif P1_B['status'] != "OK" or P2_B['status'] != "OK":
            # REMOVED fast/slow nomenclature
            capture_data = (t_B, v_B, alpha, f"Config B (PID 1: {PID_SLOW['Kp']:.1f}Kp, PID 2: {PID_FAST['Kp']:.1f}Kp)") 

# Default fallback if everything is stable
if capture_data is None:
    t_A, v_A = run_simulation(0.9, PID_FAST, PID_SLOW)
    capture_data = (t_A, v_A, 0.9, "Stable (Config A)")

# ==========================================
# 6. Visualization (Two Tables + Plot)
# ==========================================
t_hist, v_hist, p_alpha, p_title = capture_data
time_ax = np.linspace(0, t_end, len(t_hist))

# --- Parameter String for Title ---
# REMOVED fast/slow nomenclature
param_str = (
    f"Process: SOPDT $\\tau_1={tau11:.0f},\\tau_2={tau12:.0f}$ | Coupling: $\\tau_{{slow}}={TAU_CROSS_SLOW:.0f},\\tau_{{fast}}={TAU_CROSS_FAST:.0f}$"
    f"\nPID 1: $K_p={PID_FAST['Kp']:.1f}, \\tau_I={PID_FAST['Ti']:.0f}$ | PID 2: $K_p={PID_SLOW['Kp']:.1f}, \\tau_I={PID_SLOW['Ti']:.0f}$"
)

# Figure Size maintained
fig = plt.figure(figsize=(14, 15)) 
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])

# --- Row 1: Table for Configuration A ---
ax_tbl_A = fig.add_subplot(gs[0])
ax_tbl_A.axis('off'); ax_tbl_A.axis('tight')
# REMOVED fast/slow nomenclature from column labels
cols = ("Alpha", "RGA", "PID 1 Status", "P1 MV Var", "P1 Temp Var", 
        "PID 2 Status", "P2 MV Var", "P2 Temp Var")
tbl_A = ax_tbl_A.table(cellText=results_data_A, colLabels=cols, loc='center', cellLoc='center')
tbl_A.auto_set_font_size(False); tbl_A.set_fontsize(8)
tbl_A.scale(1, 1.4)
# REMOVED fast/slow nomenclature from title
ax_tbl_A.set_title(f"Stability Sweep: Configuration A - Standard Assignment", fontweight='bold')

# --- Row 2: Table for Configuration B ---
ax_tbl_B = fig.add_subplot(gs[1])
ax_tbl_B.axis('off'); ax_tbl_B.axis('tight')
# REMOVED fast/slow nomenclature from column labels
cols_B = ("Alpha", "RGA", "PID 1 Status", "P1 MV Var", "P1 Temp Var", 
          "PID 2 Status", "P2 MV Var", "P2 Temp Var")
tbl_B = ax_tbl_B.table(cellText=results_data_B, colLabels=cols_B, loc='center', cellLoc='center')
tbl_B.auto_set_font_size(False); tbl_B.set_fontsize(8)
tbl_B.scale(1, 1.4)
# UPDATED TITLE as requested
ax_tbl_B.set_title(f"Stability Sweep: Configuration B - Controller Swap", fontweight='bold')

# --- Row 3: The Failure Plot (Includes requested text) ---
ax_plt = fig.add_subplot(gs[2])
ax_plt.plot(time_ax, t_hist[:, 0], 'r-', linewidth=1.5, label='Temp 1 (PV)')
ax_plt.plot(time_ax, t_hist[:, 1], 'b-', linewidth=1.5, label='Temp 2 (PV)')
ax_plt.plot(time_ax, v_hist[:, 0], 'r--', alpha=0.6, label='Valve 1 (MV)')
ax_plt.plot(time_ax, v_hist[:, 1], 'b--', alpha=0.6, label='Valve 2 (MV)')

# Title spacing adjusted in the previous step (pad=40, y=1.03)
ax_plt.set_title(f"Dynamic Analysis at Stability Limit (Alpha = {p_alpha:.2f})", pad=40, fontweight='bold')
ax_plt.text(0.5, 1.03, param_str, transform=ax_plt.transAxes, ha='center', fontsize=9, color='gray') 

ax_plt.set_xlabel("Time (s)")
ax_plt.set_ylabel("Temp (C) / Valve Position (%)")
ax_plt.legend(loc='upper right')
ax_plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
